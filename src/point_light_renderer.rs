use std::sync::Arc;

use glam::Vec3;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::DeviceOwned,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

use crate::{camera::Camera, game::Game, render_ctx::RenderContext};

pub struct PointLightRenderer {
    pub pipeline: Arc<GraphicsPipeline>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub uniform_buffer: SubbufferAllocator,
}

impl RenderContext for PointLightRenderer {
    fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        vertex_shader: &Arc<vulkano::shader::ShaderModule>,
        fragment_shader: &Arc<vulkano::shader::ShaderModule>,
    ) -> Self {
        let dimensions = images[0].dimensions().width_height();

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(&memory_allocator, dimensions, Format::D16_UNORM).unwrap(),
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // In the triangle example we use a dynamic viewport, as its a simple example. However in the
        // teapot example, we recreate the pipelines with a hardcoded viewport instead. This allows the
        // driver to optimize things, at the cost of slower window resizes.
        // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(VertexInputState::default())
            .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                },
            ]))
            .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(memory_allocator.device().clone())
            .unwrap();

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        Self {
            pipeline,
            framebuffers,
            uniform_buffer,
        }
    }

    fn render(
        &self,
        game: &mut Game,
        swapchain: &Arc<vulkano::swapchain::Swapchain>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        let mut camera = game.world().query::<&mut Camera>().single_mut(game.world());
        let uniform_buffer_subbuffer = {
            let aspect_ratio =
                swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;

            let uniform_data = point_vs::Data {
                projection: camera
                    .calc_perspective_projection(aspect_ratio)
                    .to_cols_array_2d(),
                view: camera
                    .calc_view_direction(Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array_2d(),
                ambientLightColor: [1.0, 1.0, 1.0, 0.2],
                lightPosition: Vec3::new(-1.0, 1.0, -1.0).to_array().into(),
                lightColor: [1.0, 1.0, 1.0, 1.0],
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            );

        builder.draw(6, 1, 0, 0).unwrap();
    }
}

pub mod point_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/point_light.vert",
    }
}

pub mod point_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/point_light.frag",
    }
}
