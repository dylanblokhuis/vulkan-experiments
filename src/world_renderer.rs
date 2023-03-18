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
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};

use crate::{
    camera::Camera,
    game::{Assets, Game},
    game_object::{GameObject, PointLight},
    model::ModelVertex,
    render_ctx::RenderContext,
};

pub struct WorldRenderer {
    pub pipeline: Arc<GraphicsPipeline>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub uniform_buffer: SubbufferAllocator,
}

impl RenderContext for WorldRenderer {
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
            .vertex_input_state(ModelVertex::per_vertex())
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
        let ubo_lights = game
            .world()
            .query::<&PointLight>()
            .iter(game.world())
            .map(|light| {
                let position = light.transform.translation;
                let color = light.color;
                world_vs::PointLight {
                    position: [position.x, position.y, position.z, 1.0],
                    color: [color.x, color.y, color.z, 1.0],
                }
            })
            .collect::<Vec<_>>();

        let ubo_lights_array: [world_vs::PointLight; 3] = ubo_lights.as_slice().try_into().unwrap();

        let uniform_buffer_subbuffer = {
            let aspect_ratio =
                swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
            let camera = game.world().query::<&Camera>().single(game.world());

            let uniform_data = world_vs::Data {
                projection: camera
                    .calc_perspective_projection(aspect_ratio)
                    .to_cols_array_2d(),
                view: camera
                    .calc_view_direction(Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array_2d(),
                inverseView: camera
                    .calc_view_direction_inverse(Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array_2d(),
                ambientLightColor: [1.0, 1.0, 1.0, 0.02], // w is intensity
                numLights: ubo_lights.len() as i32,
                pointLights: ubo_lights_array,
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

        let world = game.world();
        for obj in world.query::<&GameObject>().iter(world) {
            let model_matrix = obj.transform.mat4();
            let push_constants = world_vs::Push {
                modelMatrix: model_matrix.to_cols_array_2d(),
                normalMatrix: obj.transform.normal_matrix().to_cols_array_2d(),
            };

            let model_handle = if let Some(model_handle) = obj.model {
                model_handle
            } else {
                continue;
            };
            let assets = world.resource::<Assets>();
            let model = assets.map.get(model_handle).unwrap();
            // this should be moved to handles
            let vertex_buffer = &model.vertex_buffer;
            let index_buffer = &model.index_buffer.to_owned().unwrap();

            builder
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_index_buffer(index_buffer.clone())
                .push_constants(self.pipeline.layout().clone(), 0, push_constants)
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap();
        }
    }
}

pub mod world_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader.vert",
    }
}

pub mod world_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader.frag",
    }
}
