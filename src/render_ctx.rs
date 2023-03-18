use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{DeviceOwned},
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::VertexDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
};

use crate::game::Game;

pub trait RenderContextT {
    fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        vertex_shader: &Arc<ShaderModule>,
        fragment_shader: &Arc<ShaderModule>,
    ) -> Self
    where
        Self: std::marker::Sized;

    fn render(
        &self,
        game: &mut Game,
        swapchain: &Arc<vulkano::swapchain::Swapchain>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    );
}

pub struct RenderContext {
    pub pipeline: Arc<GraphicsPipeline>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
}

impl RenderContext {
    pub fn new<T>(
        memory_allocator: &StandardMemoryAllocator,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        vertex_shader: &Arc<ShaderModule>,
        fragment_shader: &Arc<ShaderModule>,
        vertex_input: T,
    ) -> Self
    where
        T: VertexDefinition,
    {
        let dimensions = images[0].dimensions().width_height();

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(memory_allocator, dimensions, Format::D16_UNORM).unwrap(),
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
            .vertex_input_state(vertex_input)
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

        Self {
            pipeline,
            framebuffers,
        }
    }

    pub fn render(
        &self,

        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        callback: impl FnOnce(&RenderContext, &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>),
    ) {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into()), Some(1f32.into())],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index].clone())
                },
                SubpassContents::Inline,
            )
            .unwrap();

        callback(self, builder);

        builder.end_render_pass().unwrap();
    }
}

// impl RenderContext {
//     pub fn new<T>(
//         memory_allocator: &StandardMemoryAllocator,
//         vs: &ShaderModule,
//         fs: &ShaderModule,
//         images: &[Arc<SwapchainImage>],
//         render_pass: Arc<RenderPass>,
//         vertex_input: T,
//     ) -> Self
//     where
//         T: VertexDefinition,
//     {
//         let dimensions = images[0].dimensions().width_height();

//         let depth_buffer = ImageView::new_default(
//             AttachmentImage::transient(memory_allocator, dimensions, Format::D16_UNORM).unwrap(),
//         )
//         .unwrap();

//         let framebuffers = images
//             .iter()
//             .map(|image| {
//                 let view = ImageView::new_default(image.clone()).unwrap();
//                 Framebuffer::new(
//                     render_pass.clone(),
//                     FramebufferCreateInfo {
//                         attachments: vec![view, depth_buffer.clone()],
//                         ..Default::default()
//                     },
//                 )
//                 .unwrap()
//             })
//             .collect::<Vec<_>>();

//         // In the triangle example we use a dynamic viewport, as its a simple example. However in the
//         // teapot example, we recreate the pipelines with a hardcoded viewport instead. This allows the
//         // driver to optimize things, at the cost of slower window resizes.
//         // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
//         let pipeline = GraphicsPipeline::start()
//             .vertex_input_state(vertex_input)
//             .vertex_shader(vs.entry_point("main").unwrap(), ())
//             .input_assembly_state(InputAssemblyState::new())
//             .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
//                 Viewport {
//                     origin: [0.0, 0.0],
//                     dimensions: [dimensions[0] as f32, dimensions[1] as f32],
//                     depth_range: 0.0..1.0,
//                 },
//             ]))
//             .fragment_shader(fs.entry_point("main").unwrap(), ())
//             .depth_stencil_state(DepthStencilState::simple_depth_test())
//             .render_pass(Subpass::from(render_pass, 0).unwrap())
//             .build(memory_allocator.device().clone())
//             .unwrap();

//         Self {
//             pipeline,
//             framebuffers,
//         }
//     }
// }

// trait RenderPipelineBuilder {
//     fn new() -> Self;
//     fn build(&self) -> RenderPipeline;
// }
