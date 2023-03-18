use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    image::SwapchainImage,
    memory::allocator::StandardMemoryAllocator,
    render_pass::RenderPass,
    shader::ShaderModule,
};

use crate::game::Game;

pub trait RenderContext {
    fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        vertex_shader: &Arc<ShaderModule>,
        fragment_shader: &Arc<ShaderModule>,
    ) -> Self;

    fn render(
        &self,
        game: &mut Game,
        swapchain: &Arc<vulkano::swapchain::Swapchain>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    );
}
