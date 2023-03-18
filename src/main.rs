use bevy_ecs::system::{Commands, Query, Res};
use bevy_time::Time;
use glam::Vec3;
use model::ModelVertex;
use std::{collections::HashMap, ops::AddAssign, sync::Arc};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
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
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    camera::Camera,
    game::{Game, Keycode},
    game_object::GameObject,
    model::Model,
};

mod camera;
mod game;
mod game_object;
mod model;

type ModelHandle = &'static str;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the `triangle`
    // example if you haven't done so yet.

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let uniform_buffer = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
    );

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let (mut pipeline, mut framebuffers) =
        window_size_dependent_setup(&memory_allocator, &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut game = Game::new();

    let mut assets = HashMap::<ModelHandle, Model>::new();
    let teapot_handle = "assets/teapot.obj";
    let quad_handle = "assets/quad.obj";
    assets.insert(
        teapot_handle,
        Model::from_obj_path(&memory_allocator, teapot_handle),
    );
    assets.insert(
        quad_handle,
        Model::from_obj_path(&memory_allocator, quad_handle),
    );

    game.add_startup_system(|mut commands: Commands| {
        let mut obj = GameObject::new();
        obj.transform.translation = Vec3::new(-0.7, -0.3, 0.0);
        obj.transform.scale = Vec3::splat(0.2);
        obj.model = Some(teapot_handle);
        commands.spawn(obj);

        let mut obj = GameObject::new();
        obj.transform.translation = Vec3::new(0.7, -0.3, 0.0);
        obj.transform.scale = Vec3::splat(0.2);
        obj.model = Some(teapot_handle);
        commands.spawn(obj);

        let mut obj = GameObject::new();
        obj.transform.translation = Vec3::new(0.0, -0.5, 0.0);
        obj.transform.scale = Vec3::new(3.0, 1.0, 3.0);
        obj.model = Some(quad_handle);
        commands.spawn(obj);

        commands.spawn(Camera::new(
            Vec3::new(0.0, 0.0, -3.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
    });
    game.add_system(|keycode: Res<Keycode>, mut camera_q: Query<&mut Camera>| {
        let mut camera = camera_q.single_mut();
        let keyboard_camera_increment = 0.05;
        match keycode.keycode {
            Some(VirtualKeyCode::W) => {
                camera.y_angle += keyboard_camera_increment;
            }
            Some(VirtualKeyCode::S) => {
                camera.y_angle -= keyboard_camera_increment;
            }
            Some(VirtualKeyCode::A) => {
                camera.x_angle -= keyboard_camera_increment;
            }
            Some(VirtualKeyCode::D) => {
                camera.x_angle += keyboard_camera_increment;
            }
            _ => {}
        }

        let x = camera.radius * camera.y_angle.cos() * camera.x_angle.sin();
        let y = camera.radius * camera.y_angle.sin();
        let z = camera.radius * camera.y_angle.cos() * camera.x_angle.cos();

        camera.position = Vec3::ZERO + Vec3::new(x, y, z);
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => {
            game.handle_keyboard_events(input);
        }
        Event::RedrawEventsCleared => {
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let dimensions = window.inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("failed to recreate swapchain: {e}"),
                };

                swapchain = new_swapchain;
                let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                    &memory_allocator,
                    &vs,
                    &fs,
                    &new_images,
                    render_pass.clone(),
                );
                pipeline = new_pipeline;
                framebuffers = new_framebuffers;
                recreate_swapchain = false;
            }

            game.run();

            let mut camera = game.world().query::<&mut Camera>().single_mut(game.world());

            let uniform_buffer_subbuffer = {
                let aspect_ratio =
                    swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                camera.calc_perspective_projection(aspect_ratio);

                let projection_view = camera.calc_perspective_projection(aspect_ratio)
                    * camera.calc_view_direction(Vec3::new(0.0, -1.0, 0.0));

                let uniform_data = vs::Data {
                    projectionView: projection_view.to_cols_array_2d(),
                    ambientLightColor: [1.0, 1.0, 1.0, 0.2],
                    lightPosition: Vec3::splat(-1.0).to_array().into(),
                    lightColor: [1.0, 1.0, 1.0, 1.0],
                };

                let subbuffer = uniform_buffer.allocate_sized().unwrap();
                *subbuffer.write().unwrap() = uniform_data;

                subbuffer
            };

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
            )
            .unwrap();

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into()), Some(1f32.into())],
                        ..RenderPassBeginInfo::framebuffer(
                            framebuffers[image_index as usize].clone(),
                        )
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set,
                );

            for obj in game.world().query::<&GameObject>().iter(game.world()) {
                let model_matrix = obj.transform.mat4();
                let push_constants = vs::Push {
                    modelMatrix: model_matrix.to_cols_array_2d(),
                    normalMatrix: obj.transform.normal_matrix().to_cols_array_2d(),
                };

                let model_handle = if let Some(model_handle) = obj.model {
                    model_handle
                } else {
                    continue;
                };

                let model = assets.get(model_handle).unwrap();
                // this should be moved to handles
                let vertex_buffer = &model.vertex_buffer;
                let index_buffer = &model.index_buffer.to_owned().unwrap();

                builder
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .push_constants(pipeline.layout().clone(), 0, push_constants);
                builder
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap();
            }

            builder.end_render_pass().unwrap();
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    memory_allocator: &StandardMemoryAllocator,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
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
        .vertex_input_state(ModelVertex::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            },
        ]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(memory_allocator.device().clone())
        .unwrap();

    (pipeline, framebuffers)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shader.frag",
    }
}
