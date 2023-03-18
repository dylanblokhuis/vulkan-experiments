use bevy_ecs::system::{Commands, Query, Res, ResMut};
use glam::Vec3;

use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::ImageUsage,
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, MouseScrollDelta, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    camera::Camera,
    game::{Assets, Game, Keycode, MouseWheelDelta},
    game_object::GameObject,
    model::Model,
    render_ctx::RenderContextT,
    world_renderer::{world_fs, world_vs, WorldRenderer},
};

mod camera;
mod game;
mod game_object;
mod model;
mod render_ctx;
mod world_renderer;

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

    let world_vs_ref = world_vs::load(device.clone()).unwrap();
    let world_fs_ref = world_fs::load(device.clone()).unwrap();

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

    let mut world_r = WorldRenderer::new(
        memory_allocator.clone(),
        &images,
        render_pass.clone(),
        &world_vs_ref,
        &world_fs_ref,
    );

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut game = Game::new();
    let allocator_cl = memory_allocator.clone();
    game.add_startup_system(move |mut commands: Commands, mut assets: ResMut<Assets>| {
        let teapot_handle = "assets/teapot.obj";
        let quad_handle = "assets/quad.obj";
        assets.map.insert(
            teapot_handle,
            Model::from_obj_path(allocator_cl.clone(), teapot_handle),
        );
        assets.map.insert(
            quad_handle,
            Model::from_obj_path(allocator_cl.clone(), quad_handle),
        );

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
    game.add_system(
        |mut mouse_wheel: ResMut<MouseWheelDelta>, mut camera_q: Query<&mut Camera>| {
            let delta = mouse_wheel.delta;

            if let Some(delta) = delta {
                let mut camera = camera_q.single_mut();

                let mouse_camera_increment = 0.2;

                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y < 0.0 {
                            camera.radius += mouse_camera_increment;
                        } else {
                            camera.radius -= mouse_camera_increment;
                        }
                    }
                    MouseScrollDelta::PixelDelta(_) => {}
                }
            }

            mouse_wheel.delta = None;
        },
    );

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
        Event::WindowEvent {
            event: WindowEvent::MouseWheel { delta, .. },
            ..
        } => {
            game.handle_mouse_wheel_events(delta);
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
                world_r = WorldRenderer::new(
                    memory_allocator.clone(),
                    &new_images,
                    render_pass.clone(),
                    &world_vs_ref,
                    &world_fs_ref,
                );

                // pipeline = new_pipeline;
                // framebuffers = new_framebuffers;
                recreate_swapchain = false;
            }

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

            game.run();

            world_r.render(
                &mut game,
                &swapchain,
                &mut builder,
                image_index as usize,
                descriptor_set_allocator.clone(),
            );

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
