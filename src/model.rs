use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    memory::allocator::StandardMemoryAllocator,
    sync::GpuFuture,
};

// We now create a buffer that will store the shape of our triangle.
// We use #[repr(C)] here to force rustc to not do anything funky with our data, although for this
// particular example, it doesn't actually change the in-memory representation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
}

impl Model {
    pub fn new(vertices: Vec<Vertex>, indices: Option<Vec<u32>>) -> Self {
        Self { vertices, indices }
    }

    pub fn vertex_buffer(
        &self,
        memory_allocator: &StandardMemoryAllocator,
    ) -> Arc<CpuAccessibleBuffer<[Vertex]>> {
        CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            self.vertices.clone(),
        )
        .unwrap()
    }

    pub fn staging_vertex_buffer(
        &self,
        memory_allocator: &StandardMemoryAllocator,
        device: &Arc<vulkano::device::Device>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<vulkano::device::Queue>,
    ) -> Arc<DeviceLocalBuffer<[Vertex]>> {
        let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            }, // Specify this buffer will be used as a transfer source.
            false,
            self.vertices.clone(),
        )
        .unwrap();

        // Create a buffer array on the GPU with enough space for `10_000` floats.
        let device_local_buffer = DeviceLocalBuffer::<[Vertex]>::array(
            memory_allocator,
            self.vertices.len() as vulkano::DeviceSize,
            BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                vertex_buffer: true,
                ..BufferUsage::empty()
            }, // Specify use as a storage buffer and transfer destination.
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // Create a one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer,
            device_local_buffer.clone(),
        ))
        .unwrap();
        let cb = cbb.build().unwrap();

        cb.execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        return device_local_buffer;
    }

    pub fn index_buffer(
        &self,
        memory_allocator: &StandardMemoryAllocator,
    ) -> Option<Arc<CpuAccessibleBuffer<[u32]>>> {
        self.indices.clone().map(|indices| {
            CpuAccessibleBuffer::from_iter(
                memory_allocator,
                BufferUsage {
                    index_buffer: true,
                    ..BufferUsage::empty()
                },
                false,
                indices,
            )
            .unwrap()
        })
    }

    pub fn staging_index_buffer(
        &self,
        memory_allocator: &StandardMemoryAllocator,
        device: &Arc<vulkano::device::Device>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<vulkano::device::Queue>,
    ) -> Option<Arc<DeviceLocalBuffer<[u32]>>> {
        let Some(indices) = &self.indices else {
            return None;
        };

        let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            }, // Specify this buffer will be used as a transfer source.
            false,
            indices.clone(),
        )
        .unwrap();

        // Create a buffer array on the GPU with enough space for `10_000` floats.
        let device_local_buffer = DeviceLocalBuffer::<[u32]>::array(
            memory_allocator,
            self.indices.clone().unwrap().len() as vulkano::DeviceSize,
            BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                index_buffer: true,
                ..BufferUsage::empty()
            }, // Specify use as a storage buffer and transfer destination.
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // Create a one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer,
            device_local_buffer.clone(),
        ))
        .unwrap();
        let cb = cbb.build().unwrap();

        cb.execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        return Some(device_local_buffer);
    }
}

pub fn make_cube() -> Model {
    let vertices = vec![
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.9, 0.9, 0.9],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.9, 0.9, 0.9],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [0.9, 0.9, 0.9],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.9, 0.9, 0.9],
        },
        // right face (yellow)
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.8, 0.8, 0.1],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.8, 0.8, 0.1],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [0.8, 0.8, 0.1],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.8, 0.8, 0.1],
        },
        // top face (orange, remember y axis points down)
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.9, 0.6, 0.1],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [0.9, 0.6, 0.1],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [0.9, 0.6, 0.1],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.9, 0.6, 0.1],
        },
        // bottom face (red)
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.8, 0.1, 0.1],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.8, 0.1, 0.1],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.8, 0.1, 0.1],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.8, 0.1, 0.1],
        },
    ];

    let indices = vec![
        0, 1, 2, 0, 3, 1, 4, 5, 6, 4, 7, 5, 8, 9, 10, 8, 11, 9, 12, 13, 14, 12, 15, 13, 16, 17, 18,
        16, 19, 17, 20, 21, 22, 20, 23, 21,
    ];

    Model::new(vertices, Some(indices))
}
