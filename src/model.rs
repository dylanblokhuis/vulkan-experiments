use std::{collections::HashMap, fs::File, io::BufReader, sync::Arc};

use glam::Vec3;
use gltf::mesh::util::{ReadColors, ReadIndices};
use obj::raw::{object::Polygon, RawObj};

use vulkano::{
    buffer::{Buffer, BufferAllocateInfo, BufferContents, BufferUsage, Subbuffer},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
};

// We now create a buffer that will store the shape of our triangle.
// We use #[repr(C)] here to force rustc to not do anything funky with our data, although for this
// particular example, it doesn't actually change the in-memory representation.
#[repr(C)]
#[derive(BufferContents, Vertex, Clone, Debug)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
}

// impl_vertex!(Vertex, position, color, normal);

#[derive(Clone)]
pub struct Model {
    pub vertex_buffer: Subbuffer<[ModelVertex]>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
}

type VertexKey = (usize, usize, usize);
struct MeshIndices {
    indices: Vec<u32>,
    saved: HashMap<VertexKey, u32>,
    next: u32,
}

impl MeshIndices {
    pub fn new(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            saved: HashMap::with_capacity(capacity),
            next: 0,
        }
    }

    pub fn insert<F: FnOnce()>(&mut self, key: VertexKey, create_vertex: F) {
        // Check if the vertex is already saved
        match self.saved.get(&key) {
            Some(index) => self.indices.push(*index), // If saved, just use the existing index
            None => {
                // Save the index to both the indices and saved
                self.indices.push(self.next);
                self.saved.insert(key, self.next);
                // Increment next index
                self.next += 1;
                // Create a vertex externally
                create_vertex()
            }
        }
    }
}

struct DataUri<'a> {
    mime_type: &'a str,
    base64: bool,
    data: &'a str,
}

fn split_once(input: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut iter = input.splitn(2, delimiter);
    Some((iter.next()?, iter.next()?))
}

impl<'a> DataUri<'a> {
    fn parse(uri: &'a str) -> Result<DataUri<'a>, ()> {
        let uri = uri.strip_prefix("data:").ok_or(())?;
        let (mime_type, data) = split_once(uri, ',').ok_or(())?;

        let (mime_type, base64) = match mime_type.strip_suffix(";base64") {
            Some(mime_type) => (mime_type, true),
            None => (mime_type, false),
        };

        Ok(DataUri {
            mime_type,
            base64,
            data,
        })
    }

    fn decode(&self) -> Result<Vec<u8>, base64::DecodeError> {
        if self.base64 {
            base64::decode(self.data)
        } else {
            Ok(self.data.as_bytes().to_owned())
        }
    }
}

impl Model {
    pub fn new(
        memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        vertices: Vec<ModelVertex>,
        indices: Option<Vec<u32>>,
    ) -> Self {
        println!("{} vertices", vertices.len());
        println!("{} indices", indices.as_ref().map(|i| i.len()).unwrap_or(0));
        let maybe_index_buffer =
            indices.map(|indices| index_buffer(indices, memory_allocator.clone()));

        Self {
            vertex_buffer: vertex_buffer(vertices, memory_allocator),
            index_buffer: maybe_index_buffer,
        }
    }

    pub fn from_obj_path(
        memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        path: &str,
    ) -> Self {
        let bytes = BufReader::new(File::open(path).unwrap());
        let raw = obj::raw::parse_obj(bytes).unwrap();
        let vertcount = raw.polygons.len() * 3;
        let mut indices = MeshIndices::new(vertcount);

        let mut vertices: Vec<ModelVertex> = Vec::with_capacity(vertcount);

        let color = [1.0, 1.0, 1.0];

        for polygon in &raw.polygons {
            match polygon {
                Polygon::P(poly) if poly.len() == 3 => {
                    let normal = calculate_normal(&raw, poly);

                    for ipos in poly {
                        indices.insert((*ipos, 0, 0), || {
                            vertices.push(ModelVertex {
                                position: convert_position(&raw, *ipos),
                                normal,
                                color,
                            });
                        });
                    }
                }
                Polygon::PT(poly) if poly.len() == 3 => {
                    let triangle: Vec<usize> = poly.iter().map(|(ipos, _)| *ipos).collect();
                    let normal = calculate_normal(&raw, &triangle);

                    for (ipos, itex) in poly {
                        indices.insert((*ipos, 0, *itex), || {
                            vertices.push(ModelVertex {
                                position: convert_position(&raw, *ipos),
                                normal,
                                color,
                            });
                        });
                    }
                }
                Polygon::PN(poly) if poly.len() == 3 => {
                    for (ipos, inorm) in poly {
                        indices.insert((*ipos, *inorm, 0), || {
                            vertices.push(ModelVertex {
                                position: convert_position(&raw, *ipos),
                                normal: convert_normal(&raw, *inorm),
                                color,
                            });
                        });
                    }
                }
                Polygon::PTN(poly) if poly.len() == 3 => {
                    for (ipos, itex, inorm) in poly {
                        indices.insert((*ipos, *inorm, *itex), || {
                            vertices.push(ModelVertex {
                                position: convert_position(&raw, *ipos),
                                normal: convert_normal(&raw, *inorm),
                                color,
                            });
                        });
                    }
                }
                _ => {}
            }
        }

        let maybe_indices = if !indices.indices.is_empty() {
            Some(indices.indices)
        } else {
            None
        };

        Model::new(memory_allocator, vertices, maybe_indices)
    }

    pub fn from_first_mesh_in_gltf_path(
        memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        path: &str,
    ) -> Self {
        const VALID_MIME_TYPES: &[&str] = &["application/octet-stream", "application/gltf-buffer"];
        let bytes = BufReader::new(File::open(path).unwrap());
        let gltf = gltf::Gltf::from_reader(bytes).unwrap();
        let mut buffer_data: Vec<Vec<u8>> = Vec::new();
        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Uri { 0: uri } => {
                    let uri = percent_encoding::percent_decode_str(uri)
                        .decode_utf8()
                        .unwrap();
                    let uri = uri.as_ref();
                    let buffer_bytes = match DataUri::parse(uri) {
                        Ok(data_uri) if VALID_MIME_TYPES.contains(&data_uri.mime_type) => {
                            data_uri.decode().unwrap()
                        }
                        Ok(_) => panic!("Invalid mime type"),
                        _ => panic!("Invalid uri"),
                    };
                    buffer_data.push(buffer_bytes);
                }
                gltf::buffer::Source::Bin => {
                    if let Some(blob) = gltf.blob.as_deref() {
                        buffer_data.push(blob.into());
                    } else {
                        panic!("No blob found in gltf");
                    }
                }
                _ => {}
            }
        }

        for mesh in gltf.meshes() {
            #[allow(clippy::never_loop)]
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

                let mut normal = [0.0, 0.0, 0.0];
                let mut color = [1.0, 1.0, 1.0];
                let mut position: [f32; 3] = [0.0, 0.0, 0.0];

                let mut vertices: Vec<ModelVertex> = Vec::new();
                let mut indices: Vec<u32> = Vec::new();

                if let Some(vertex_attribute) = reader.read_positions() {
                    for position in vertex_attribute {
                        vertices.push(ModelVertex {
                            position,
                            normal,
                            color,
                        });
                    }
                }

                if let Some(vertex_attribute) = reader.read_normals() {
                    for (i, normal) in vertex_attribute.enumerate() {
                        vertices[i].normal = normal;
                    }
                }

                if let Some(read_indices) = reader.read_indices() {
                    match read_indices {
                        ReadIndices::U8(_indices) => {
                            for index in _indices {
                                indices.push(index as u32);
                            }
                        }
                        ReadIndices::U16(_indices) => {
                            for index in _indices {
                                indices.push(index as u32);
                            }
                        }
                        ReadIndices::U32(_indices) => {
                            for index in _indices {
                                indices.push(index);
                            }
                        }
                    }
                }

                if let Some(read_colors) = reader.read_colors(0) {
                    let rgba = read_colors.into_rgb_f32();
                    for (i, color) in rgba.enumerate() {
                        vertices[i].color = color;
                    }
                }

                return Model::new(memory_allocator, vertices, Some(indices));
            }
        }

        panic!("No mesh found in gltf");
    }
    // pub fn staging_vertex_buffer(
    //     &self,
    //     memory_allocator: &StandardMemoryAllocator,
    //     device: &Arc<vulkano::device::Device>,
    //     command_buffer_allocator: &StandardCommandBufferAllocator,
    //     queue: &Arc<vulkano::device::Queue>,
    // ) -> Arc<DeviceLocalBuffer<[Vertex]>> {
    //     let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
    //         memory_allocator,
    //         BufferUsage {
    //             transfer_src: true,
    //             ..BufferUsage::empty()
    //         }, // Specify this buffer will be used as a transfer source.
    //         false,
    //         self.vertices.clone(),
    //     )
    //     .unwrap();

    //     // Create a buffer array on the GPU with enough space for `10_000` floats.
    //     let device_local_buffer = DeviceLocalBuffer::<[Vertex]>::array(
    //         memory_allocator,
    //         self.vertices.len() as vulkano::DeviceSize,
    //         BufferUsage {
    //             storage_buffer: true,
    //             transfer_dst: true,
    //             vertex_buffer: true,
    //             ..BufferUsage::empty()
    //         }, // Specify use as a storage buffer and transfer destination.
    //         device.active_queue_family_indices().iter().copied(),
    //     )
    //     .unwrap();

    //     // Create a one-time command to copy between the buffers.
    //     let mut cbb = AutoCommandBufferBuilder::primary(
    //         command_buffer_allocator,
    //         queue.queue_family_index(),
    //         CommandBufferUsage::OneTimeSubmit,
    //     )
    //     .unwrap();
    //     cbb.copy_buffer(CopyBufferInfo::buffers(
    //         temporary_accessible_buffer,
    //         device_local_buffer.clone(),
    //     ))
    //     .unwrap();
    //     let cb = cbb.build().unwrap();

    //     cb.execute(queue.clone())
    //         .unwrap()
    //         .then_signal_fence_and_flush()
    //         .unwrap()
    //         .wait(None /* timeout */)
    //         .unwrap();

    //     return device_local_buffer;
    // }

    // pub fn index_buffer(
    //     &self,
    //     memory_allocator: &StandardMemoryAllocator,
    // ) -> Option<Arc<CpuAccessibleBuffer<[u32]>>> {
    //     self.indices.clone().map(|indices| {
    //         CpuAccessibleBuffer::from_iter(
    //             memory_allocator,
    //             BufferUsage {
    //                 index_buffer: true,
    //                 ..BufferUsage::empty()
    //             },
    //             false,
    //             indices,
    //         )
    //         .unwrap()
    //     })
    // }

    // pub fn staging_index_buffer(
    //     &self,
    //     memory_allocator: &StandardMemoryAllocator,
    //     device: &Arc<vulkano::device::Device>,
    //     command_buffer_allocator: &StandardCommandBufferAllocator,
    //     queue: &Arc<vulkano::device::Queue>,
    // ) -> Option<Arc<DeviceLocalBuffer<[u32]>>> {
    //     let Some(indices) = &self.indices else {
    //         return None;
    //     };

    //     let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
    //         memory_allocator,
    //         BufferUsage {
    //             transfer_src: true,
    //             ..BufferUsage::empty()
    //         }, // Specify this buffer will be used as a transfer source.
    //         false,
    //         indices.clone(),
    //     )
    //     .unwrap();

    //     // Create a buffer array on the GPU with enough space for `10_000` floats.
    //     let device_local_buffer = DeviceLocalBuffer::<[u32]>::array(
    //         memory_allocator,
    //         indices.len() as vulkano::DeviceSize,
    //         BufferUsage {
    //             storage_buffer: true,
    //             transfer_dst: true,
    //             index_buffer: true,
    //             ..BufferUsage::empty()
    //         }, // Specify use as a storage buffer and transfer destination.
    //         device.active_queue_family_indices().iter().copied(),
    //     )
    //     .unwrap();

    //     // Create a one-time command to copy between the buffers.
    //     let mut cbb = AutoCommandBufferBuilder::primary(
    //         command_buffer_allocator,
    //         queue.queue_family_index(),
    //         CommandBufferUsage::OneTimeSubmit,
    //     )
    //     .unwrap();
    //     cbb.copy_buffer(CopyBufferInfo::buffers(
    //         temporary_accessible_buffer,
    //         device_local_buffer.clone(),
    //     ))
    //     .unwrap();
    //     let cb = cbb.build().unwrap();

    //     cb.execute(queue.clone())
    //         .unwrap()
    //         .then_signal_fence_and_flush()
    //         .unwrap()
    //         .wait(None /* timeout */)
    //         .unwrap();

    //     Some(device_local_buffer)
    // }
}

fn vertex_buffer(
    vertices: Vec<ModelVertex>,
    memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
) -> Subbuffer<[ModelVertex]> {
    Buffer::from_iter(
        &memory_allocator,
        BufferAllocateInfo {
            buffer_usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        vertices,
    )
    .unwrap()
}

fn index_buffer(
    indices: Vec<u32>,
    memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        &memory_allocator,
        BufferAllocateInfo {
            buffer_usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        indices,
    )
    .unwrap()
}

pub fn make_cube(memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>) -> Model {
    let vertices = vec![
        ModelVertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.9, 0.9, 0.9],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.9, 0.9, 0.9],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [-0.5, -0.5, 0.5],
            color: [0.9, 0.9, 0.9],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.9, 0.9, 0.9],
            normal: [0.0, 0.0, -1.0],
        },
        // right face (yellow)
        ModelVertex {
            position: [0.5, -0.5, -0.5],
            color: [0.8, 0.8, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, 0.5, 0.5],
            color: [0.8, 0.8, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, -0.5, 0.5],
            color: [0.8, 0.8, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, 0.5, -0.5],
            color: [0.8, 0.8, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        // top face (orange, remember y axis points down)
        ModelVertex {
            position: [-0.5, -0.5, -0.5],
            color: [0.9, 0.6, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, -0.5, 0.5],
            color: [0.9, 0.6, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [-0.5, -0.5, 0.5],
            color: [0.9, 0.6, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, -0.5, -0.5],
            color: [0.9, 0.6, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        // bottom face (red)
        ModelVertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.8, 0.1, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, 0.5, 0.5],
            color: [0.8, 0.1, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.8, 0.1, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: [0.5, 0.5, -0.5],
            color: [0.8, 0.1, 0.1],
            normal: [0.0, 0.0, -1.0],
        },
    ];

    let indices = vec![
        0, 1, 2, 0, 3, 1, 4, 5, 6, 4, 7, 5, 8, 9, 10, 8, 11, 9, 12, 13, 14, 12, 15, 13, 16, 17, 18,
        16, 19, 17, 20, 21, 22, 20, 23, 21,
    ];

    Model::new(memory_allocator, vertices, Some(indices))
}

fn convert_position(raw: &RawObj, index: usize) -> [f32; 3] {
    let position = raw.positions[index];
    [position.0, position.1, position.2]
}

fn convert_normal(raw: &RawObj, index: usize) -> [f32; 3] {
    let normal = raw.normals[index];
    [normal.0, normal.1, normal.2]
}

fn convert_texture(raw: &RawObj, index: usize) -> [f32; 2] {
    let tex_coord = raw.tex_coords[index];
    // Flip UV for correct values
    [tex_coord.0, 1.0 - tex_coord.1]
}

fn calculate_normal(raw: &RawObj, polygon: &[usize]) -> [f32; 3] {
    // Extract triangle
    let triangle: Vec<Vec3> = polygon
        .iter()
        .map(|index| Vec3::from(convert_position(raw, *index)))
        .collect();

    // Calculate normal
    let v1 = triangle[1] - triangle[0];
    let v2 = triangle[2] - triangle[0];
    let n = v1.cross(v2);

    n.to_array()
}
