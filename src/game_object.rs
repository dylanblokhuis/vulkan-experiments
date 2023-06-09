use bevy_ecs::prelude::Component;
use glam::{Mat4, Quat, Vec3};

use crate::ModelHandle;

#[derive(Clone)]
pub struct TransformComponent {
    pub translation: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
}

impl TransformComponent {
    pub fn new() -> TransformComponent {
        TransformComponent {
            translation: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    pub fn mat4(&self) -> Mat4 {
        // glam::Mat4::from_translation(self.translation)
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn normal_matrix(&self) -> Mat4 {
        self.mat4().inverse()
    }
}

#[derive(Component, Clone)]
pub struct GameObject {
    pub transform: TransformComponent,
    pub model: Option<ModelHandle>,
}

// static mut LATEST_ID: u32 = 0;

impl GameObject {
    pub fn new() -> GameObject {
        GameObject {
            transform: TransformComponent::new(),
            model: None,
        }
    }
}

#[derive(Component, Clone)]
pub struct PointLight {
    pub color: Vec3,
    pub radius: f32,
    pub transform: TransformComponent,
}

impl PointLight {
    pub fn new() -> PointLight {
        PointLight {
            color: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.1,
            transform: TransformComponent::new(),
        }
    }
}
