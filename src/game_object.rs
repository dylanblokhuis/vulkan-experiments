use std::f32::consts::PI;

use bevy_ecs::prelude::Component;
use glam::{Vec2, Vec3};

#[derive(Clone)]
pub struct Transform2dComponent {
    pub translation: Vec2,
    pub scale: Vec2,
    pub rotation: f32,
}

#[derive(Component, Clone)]
pub struct GameObject {
    pub color: Vec3,
    pub transform: Transform2dComponent,
}

// static mut LATEST_ID: u32 = 0;

impl GameObject {
    pub fn new() -> GameObject {
        GameObject {
            color: Vec3::new(1.0, 0.8, 1.0),
            transform: Transform2dComponent {
                translation: Vec2::new(0.4, 0.0),
                scale: Vec2::new(2.0, 0.5),
                rotation: 0.25 * PI,
            },
        }
    }
}
