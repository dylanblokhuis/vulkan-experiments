use bevy_ecs::prelude::Component;
use glam::{Mat4, Quat, Vec3};

#[derive(Component)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub x_angle: f32,
    pub y_angle: f32,
    pub radius: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(position: Vec3, direction: Vec3) -> Camera {
        Camera {
            position,
            direction,
            fov: 50.0,
            near: 0.1,
            far: 1000.0,
            radius: 5.0,
            x_angle: 8.5,
            y_angle: 0.66,
        }
    }

    pub fn calc_orthographic_projection(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
    ) -> Mat4 {
        Mat4::orthographic_rh(left, right, bottom, top, self.near, self.far)
    }

    pub fn calc_perspective_projection(&mut self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), aspect, self.near, self.far)
    }

    pub fn calc_view_direction(&mut self, up: Vec3) -> Mat4 {
        Mat4::look_at_rh(self.position, self.direction, up)
    }

    // pub fn calc_view_target(&mut self, target: Vec3, up: Vec3) -> Mat4 {
    //     self.calc_view_direction(target - self.position, up)
    // }

    pub fn calc_view_xyz(&mut self, position: Vec3, rotation: Vec3) -> Mat4 {
        Mat4::from_rotation_translation(
            Quat::from_rotation_x(rotation.x)
                * Quat::from_rotation_y(rotation.y)
                * Quat::from_rotation_z(rotation.z),
            position,
        )
    }
}
