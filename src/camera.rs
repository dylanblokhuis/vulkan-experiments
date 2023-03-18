use glam::Mat4;

pub struct Camera {
    pub projection: Mat4,
    pub view: Mat4,
    pub position: glam::Vec3,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            projection: Mat4::NAN,
            view: Mat4::NAN,
            position: glam::Vec3::new(0.0, 0.0, -1.0),
        }
    }

    pub fn set_orthographic_projection(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
    ) {
        self.projection = Mat4::orthographic_rh(left, right, bottom, top, near, far);
    }

    pub fn set_perspective_projection(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
        self.projection = Mat4::perspective_rh(fov, aspect, near, far);
    }

    pub fn set_view_direction(
        &mut self,
        position: glam::Vec3,
        direction: glam::Vec3,
        up: glam::Vec3,
    ) {
        self.view = Mat4::look_at_rh(position, direction, up);
    }

    pub fn set_view_target(&mut self, position: glam::Vec3, target: glam::Vec3, up: glam::Vec3) {
        self.set_view_direction(position, target - position, up);
    }

    pub fn set_view_xyz(&mut self, position: glam::Vec3, rotation: glam::Vec3) {
        self.view = Mat4::from_rotation_translation(
            glam::Quat::from_rotation_x(rotation.x)
                * glam::Quat::from_rotation_y(rotation.y)
                * glam::Quat::from_rotation_z(rotation.z),
            position,
        );
    }
}
