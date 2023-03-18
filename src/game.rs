use std::{collections::HashMap, time::Instant};

use bevy_ecs::{
    schedule::{IntoSystemConfig, Schedule, ScheduleLabel},
    system::Resource,
    world::World,
};
use bevy_time::Time;
use winit::event::{ElementState, KeyboardInput, MouseScrollDelta, TouchPhase, VirtualKeyCode};

use crate::{model::Model, ModelHandle};

pub struct Game {
    world: World,
    startup_schedule: Schedule,
    main_schedule: Schedule,
}

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoreSchedule {
    Startup,
    Main,
}

#[derive(Resource, Default)]
pub struct Keycode {
    pub keycode: Option<VirtualKeyCode>,
}

#[derive(Resource, Default)]
pub struct MouseWheelDelta {
    pub delta: Option<MouseScrollDelta>,
}

// let mut assets = HashMap::<ModelHandle, Model>::new();
#[derive(Resource, Default)]
pub struct Assets {
    pub map: HashMap<ModelHandle, Model>,
}

impl Game {
    pub fn new() -> Game {
        let mut world = World::new();
        world.insert_resource(Keycode::default());
        world.insert_resource(Time::default());
        world.insert_resource(MouseWheelDelta::default());
        world.insert_resource(Assets::default());

        let startup_schedule = Schedule::default();
        let main_schedule = Schedule::default();

        Game {
            world,
            startup_schedule,
            main_schedule,
        }
    }

    pub fn handle_keyboard_events(&mut self, input: KeyboardInput) {
        if input.state == ElementState::Pressed {
            match input.virtual_keycode {
                Some(key) => {
                    self.world.insert_resource(Keycode { keycode: Some(key) });
                }
                None => {
                    self.world.insert_resource(Keycode { keycode: None });
                }
            }
        } else {
            self.world.insert_resource(Keycode { keycode: None });
        }
    }

    pub fn handle_mouse_wheel_events(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                self.world.insert_resource(MouseWheelDelta {
                    delta: Some(MouseScrollDelta::LineDelta(x, y)),
                });
            }
            MouseScrollDelta::PixelDelta(pos) => {
                self.world.insert_resource(MouseWheelDelta {
                    delta: Some(MouseScrollDelta::PixelDelta(pos)),
                });
            }
        }
    }
    pub fn add_startup_system<M>(&mut self, system: impl IntoSystemConfig<M>) {
        self.startup_schedule.add_system(system);
        self.startup_schedule.run(&mut self.world);
    }

    pub fn add_system<M>(&mut self, system: impl IntoSystemConfig<M>) {
        self.main_schedule.add_system(system);
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn run(&mut self) {
        self.world.resource_mut::<Time>().update();
        // self.time.update();
        self.main_schedule.run(&mut self.world);
    }
}
