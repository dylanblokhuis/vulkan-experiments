use bevy_ecs::{
    schedule::{IntoSystemConfig, Schedule, ScheduleLabel},
    world::World,
};

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

impl Game {
    pub fn new() -> Game {
        let world = World::new();
        let startup_schedule = Schedule::default();
        let main_schedule = Schedule::default();

        Game {
            world,
            startup_schedule,
            main_schedule,
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
        self.main_schedule.run(&mut self.world);
    }
}
