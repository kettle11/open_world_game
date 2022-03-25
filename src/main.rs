//! This code is messy and underdocumented. Venture forth at your own risk!
//!
#![feature(portable_simd)]

use koi::*;

fn sample_with_octaves<const LANES: usize>(
    noise: &mut clatter::Simplex2d,
    persistence: f32,
    x: f32,
    y: f32,
) -> f32
where
    std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
{
    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut max_value = 0.0;

    let mut amplitudes: [f32; LANES] = [0.0; LANES];
    let mut frequencies: [f32; LANES] = [0.0; LANES];

    for i in 0..LANES {
        amplitudes[i] = amplitude;
        frequencies[i] = frequency;

        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    let amplitudes = core::simd::Simd::<f32, LANES>::from_array(amplitudes);
    let frequencies = core::simd::Simd::<f32, LANES>::from_array(frequencies);
    let sample = noise.sample([
        core::simd::Simd::<f32, LANES>::splat(x) * frequencies,
        core::simd::Simd::<f32, LANES>::splat(y) * frequencies,
    ]) * amplitudes;

    sample.value.reduce_sum() / max_value
}

#[derive(Clone)]
struct Terrain {
    resolution: usize,
    generation: usize,
    heights: Vec<f32>,
    colors: Vec<Vec4>,
    tile_space_position: Vec2u,
}

impl Terrain {
    fn new() -> Self {
        Self {
            generation: 0,
            resolution: 0,
            tile_space_position: Vec2u::ZERO,
            heights: Vec::new(),
            colors: Vec::new(),
        }
    }

    fn generate_terrain_data(
        &mut self,
        offset: Vec3,
        world_units_per_chunk: f32,
        height: f32,
        y_smoothing: f32,
        resolution: usize,
    ) {
        let snow = Color::WHITE.to_linear_srgb();
        let rocky_mountain = Color::BLACK.with_lightness(0.7).to_linear_srgb();
        let grassy_hill = Color::new_from_bytes(149, 130, 70, 255).to_linear_srgb();
        let sand = Color::new_from_bytes(197, 167, 132, 255).to_linear_srgb();
        let mut noise = clatter::Simplex2d::new();

        let world_units_per_tile = world_units_per_chunk / resolution as f32;

        let resolution = resolution + 1;

        self.heights.clear();
        self.colors.clear();

        self.heights.reserve(resolution * resolution);
        self.colors.reserve(resolution * resolution);

        for i in 0..resolution {
            for j in 0..resolution {
                let p = Vec3::new(i as f32, 0.0, j as f32) * world_units_per_tile + offset;

                let y_raw = sample_with_octaves::<16>(&mut noise, 0.5, p.x / 250., p.z / 250.);

                let y_tweak = 2.0;
                let y = y_raw * height * y_tweak;

                let t = (y.abs() / y_smoothing).min(1.0);
                let mut y = y * t * t;

                if y.abs() < 0.01 {
                    y += 0.02;
                }

                let color = if y > 20.0 {
                    snow
                } else if y > 15.0 {
                    rocky_mountain
                } else if y > 0.1 {
                    grassy_hill
                } else {
                    sand
                };

                let mut y = y / world_units_per_tile;

                if y.abs() < 0.01 {
                    y += 0.02;
                }

                self.colors.push(color);
                self.heights.push(y)
            }
        }
    }

    fn generate_mesh(
        &self,
        offset: Vec3,
        resolution: usize,
        mesh_size: f32,
        mesh_data: &mut MeshData,
        mesh_normal_calculator: &mut MeshNormalCalculator,
    ) {
        let resolution_scale = mesh_size / resolution as f32;

        let resolution = resolution + 1;
        let offset = offset * mesh_size;

        mesh_data.positions.clear();
        mesh_data.colors.clear();
        mesh_data.indices.clear();

        mesh_data.positions.reserve(resolution * resolution);
        mesh_data.colors.reserve(resolution * resolution);
        mesh_data.indices.reserve(2 * resolution * resolution);

        for i in 0..resolution {
            for j in 0..resolution {
                let y = self.heights[i * resolution + j];
                let p = Vec3::new(i as f32, y, j as f32) * resolution_scale;
                mesh_data.positions.push(p + offset);
            }
        }
        mesh_data.colors.extend_from_slice(&self.colors);

        let size = resolution as u32;
        for i in 0..size - 1 {
            for j in 0..size - 1 {
                mesh_data
                    .indices
                    .push([i * size + j, (i + 1) * size + j + 1, (i + 1) * size + j]);
                mesh_data
                    .indices
                    .push([i * size + j, i * size + j + 1, (i + 1) * size + j + 1]);
            }
        }

        mesh_normal_calculator.calculate_normals(mesh_data);
    }
}

fn main() {
    App::new().setup_and_run(|world: &mut World| {
        let tiles: usize = 3;

        world
            .get_singleton::<Graphics>()
            .set_automatic_redraw(false);

        // The light and shadow caster is spawned as part of this.
        spawn_skybox(world, "assets/qwantani_1k.hdr");

        let mesh_size = 1024.0 / tiles as f32;

        // Spawn a camera
        let center = Vec3::new(mesh_size as f32 / 2.0, 0.0, mesh_size as f32 / 2.0) * tiles as f32;
        let mut camera = Camera::new();
        camera.set_near_plane(1.0);

        let mut camera_controls = CameraControls::new();
        camera_controls.max_speed *= 10.0;
        world.spawn((
            Transform::new()
                .with_position(center + Vec3::Y * 400.0 + Vec3::Z * 400.)
                .looking_at(center, Vec3::Y),
            camera,
            camera_controls,
        ));

        // Setup the water plane
        let water_material = (|materials: &mut Assets<Material>| {
            materials.add(new_pbr_material(
                Shader::PHYSICALLY_BASED_TRANSPARENT_DOUBLE_SIDED,
                PBRProperties {
                    roughness: 0.02,
                    base_color: Color::new_from_bytes(7, 80, 97, 200),
                    ..Default::default()
                },
            ))
        })
        .run(world);

        // Spawn water planes
        for i in 0..tiles {
            for j in 0..tiles {
                let offset = Vec3::new(i as f32, 0.0, j as f32) * mesh_size;
                world.spawn((
                    Transform::new()
                        .with_position(offset + Vec3::XZ * (mesh_size / 2.0))
                        .with_scale(Vec3::fill(mesh_size)),
                    Mesh::PLANE,
                    water_material.clone(),
                    RenderFlags::DEFAULT.with_layer(RenderFlags::DO_NOT_CAST_SHADOWS),
                ));
            }
        }

        let ground_material = (|materials: &mut Assets<Material>| {
            materials.add(new_pbr_material(
                Shader::PHYSICALLY_BASED,
                PBRProperties {
                    roughness: 0.97,
                    ..Default::default()
                },
            ))
        })
        .run(world);
        let mut new_meshes: Vec<_> = Vec::new();
        (|graphics: &mut Graphics, meshes: &mut Assets<Mesh>| {
            for _ in 0..tiles * tiles {
                new_meshes.push(meshes.add(Mesh::new(graphics, MeshData::default())));
            }
        })
        .run(world);

        for new_mesh in new_meshes.iter().cloned() {
            world.spawn((Transform::new(), new_mesh.clone(), ground_material.clone()));
        }

        let mut random = Random::new();
        let mut offset = Vec3::new(random.f32(), 0.0, random.f32()) * 100.0;

        // Setup the UI
        let mut fonts = Fonts::empty();
        fonts.load_default_fonts();

        let mut standard_context = StandardContext::new(
            StandardStyle {
                primary_text_color: Color::WHITE,
                primary_color: Color::BLACK.with_alpha(0.5),
                padding: 12.,
                ..Default::default()
            },
            StandardInput::default(),
            fonts,
        );

        let mut root_widget = padding(column((
            align(
                Alignment::End,
                Alignment::Start,
                row((
                    button("Random Location", |ui_state: &mut UIState| {
                        ui_state.generate = true
                    }),
                    toggle_button(
                        text("Settings"),
                        |ui_state: &mut UIState| &mut ui_state.settings_open,
                        |ui_state: &mut UIState| !ui_state.settings_open,
                    ),
                    toggle_button(
                        text("About"),
                        |ui_state: &mut UIState| &mut ui_state.about_open,
                        |ui_state: &mut UIState| !ui_state.about_open,
                    ),
                )),
            ),
            conditional(
                |state, _| state.settings_open,
                align(
                    Alignment::End,
                    Alignment::Start,
                    stack((
                        rounded_fill(
                            |_, _, _: &StandardContext<_>| Color::BLACK.with_alpha(0.5),
                            |_, c| c.standard_style().rounding,
                        ),
                        padding(fit(width(
                            350.,
                            column((
                                text("Zoom:").with_color(|_, _, _| Color::WHITE),
                                slider(
                                    |data: &mut UIState| &mut data.world_units_per_chunk,
                                    15.0 / tiles as f32,
                                    1024.0 / tiles as f32,
                                ),
                                text("Height Scale:").with_color(|_, _, _| Color::WHITE),
                                slider(|data: &mut UIState| &mut data.height, 1.0, 70.0),
                                text("Coastal Flattening:").with_color(|_, _, _| Color::WHITE),
                                slider(|data: &mut UIState| &mut data.y_smoothing, 0.1, 50.0),
                                text("Terrain Detail:").with_color(|_, _, _| Color::WHITE),
                                slider(
                                    |data: &mut UIState| &mut data.resolution,
                                    128 / tiles,
                                    2048 / tiles,
                                ),
                            )),
                        ))),
                    )),
                ),
            ),
            conditional(
                |state: &mut UIState, _| state.about_open,
                align(
                    Alignment::End,
                    Alignment::Start,
                    stack((
                        rounded_fill(
                            |_, _, _: &StandardContext<_>| Color::BLACK.with_alpha(0.5),
                            |_, c| c.standard_style().rounding,
                        ),
                        padding(fit(width(
                            350.,
                            column((
                                row_unspaced((
                                    text("Built by  "),
                                    link(|_| "https://twitter.com/kettlecorn", text("@kettlecorn")),
                                    text(" with Rust and: "),
                                )),
                                row_unspaced((
                                    text("•  "),
                                    link(|_| "https://github.com/kettle11/koi", text("koi")),
                                    text(" for visuals, UI, and controls"),
                                )),
                                row_unspaced((
                                    text("•  "),
                                    link(|_| "https://github.com/Ralith/clatter", text("clatter")),
                                    text(" for SIMD-accelerated simplex noise"),
                                )),
                                row_unspaced((
                                    text("•  "),
                                    link(
                                        |_| "https://github.com/mooman219/fontdue",
                                        text("fontdue"),
                                    ),
                                    text(" for text-rendering"),
                                )),
                                row_unspaced((
                                    text("Check out the source on  "),
                                    link(
                                        |_| "https://github.com/kettle11/open_world_game",
                                        text("GitHub"),
                                    ),
                                )),
                            )),
                        ))),
                    )),
                ),
            ),
        )));

        let mut ui_manager = UIManager::new(world);
        world.spawn((Transform::new(), Camera::new_for_user_interface()));

        #[derive(Copy, Clone, PartialEq)]
        struct UIState {
            generate: bool,
            world_units_per_chunk: f32,
            height: f32,
            y_smoothing: f32,
            resolution: usize,
            settings_open: bool,
            about_open: bool,
        }

        let mut ui_state = UIState {
            generate: false,
            world_units_per_chunk: 0.2 * 1024.0 / tiles as f32,
            height: 10.0,
            y_smoothing: 5.0,
            resolution: 512 / tiles,
            settings_open: true,
            about_open: true,
        };
        let mut last_state = ui_state.clone();
        let mut regenerate = true;

        let mut terrain_pool = Vec::new();

        let mut generation: usize = 0;

        let mut outstanding_events = 0;

        let mut mesh_normal_calculator = MeshNormalCalculator::new();

        let mut first_generation = true;
        let (complete_chunks_sender, complete_chunks_receiver) = std::sync::mpsc::channel();
        request_window_redraw(world);

        move |event, world| {
            match &event {
                Event::KappEvent(e) => {
                    if ui_manager.handle_event(e, &mut ui_state, &mut standard_context) {
                        if ui_state.height != last_state.height
                            || ui_state.y_smoothing != last_state.y_smoothing
                            || ui_state.resolution != last_state.resolution
                            || ui_state.world_units_per_chunk != last_state.world_units_per_chunk
                            || ui_state.generate
                        {
                            regenerate = true;
                            request_window_redraw(world);
                        }

                        if ui_state.settings_open && !last_state.settings_open {
                            ui_state.about_open = false;
                        }
                        if ui_state.about_open && !last_state.about_open {
                            ui_state.settings_open = false;
                        }
                        last_state = ui_state;

                        return true;
                    }
                }
                _ => {}
            }

            if ui_state.height != last_state.height
                || ui_state.y_smoothing != last_state.y_smoothing
                || ui_state.resolution != last_state.resolution
                || ui_state.world_units_per_chunk != last_state.world_units_per_chunk
            {
                regenerate = true;
                request_window_redraw(world);
            }
            last_state = ui_state;

            if ui_state.generate {
                ui_state.generate = false;
                offset = Vec3::new(random.f32(), 0.0, random.f32()) * 8000.0;
                regenerate = true;
            }
            match event {
                Event::KappEvent(KappEvent::KeyDown {
                    key: Key::Space, ..
                }) => {
                    ui_state.generate = true;
                    request_window_redraw(world)
                }
                Event::KappEvent(event) => {
                    match event {
                        KappEvent::KeyDown { .. }
                        | KappEvent::PointerDown { .. }
                        | KappEvent::PointerUp { .. }
                        | KappEvent::Scroll { .. }
                        | KappEvent::PinchGesture { .. }
                        | KappEvent::PointerMoved{.. }
                        // Probably this WindowResized check should be in `koi` instead.
                        | KappEvent::WindowResized { .. } =>  request_window_redraw(world),
                        _ => {},
                    };

                    let input = world.get_singleton::<Input>();
                    if input.key(Key::W)
                        || input.key(Key::A)
                        || input.key(Key::S)
                        || input.key(Key::D)
                        || input.pointer_button(PointerButton::Secondary)
                    {
                        request_window_redraw(world)
                    }
                }

                Event::Draw => {
                    if outstanding_events > 0 || first_generation {
                        request_window_redraw(world);
                    }

                    if regenerate && outstanding_events == 0 {
                        generation = generation.overflowing_add(1).0;
                        regenerate = false;
                        for i in 0..tiles {
                            for j in 0..tiles {
                                let offset = offset
                                    + ui_state.world_units_per_chunk
                                        * Vec3::new(i as f32, 0.0, j as f32);

                                let world_units_per_chunk = ui_state.world_units_per_chunk;
                                let height = ui_state.height;
                                let y_smoothing = ui_state.y_smoothing;
                                let resolution = ui_state.resolution;

                                let mut terrain =
                                    terrain_pool.pop().unwrap_or_else(|| Terrain::new());

                                terrain.tile_space_position = Vec2u::new(i, j);

                                terrain.generation = generation;
                                terrain.resolution = resolution;
                                let complete_chunks_sender = complete_chunks_sender.clone();
                                ktasks::spawn(async move {
                                    terrain.generate_terrain_data(
                                        offset,
                                        world_units_per_chunk,
                                        height,
                                        y_smoothing,
                                        resolution,
                                    );
                                    let _ = complete_chunks_sender.send(terrain);
                                })
                                .run();
                                outstanding_events += 1;
                            }
                        }
                    }

                    (|graphics: &mut Graphics, meshes: &mut Assets<Mesh>| {
                        while let Ok(generated_terrain) = complete_chunks_receiver.try_recv() {
                            // if generated_terrain.generation == generation {
                            let (i, j) = generated_terrain.tile_space_position.into();
                            let mesh = meshes.get_mut(&new_meshes[i * tiles + j]);
                            let mesh_data = mesh.mesh_data.as_mut().unwrap();
                            generated_terrain.generate_mesh(
                                Vec3::new(i as f32, 0.0, j as f32),
                                generated_terrain.resolution,
                                mesh_size,
                                mesh_data,
                                &mut mesh_normal_calculator,
                            );
                            mesh.update_mesh_on_gpu(graphics);
                            //  }
                            terrain_pool.push(generated_terrain);
                            outstanding_events -= 1;

                            first_generation = false;
                        }
                    })
                    .run(world);

                    ui_manager.prepare(world, &mut standard_context);
                    ui_manager.layout(&mut ui_state, &mut standard_context, &mut root_widget);
                    ui_manager.render_ui(world);
                }
                _ => {}
            }

            false
        }
    });
}

struct MeshNormalCalculator {
    normal_use_count: Vec<i32>,
}

impl MeshNormalCalculator {
    pub fn new() -> Self {
        Self {
            normal_use_count: Vec::new(),
        }
    }
    pub fn calculate_normals(&mut self, mesh_data: &mut MeshData) {
        self.normal_use_count.clear();
        self.normal_use_count.resize(mesh_data.positions.len(), 0);

        mesh_data.normals.clear();
        mesh_data
            .normals
            .resize(mesh_data.positions.len(), Vec3::ZERO);
        for [p0, p1, p2] in mesh_data.indices.iter().cloned() {
            let dir0 = mesh_data.positions[p1 as usize] - mesh_data.positions[p0 as usize];
            let dir1 = mesh_data.positions[p2 as usize] - mesh_data.positions[p1 as usize];
            let normal = dir0.cross(dir1).normalized();
            self.normal_use_count[p0 as usize] += 1;
            self.normal_use_count[p1 as usize] += 1;
            self.normal_use_count[p2 as usize] += 1;
            mesh_data.normals[p0 as usize] += normal;
            mesh_data.normals[p1 as usize] += normal;
            mesh_data.normals[p2 as usize] += normal;
        }

        for (normal, &normal_use_count) in mesh_data
            .normals
            .iter_mut()
            .zip(self.normal_use_count.iter())
        {
            *normal = *normal / normal_use_count as f32;
        }
    }
}
