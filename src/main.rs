#![feature(portable_simd)]

use koi::*;

mod pathfinding;

struct Brush {
    max_influence: f32,
    center: Vec2,
    inner_radius_squared: f32,
    outer_radius_squared_to_inner_radius: f32,
    f: Box<dyn Fn(Vec3) -> f32>,
}

impl Brush {
    fn new(
        center: Vec2,
        inner_radius: f32,
        outer_radius: f32,
        max_influence: f32,
        f: Box<dyn Fn(Vec3) -> f32 + 'static>,
    ) -> Self {
        Self {
            center,
            inner_radius_squared: inner_radius * inner_radius,
            outer_radius_squared_to_inner_radius: (outer_radius - inner_radius)
                * (outer_radius - inner_radius),
            f,
            max_influence,
        }
    }

    fn apply(&self, p: Vec2, y: f32) -> f32 {
        //  println!("P: {:?}", p);
        let dis = (self.center - p).length_squared();
        //  println!("DIS: {:?}", dis);
        let effect = if dis < self.inner_radius_squared {
            1.0
        } else {
            1.0 - ((dis - self.inner_radius_squared) / self.outer_radius_squared_to_inner_radius)
                .clamp(0.0, 1.0)
        };

        let target_y = (self.f)(Vec3::new(p.x, y, p.y));
        let effect = smooth_step(effect * self.max_influence);
        (target_y - y) * effect + y
    }
}

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
fn generate_terrain_data(
    offset: Vec3,
    scale: f32,
    height: f32,
    y_smoothing: f32,
    resolution: usize,
    brushes: &[Brush],
) -> Terrain {
    let mut noise = clatter::Simplex2d::new();

    let mut heights = Vec::with_capacity(resolution * resolution);
    let mut colors = Vec::with_capacity(resolution * resolution);
    let resolution_scale = 512.0 / resolution as f32;
    let scale = scale * resolution_scale;

    for i in 0..resolution {
        for j in 0..resolution {
            let p = Vec3::new(i as f32, 0.0, j as f32) * scale + offset;

            let y_raw = sample_with_octaves::<16>(&mut noise, 0.5, p.x / 1000., p.z / 1000.);
            let y = y_raw * height;

            /*
            if y_raw > y_smoothing {
                y += 1.0;
            }

            if y_raw > y_smoothing * 1.4 {
                y += 1.0;
            }

            if y_raw > y_smoothing * 1.8 {
                y += 1.0;
            }
            */

            let t = (y.abs() / y_smoothing).min(1.0);
            let mut y = y * t * t;

            if y.abs() < 0.001 {
                y += 0.01;
            }

            for brush in brushes.iter() {
                y = brush.apply(Vec2::new(p.x, p.z), y);
            }

            let color = if y > 20.0 {
                Color::WHITE
            } else if y > 15.0 {
                Color::BLACK.with_lightness(0.7)
            } else if y > 0.1 {
                Color::new_from_bytes(149, 130, 70, 255)
            } else {
                Color::new_from_bytes(197, 167, 132, 255)
            };
            colors.push(color);
            heights.push(y)
        }
    }
    Terrain { heights, colors }
}

fn generate_terrain_mesh(resolution: usize, mesh_size: f32, terrain: &Terrain) -> MeshData {
    let mut positions = Vec::with_capacity(resolution * resolution);
    let mut colors = Vec::with_capacity(resolution * resolution);
    let mut indices = Vec::with_capacity(2 * resolution * resolution);

    let resolution_scale = mesh_size / resolution as f32;
    for i in 0..resolution {
        for j in 0..resolution {
            let y = terrain.heights[i * resolution + j];
            let mut p = Vec3::new(i as f32, 0.0, j as f32) * resolution_scale;
            p.y = y;
            positions.push(p);

            let color = terrain.colors[i * resolution + j];
            colors.push(color.to_linear_srgb());
        }
    }

    let size = resolution as u32;
    for i in 0..size - 1 {
        for j in 0..size - 1 {
            indices.push([i * size + j, (i + 1) * size + j + 1, (i + 1) * size + j]);
            indices.push([i * size + j, i * size + j + 1, (i + 1) * size + j + 1]);
        }
    }

    /*
    let mut add_side = |start: Vec2i, step: Vec2i| {
        let mut current_p = start;
        for i in 0..size - 1 {
            let y = terrain.heights[current_p.x as usize * resolution + current_p.y as usize];
            let base_pos = Vec3::new(current_p.x as f32, 0.0, current_p.y as f32) * scale;
            let p = (Vec3::new(base_pos.x, y, base_pos.z) + mesh_offset) * resolution_scale;

            let len = positions.len() as u32;

            positions.push(Vec3::new(p.x, -30.0, p.z));
            positions.push(Vec3::new(p.x, p.y, p.z));

            colors.push(Color::new_from_bytes(149, 130, 70, 255).to_linear_srgb());
            colors.push(Color::new_from_bytes(149, 130, 70, 255).to_linear_srgb());

            if i > 0 {
                indices.push([len, len + 1, len - 1]);
                indices.push([len, len - 1, len - 2]);
            }
            current_p += step;
        }
    };
    add_side(Vec2i::X * (resolution - 1) as i32, -Vec2i::X);
    add_side(Vec2i::Y * (resolution) as i32, Vec2i::X);
    */

    // add_side(Vec3u::X * resolution, Vec3u::Z);

    let mut mesh_data = MeshData {
        positions,
        colors,
        indices,
        ..Default::default()
    };
    calculate_normals(&mut mesh_data);
    mesh_data
}

struct Terrain {
    heights: Vec<f32>,
    colors: Vec<Color>,
}

fn main() {
    App::new().setup_and_run(|world: &mut World| {
        world
            .get_singleton::<Graphics>()
            .set_automatic_redraw(false);

        // The light and shadow caster is spawned as part of this.
        spawn_skybox(world, "assets/qwantani_1k.hdr");

        // Spawn a camera
        let center = Vec3::new(512.0 as f32 / 2.0, 0.0, 512.0 as f32 / 2.0);
        let mut camera = Camera::new();
        camera.set_near_plane(1.0);
        world.spawn((
            Transform::new()
                .with_position(center + Vec3::Y * 8.0 + Vec3::Z * 10.)
                .looking_at(center, Vec3::Y),
            CameraControls::new(),
            camera,
        ));

        // Spawn water
        let material = (|materials: &mut Assets<Material>| {
            materials.add(new_pbr_material(
                Shader::PHYSICALLY_BASED_TRANSPARENT_DOUBLE_SIDED,
                PBRProperties {
                    // Water is perfectly smooth but most of the time we're viewing it from a distance when it could be considered rough.
                    roughness: 0.02,
                    base_color: Color::new_from_bytes(7, 80, 97, 200),
                    ..Default::default()
                },
            ))
        })
        .run(world);
        world.spawn((
            Transform::new()
                .with_position(center)
                .with_scale(Vec3::fill(512.0)),
            Mesh::PLANE,
            material,
            RenderFlags::DEFAULT.with_layer(RenderFlags::DO_NOT_CAST_SHADOWS),
        ));

        let ground_material = (|materials: &mut Assets<Material>| {
            materials.add(new_pbr_material(
                Shader::PHYSICALLY_BASED,
                PBRProperties {
                    // Water is perfectly smooth but most of the time we're viewing it from a distance when it could be considered rough.
                    roughness: 0.97,
                    ..Default::default()
                },
            ))
        })
        .run(world);
        let tiles = 1;
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

        // Spawn a series of balls with different material properties.
        // Up is more metallic
        // Right is more more rough
        let spacing = 2.0;
        let mut commands = Commands::new();
        (|materials: &mut Assets<Material>| {
            let rows = 6;
            let columns = 6;
            for i in 0..rows {
                for j in 0..columns {
                    let new_material = materials.add(new_pbr_material(
                        Shader::PHYSICALLY_BASED,
                        PBRProperties {
                            base_color: Color::AZURE,
                            metallic: i as f32 / rows as f32,
                            roughness: (j as f32 / columns as f32).clamp(0.05, 1.0),
                            ..Default::default()
                        },
                    ));
                    commands.spawn((
                        Transform::new().with_position(
                            Vec3::new(j as f32 * spacing, i as f32 * spacing, -2.0) + center,
                        ),
                        new_material,
                        Mesh::SPHERE,
                    ))
                }
            }
        })
        .run(world);
        commands.apply(world);
        let mut random = Random::new();

        let mut offset = Vec3::new(random.f32(), 0.0, random.f32()) * 100.0;

        let mut brush_count = 20;

        let mut pathfinder = pathfinding::Pathfinder::new();
        let mut terrain = Terrain {
            heights: Vec::new(),
            colors: Vec::new(),
        };

        let mut current_path = Vec::new();

        let mut last_clicked_position = None;

        let mut fonts = Fonts::empty();
        fonts.load_default_fonts();

        let mut standard_context = StandardContext::new(
            StandardStyle {
                primary_text_color: Color::WHITE,
                padding: 20.,
                ..Default::default()
            },
            StandardInput::default(),
            fonts,
        );

        let mut root_widget = align(
            Alignment::End,
            Alignment::Start,
            padding(stack((
                rounded_fill(
                    |_, _, _: &StandardContext<_>| Color::BLACK.with_alpha(0.5),
                    |_, c| c.standard_style().rounding,
                ),
                padding(fit(width(
                    300.,
                    column((
                        button("New", |ui_state: &mut UIState| ui_state.generate = true),
                        text("Scale:"),
                        slider(|data: &mut UIState| &mut data.scale, 0.1, 30.0),
                        text("Height Scale:"),
                        slider(|data: &mut UIState| &mut data.height, 1.0, 70.0),
                        text("Coastal Flattening:"),
                        slider(|data: &mut UIState| &mut data.y_smoothing, 0.1, 70.0),
                        text("Terrain Detail:"),
                        slider(|data: &mut UIState| &mut data.resolution, 52, 1024),
                    )),
                ))),
            ))),
        );

        let mut ui_manager = UIManager::new(world);
        world.spawn((Transform::new(), Camera::new_for_user_interface()));

        #[derive(Copy, Clone, PartialEq)]
        struct UIState {
            generate: bool,
            scale: f32,
            height: f32,
            y_smoothing: f32,
            resolution: usize,
        }

        let mut ui_state = UIState {
            generate: false,
            scale: 1.0,
            height: 20.0,
            y_smoothing: 5.0,
            resolution: 512,
        };
        let mut last_state = ui_state.clone();
        let mut update_mesh = false;
        let mut regenerate = true;
        let mut regenerate_brushes = true;
        let mut brushes = Vec::new();

        let mesh_size = 256.0;
        move |event, world| {
            match &event {
                Event::KappEvent(e) => {
                    if ui_manager.handle_event(e, &mut ui_state, &mut standard_context) {
                        return true;
                    }
                    if ui_state != last_state {
                        regenerate = true;
                        last_state = ui_state;
                    }
                }
                _ => {}
            }

            if ui_state.generate {
                ui_state.generate = false;
                offset = Vec3::new(random.f32(), 0.0, random.f32()) * 8000.0;
                regenerate = true;
            }
            match event {
                Event::KappEvent(KappEvent::KeyDown { key: Key::P, .. }) => {
                    pathfinder.pathfind(
                        Vec2u::fill(0),
                        Vec2u::fill(ui_state.resolution - 1),
                        Vec2u::fill(ui_state.resolution),
                        &terrain.heights,
                        &mut current_path,
                    );
                }
                Event::KappEvent(KappEvent::KeyDown { key: Key::Up, .. }) => {
                    // y_smoothing += 0.5;
                    // y_smoothing = y_smoothing.max(0.0);
                    brush_count = 20;
                    regenerate = true;
                }
                Event::KappEvent(KappEvent::KeyDown { key: Key::Down, .. }) => {
                    // y_smoothing -= 0.5;
                    // y_smoothing = y_smoothing.max(0.0);
                    brush_count = 0;
                    regenerate = true;
                }
                Event::KappEvent(KappEvent::KeyDown {
                    key: Key::Space, ..
                }) => {
                    ui_state.generate = true;
                }
                Event::KappEvent(KappEvent::PointerDown {
                    x,
                    y,
                    button: PointerButton::Primary,
                    ..
                }) => {
                    (|meshes: &Assets<Mesh>, cameras: Query<(&Transform, &Camera)>| {
                        let (camera_transform, camera) = cameras.iter().next().unwrap();
                        let pointer_ray = camera.view_to_ray(camera_transform, x as f32, y as f32);
                        let mesh = meshes.get(&new_meshes[0]);
                        if let Some(intersection) = intersections::ray_with_mesh(
                            pointer_ray,
                            &mesh.mesh_data.as_ref().unwrap().positions,
                            &mesh.mesh_data.as_ref().unwrap().indices,
                        ) {
                            let offset = Vec3::ZERO;
                            let position = pointer_ray.get_point(intersection);
                            let new_position = ((position.xz() / mesh_size)
                                * ui_state.resolution as f32)
                                .as_usize();

                            println!("new_position: {:?}", new_position);
                            if let Some(start) = last_clicked_position {
                                pathfinder.pathfind(
                                    start,
                                    new_position,
                                    Vec2u::fill(ui_state.resolution),
                                    &terrain.heights,
                                    &mut current_path,
                                );

                                last_clicked_position = None;

                                let path_color = Color::BROWN.with_lightness(0.8).with_chroma(0.3);
                                for p in current_path.iter() {
                                    terrain.colors[p.x * ui_state.resolution + p.y] = path_color;
                                }

                                update_mesh = true;
                            } else {
                                last_clicked_position = Some(new_position);
                            }
                        }
                    })
                    .run(world);
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
                    if regenerate {
                        request_window_redraw(world);

                        if regenerate_brushes {
                            let mut random = Random::new();
                            brushes.clear();
                            let max_size = 500.0 as f32;
                            for _ in 0..brush_count {
                                let inner_radius = random.range_f32(0.0..40.);

                                brushes.push(Brush::new(
                                    Vec2::new(random.f32(), random.f32()) * max_size
                                        + Vec2::new(offset.x, offset.z),
                                    inner_radius,
                                    inner_radius * 3.0,
                                    random.f32() * 0.3,
                                    match random.range_u32(0..2) {
                                        0 => {
                                            let v = random.range_f32(0.4..5.0);
                                            Box::new(move |p| p.y * v)
                                        }
                                        1 => {
                                            let v = random.range_f32(-10.0..15.);
                                            Box::new(move |_| v)
                                        }
                                        _ => {
                                            let random_min = random.range_f32(-3.0..15.0);
                                            let random_max =
                                                random.range_f32(0.0..4.0) + random_min;
                                            Box::new(move |p: Vec3| {
                                                if p.y > random_min && p.y < random_max {
                                                    random_max
                                                } else {
                                                    p.y
                                                }
                                            })
                                        } // _ => unreachable!(),
                                    },
                                ))
                            }
                            regenerate_brushes = false;
                        }

                        terrain = generate_terrain_data(
                            offset,
                            ui_state.scale,
                            ui_state.height,
                            ui_state.y_smoothing,
                            ui_state.resolution,
                            &brushes,
                        );

                        regenerate = false;
                        update_mesh = true;
                    }

                    if update_mesh {
                        last_clicked_position = None;
                        (|graphics: &mut Graphics, meshes: &mut Assets<Mesh>| {
                            for i in 0..tiles {
                                for j in 0..tiles {
                                    let mesh_data = generate_terrain_mesh(
                                        ui_state.resolution,
                                        mesh_size,
                                        &terrain,
                                    );

                                    *meshes.get_mut(&new_meshes[i * tiles + j]) =
                                        Mesh::new(graphics, mesh_data);
                                }
                            }
                        })
                        .run(world);
                        update_mesh = false;
                    }

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

pub fn calculate_normals(mesh_data: &mut MeshData) {
    let mut normal_use_count = vec![0; mesh_data.positions.len()];
    let mut normals = vec![Vec3::ZERO; mesh_data.positions.len()];
    for [p0, p1, p2] in mesh_data.indices.iter().cloned() {
        let dir0 = mesh_data.positions[p1 as usize] - mesh_data.positions[p0 as usize];
        let dir1 = mesh_data.positions[p2 as usize] - mesh_data.positions[p1 as usize];
        let normal = dir0.cross(dir1).normalized();
        normal_use_count[p0 as usize] += 1;
        normal_use_count[p1 as usize] += 1;
        normal_use_count[p2 as usize] += 1;
        normals[p0 as usize] += normal;
        normals[p1 as usize] += normal;
        normals[p2 as usize] += normal;
    }

    for (normal, &normal_use_count) in normals.iter_mut().zip(normal_use_count.iter()) {
        *normal = *normal / normal_use_count as f32;
    }

    mesh_data.normals = normals;
}
