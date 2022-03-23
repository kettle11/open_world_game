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
fn generate_terrain_data(
    offset: Vec3,
    scale: f32,
    height: f32,
    y_smoothing: f32,
    resolution: usize,
    //  brushes: &[Brush],
) -> Terrain {
    let mut noise = clatter::Simplex2d::new();

    let mut heights = Vec::with_capacity(resolution * resolution);
    let mut colors = Vec::with_capacity(resolution * resolution);
    let resolution_scale = 512.0 / resolution as f32;
    let scale = scale * resolution_scale;

    for i in 0..resolution {
        for j in 0..resolution {
            let p = Vec3::new(i as f32, 0.0, j as f32) * scale + offset;

            let y_raw = sample_with_octaves::<16>(&mut noise, 0.5, p.x / 250., p.z / 250.);

            let y_tweak = 2.0;
            let y = y_raw * height * y_tweak;

            let t = (y.abs() / y_smoothing).min(1.0);
            let mut y = y * t * t;

            if y.abs() < 0.001 {
                y += 0.01;
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
            heights.push(y / scale)
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
            let p = Vec3::new(i as f32, y, j as f32) * resolution_scale;
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

        let mesh_size = 256.0;

        // Spawn a camera
        let center = Vec3::new(mesh_size as f32 / 2.0, 0.0, mesh_size as f32 / 2.0);
        let mut camera = Camera::new();
        camera.set_near_plane(1.0);
        world.spawn((
            Transform::new()
                .with_position(center + Vec3::Y * 100.0 + Vec3::Z * 100.)
                .looking_at(center, Vec3::Y),
            CameraControls::new(),
            camera,
        ));

        // Setup the water plane
        let material = (|materials: &mut Assets<Material>| {
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

        let mut random = Random::new();
        let mut offset = Vec3::new(random.f32(), 0.0, random.f32()) * 100.0;

        let mut terrain = Terrain {
            heights: Vec::new(),
            colors: Vec::new(),
        };

        // Setup the UI
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
                        slider(|data: &mut UIState| &mut data.scale, 0.03, 2.0),
                        text("Height Scale:"),
                        slider(|data: &mut UIState| &mut data.height, 1.0, 70.0),
                        text("Coastal Flattening:"),
                        slider(|data: &mut UIState| &mut data.y_smoothing, 0.1, 50.0),
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
            generate: true,
            scale: 0.2,
            height: 10.0,
            y_smoothing: 5.0,
            resolution: 512,
        };
        let mut last_state = ui_state.clone();
        let mut update_mesh = false;
        let mut regenerate = true;

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
                    if regenerate {
                        request_window_redraw(world);

                        terrain = generate_terrain_data(
                            offset,
                            ui_state.scale,
                            ui_state.height,
                            ui_state.y_smoothing,
                            ui_state.resolution,
                        );

                        regenerate = false;
                        update_mesh = true;
                    }

                    if update_mesh {
                        // last_clicked_position = None;
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
