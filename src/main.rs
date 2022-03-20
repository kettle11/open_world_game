#![feature(portable_simd)]

use koi::*;

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
    scale: f32,
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
        core::simd::Simd::<f32, LANES>::splat(x * scale) * frequencies,
        core::simd::Simd::<f32, LANES>::splat(y * scale) * frequencies,
    ]) * amplitudes;

    sample.value.reduce_sum() / max_value
}
fn generate_terrain(
    offset: Vec3,
    size: usize,
    scale: f32,
    height: f32,
    y_smoothing: f32,
    resolution: usize,
    mesh_offset: Vec3,
    brushes: &[Brush],
) -> MeshData {
    // Crater
    /*
    Brush::new(
        Vec2::fill(200.) * scale + Vec2::new(offset.x, offset.z),
        50. * scale,
        300. * scale,
        |_| -10.0,
    ) */
    /*
    let brushes = [
        Brush::new(
            Vec2::fill(200.) * scale + Vec2::new(offset.x, offset.z),
            50. * scale,
            300. * scale,
            1.0,
            |p| p.y * 2.0,
        ),
        Brush::new(
            Vec2::fill(300.) * scale + Vec2::new(offset.x, offset.z),
            000. * scale,
            200. * scale,
            0.8,
            |p| 5.0,
        ),
    ];
    */
    let mut noise = clatter::Simplex2d::new();

    let mut positions = Vec::with_capacity(resolution * resolution);
    let mut colors = Vec::with_capacity(resolution * resolution);

    let mut indices = Vec::with_capacity(2 * resolution * resolution);
    for i in 0..resolution {
        for j in 0..resolution {
            let i = (i as f32 / resolution as f32) * size as f32;
            let j = (j as f32 / resolution as f32) * size as f32;

            let base_pos = Vec3::new(i as f32, 0.0, j as f32) * scale;
            let p = base_pos + offset;

            let y_raw = sample_with_octaves::<16>(&mut noise, 0.004, 0.5, p.x, p.z);
            let y_raw = y_raw * height;
            let y = y_raw;

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

            if y.abs() < 0.0001 {
                y += 0.01;
            }

            for brush in brushes.iter() {
                y = brush.apply(Vec2::new(p.x, p.z), y);
            }

            //let t = ((y - 10.0).abs() / y_smoothing).min(1.0);
            //let y = y * t * t;

            let p = Vec3::new(base_pos.x, y, base_pos.z);
            positions.push((p + mesh_offset) * 3.0);

            let color = if y > 20.0 {
                Color::WHITE
            } else if y > 15.0 {
                Color::BLACK.with_lightness(0.7)
            } else if y > 0.1 {
                Color::new_from_bytes(149, 130, 70, 255)
            } else {
                Color::new_from_bytes(197, 167, 132, 255)
            };
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

    let mut mesh_data = MeshData {
        positions,
        indices,
        normals: Vec::new(),
        texture_coordinates: Vec::new(),
        colors,
    };
    calculate_normals(&mut mesh_data);
    mesh_data
}

fn main() {
    App::new().setup_and_run(|world: &mut World| {
        world
            .get_singleton::<Graphics>()
            .set_automatic_redraw(false);

        // The light and shadow caster is spawned as part of this.
        spawn_skybox(world, "assets/qwantani_1k.hdr");

        // Spawn a camera
        let size = 500.0;
        let scale = 0.2;
        let center = Vec3::new(size as f32 / 2.0, 0.0, size as f32 / 2.0) * scale;
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
                .with_scale(Vec3::fill(size * 2.5)),
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
        let tiles = 5;
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

        /*
        let worlds = world.get_single_component_mut::<Assets<World>>().unwrap();
        let gltf_world = worlds.load("assets/lowpoly_fish/scene.gltf");

        // Spawn a Handle<World> that will be replaced with the GlTf when it's loaded.
        let gltf_hierarchy = world.spawn(gltf_world);
        let scaled_down = world.spawn(Transform::new().with_scale(Vec3::fill(5.0)));
        set_parent(world, Some(scaled_down), gltf_hierarchy);
        */
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

        let mut y_smoothing: f32 = 5.0;
        let mut offset = Vec3::new(random.f32(), 0.0, random.f32()) * 100.0;
        let mut regenerate = true;

        move |event, world| {
            match event {
                Event::KappEvent(KappEvent::KeyDown { key: Key::Up, .. }) => {
                    y_smoothing += 0.5;
                    y_smoothing = y_smoothing.max(0.0);
                    regenerate = true;
                }
                Event::KappEvent(KappEvent::KeyDown { key: Key::Down, .. }) => {
                    y_smoothing -= 0.5;
                    y_smoothing = y_smoothing.max(0.0);
                    regenerate = true;
                }
                Event::KappEvent(KappEvent::KeyDown {
                    key: Key::Space, ..
                }) => {
                    offset = Vec3::new(random.f32(), 0.0, random.f32()) * 800000.0;
                    regenerate = true;
                }
                Event::KappEvent(event) => {
                    match event {
                        KappEvent::KeyDown { .. }
                        | KappEvent::PointerDown { .. }
                        | KappEvent::PointerUp { .. }
                        | KappEvent::Scroll { .. }
                        | KappEvent::PinchGesture { .. }
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
                    //    println!("DRAWING");
                }
                _ => {}
            }

            if regenerate {
                request_window_redraw(world);

                let mut brushes = Vec::new();
                let mut random = Random::new();

                let max_size = size as f32;
                for _ in 0..10 {
                    let inner_radius = random.range_f32(0.0..200.);

                    brushes.push(Brush::new(
                        Vec2::new(random.f32(), random.f32()) * max_size
                            + Vec2::new(offset.x, offset.z),
                        random.range_f32(0.0..150.),
                        inner_radius + random.range_f32(30.0..150.),
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
                                let random_max = random.range_f32(0.0..4.0) + random_min;
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

                (|graphics: &mut Graphics, meshes: &mut Assets<Mesh>| {
                    let resolutions = [100, 50, 25];
                    for i in 0..tiles {
                        for j in 0..tiles {
                            let offset =
                                offset + Vec3::new(i as f32, 0.0, j as f32) * (size - 8.0) * scale;
                            let mesh_data = generate_terrain(
                                offset,
                                size as usize,
                                scale,
                                size * scale * 0.2,
                                y_smoothing,
                                if i == tiles / 2 && j == tiles / 2 {
                                    512
                                } else {
                                    64
                                },
                                Vec3::new(i as f32, 0.0, j as f32) * (size - 8.0) * scale
                                    - (Vec3::new(tiles as f32, 0.0, tiles as f32)
                                        * (size - 8.0)
                                        * scale)
                                        / 2.0,
                                &brushes,
                            );
                            *meshes.get_mut(&new_meshes[i * tiles + j]) =
                                Mesh::new(graphics, mesh_data);
                        }
                    }
                })
                .run(world);
                regenerate = false;
            }
            false
        }
    });
}

pub fn calculate_normals(mesh_data: &mut MeshData) {
    let mut normals = vec![Vec3::ZERO; mesh_data.positions.len()];
    for [p0, p1, p2] in mesh_data.indices.iter().cloned() {
        let dir0 = mesh_data.positions[p1 as usize] - mesh_data.positions[p0 as usize];
        let dir1 = mesh_data.positions[p2 as usize] - mesh_data.positions[p1 as usize];
        let normal = dir0.cross(dir1).normalized();
        normals[p0 as usize] = normal;
        normals[p1 as usize] = normal;
        normals[p2 as usize] = normal;
    }

    mesh_data.normals = normals;
}
