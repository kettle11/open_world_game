use std::{cmp::Ordering, collections::BTreeSet};

use koi::{Vec2i, Vec2u};

#[derive(Clone)]
struct NodeInfo {
    came_from: Option<Vec2u>,
    g_score: f32,
    in_open_set_with_f_score: Option<f32>,
}
pub struct Pathfinder {
    open_set: BTreeSet<OpenSetEntry>,
    node_info: Vec<NodeInfo>,
}

#[derive(Clone, Copy)]
struct OpenSetEntry {
    f_score: f32,
    node: Vec2u,
}

impl Ord for OpenSetEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f_score.partial_cmp(&other.f_score).unwrap() {
            Ordering::Equal => match self.node.x.cmp(&other.node.x) {
                Ordering::Equal => self.node.y.cmp(&other.node.y),
                o => o,
            },
            o => o,
        }
    }
}

impl PartialOrd for OpenSetEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for OpenSetEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}

impl Eq for OpenSetEntry {}

impl Pathfinder {
    pub fn new() -> Self {
        Self {
            open_set: BTreeSet::new(),
            node_info: Vec::new(),
        }
    }
}

impl Pathfinder {
    fn clear_and_reserve(&mut self, size: Vec2u) {
        let size = size.x * size.y;
        self.open_set.clear();
        self.node_info.clear();
        self.node_info.resize(
            size,
            NodeInfo {
                came_from: None,
                g_score: f32::INFINITY,
                in_open_set_with_f_score: None,
            },
        );
    }

    /// Returns a path from goal to start in path_out.
    pub fn pathfind(
        &mut self,
        start: Vec2u,
        goal: Vec2u,
        size: Vec2u,
        heights: &[f32],
        path_out: &mut Vec<Vec2u>,
    ) {
        fn heuristic(n: Vec2u, goal: Vec2u) -> f32 {
            let n = n.as_f32();
            let goal = goal.as_f32();
            let v = (n - goal).abs();
            v.x + v.y
        }
        fn traversal_cost_between(
            current: Vec2u,
            neighbor: Vec2u,
            size: Vec2u,
            heights: &[f32],
        ) -> f32 {
            let current_height = heights[current.x * size.y + current.y];
            let neighbor_height = heights[neighbor.x * size.y + neighbor.y];
            let diff = (current_height - neighbor_height).abs();
            let mut cost = (current_height - neighbor_height).abs() * 2.0;
            if diff < 0.01 {
                cost = 0.0;
            }
            if diff > 0.1 {
                cost = f32::MAX;
            }
            if diff > 0.04 {
                cost *= 10.0;
            }

            if neighbor_height < 0.0 {
                cost = 100.0;
            }
            cost + (current.as_f32() - neighbor.as_f32()).length()
        }
        path_out.clear();
        self.clear_and_reserve(size);
        let start_f_score = heuristic(start, goal);
        self.node_info[start.x * size.y + start.y] = NodeInfo {
            g_score: 0.0,
            came_from: None,
            in_open_set_with_f_score: Some(start_f_score),
        };
        self.open_set.insert(OpenSetEntry {
            node: start,
            f_score: start_f_score,
        });

        while let Some(&current) = self.open_set.iter().next().clone() {
            self.open_set.remove(&current);

            let current = current.node;

            let current_node_info = &mut self.node_info[current.x * size.y + current.y];
            current_node_info.in_open_set_with_f_score = None;

            let current_g_score = current_node_info.g_score;

            if current == goal {
                path_out.push(current);
                let mut current_node_info = &self.node_info[current.x * size.y + current.y];
                while let Some(came_from) = current_node_info.came_from {
                    current_node_info = &self.node_info[came_from.x * size.y + came_from.y];
                    path_out.push(came_from);
                }
                return;
            }

            // println!("CURRENT: {:?}", current);
            // For each neighbor
            for offset in [
                Vec2i::new(-1, 0),
                Vec2i::new(1, 0),
                Vec2i::new(0, 1),
                Vec2i::new(0, -1),
                Vec2i::new(-1, -1),
                Vec2i::new(-1, 1),
                Vec2i::new(1, 1),
                Vec2i::new(1, -1),
            ] {
                let neighbor = (current.as_i32() + offset)
                    .clamp(Vec2i::ZERO, size.as_i32() - Vec2i::ONE)
                    .as_usize();

                // The clamp will result in self if on the edge.
                if neighbor == current {
                    continue;
                }

                let tentative_g_score =
                    current_g_score + traversal_cost_between(current, neighbor, size, heights);

                if tentative_g_score == f32::MAX {
                    continue;
                }
                let neighbor_info = &mut self.node_info[neighbor.x * size.y + neighbor.y];
                if tentative_g_score < neighbor_info.g_score {
                    if let Some(previous_f_score) = neighbor_info.in_open_set_with_f_score {
                        self.open_set.remove(&OpenSetEntry {
                            f_score: previous_f_score,
                            node: neighbor,
                        });
                    }

                    let f_score = tentative_g_score + heuristic(neighbor, goal);

                    self.open_set.insert(OpenSetEntry {
                        f_score,
                        node: neighbor,
                    });
                    *neighbor_info = NodeInfo {
                        g_score: tentative_g_score,
                        came_from: Some(current),
                        in_open_set_with_f_score: Some(f_score),
                    };
                }
            }
        }
    }
}

#[test]
fn pathfind() {
    let mut pathfinder = Pathfinder::new();
    const SIZE: usize = 4;
    let mut path_out = Vec::new();
    pathfinder.pathfind(
        Vec2u::fill(1),
        Vec2u::fill(3),
        Vec2u::fill(SIZE),
        &[1.0; SIZE * SIZE],
        &mut path_out,
    );

    println!("PATH OUT: {:#?}", path_out);
}
