#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "container.h"
#include "sph_baseline.h"

Cup source_cup;
Cup receiver_cup;
std::vector<Particle> boundary_particles;

static float max_source_penetration = 0.0f;
static float max_receiver_penetration = 0.0f;
static int below_source_bottom_count = 0;
static int below_receiver_bottom_count = 0;
static int inside_source_wall_count = 0;
static int inside_receiver_wall_count = 0;

// Build one cup from geometry and tilt inputs
Cup make_cup(float center_x, float center_y, float center_z, float width,
             float height, float depth, float wall_thickness, float tilt_deg) {

  // Fill the basic cup fields
  Cup cup;
  cup.center_x = center_x;
  cup.center_y = center_y;
  cup.center_z = center_z;
  cup.width = width;
  cup.height = height;
  cup.depth = depth;
  cup.wall_thickness = wall_thickness;
  cup.tilt_deg = tilt_deg;
  cup.angular_velocity = 0.0f;

  // Precompute rotation values for fast transforms
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);

  return cup;
}

// Update one cup rotation after the tilt changes
void update_cup_rotation(Cup &cup, float tilt_deg) {

  // Store the new tilt angle
  cup.tilt_deg = tilt_deg;

  // Recompute the rotation values
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);
}

// Convert a cup local point into world coordinates
static void cup_local_to_world(const Cup &cup, float local_x, float local_y,
                               float &world_x, float &world_y) {

  // Rotate and shift the point into world space
  world_x = cup.center_x + local_x * cup.cos_t + local_y * cup.sin_t;
  world_y = cup.center_y - local_x * cup.sin_t + local_y * cup.cos_t;
}

// Convert a world point into cup local coordinates
static void world_to_cup_local(const Cup &cup, float world_x, float world_y,
                               float &local_x, float &local_y) {

  // Shift the point into the cup frame
  float dx = world_x - cup.center_x;
  float dy = world_y - cup.center_y;

  // Rotate the point into local coordinates
  local_x = dx * cup.cos_t - dy * cup.sin_t;
  local_y = dx * cup.sin_t + dy * cup.cos_t;
}

// Convert a world velocity into cup local velocity
static void world_velocity_to_cup_local(const Cup &cup, float world_vx,
                                        float world_vy, float &local_vx,
                                        float &local_vy) {

  // Rotate the velocity into the cup frame
  local_vx = world_vx * cup.cos_t - world_vy * cup.sin_t;
  local_vy = world_vx * cup.sin_t + world_vy * cup.cos_t;
}

// Convert a cup local velocity back into world velocity
static void cup_velocity_to_world(const Cup &cup, float local_vx,
                                  float local_vy, float &world_vx,
                                  float &world_vy) {

  // Rotate the velocity back into world space
  world_vx = local_vx * cup.cos_t + local_vy * cup.sin_t;
  world_vy = -local_vx * cup.sin_t + local_vy * cup.cos_t;
}

// Compute wall velocity from cup rotation
static void get_local_wall_velocity(const Cup &cup, float local_x,
                                    float local_y, float &wall_vx,
                                    float &wall_vy) {

  // Build rigid body wall motion in the cup frame
  wall_vx = -cup.angular_velocity * local_y;
  wall_vy = cup.angular_velocity * local_x;
}

// Add one boundary particle to the render boundary array
static void add_boundary_particle(float x, float y, float z, int kind) {

  // Fill the boundary particle values
  Particle particle;
  particle.x = x;
  particle.y = y;
  particle.z = z;
  particle.vx = 0.0f;
  particle.vy = 0.0f;
  particle.vz = 0.0f;
  particle.fx = 0.0f;
  particle.fy = 0.0f;
  particle.fz = 0.0f;
  particle.rho = REST_DENS;
  particle.p = 0.0f;
  particle.is_boundary = true;
  particle.kind = kind;

  // Store the boundary particle
  boundary_particles.push_back(particle);
}

// Add one jittered fluid particle to the fluid array
static void add_fluid_particle(float x, float y, float z) {

  // Build random position jitter
  float jitter_scale = POS_JITTER_CONST * INITIAL_PARTICLE_SPACING;
  float jitter_x =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_y =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_z =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

  // Fill the fluid particle values
  Particle particle;
  particle.x = x + jitter_x;
  particle.y = y + jitter_y;
  particle.z = z + jitter_z;
  particle.vx =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
  particle.vy =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
  particle.vz =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
  particle.fx = 0.0f;
  particle.fy = 0.0f;
  particle.fz = 0.0f;
  particle.rho = REST_DENS;
  particle.p = 0.0f;
  particle.is_boundary = false;
  particle.kind = 0;

  // Store the fluid particle
  fluid_particles.push_back(particle);
}

// Add one rectangular cup surface as render particles
static void add_cup_surface(const Cup &cup, float local_x_min,
                            float local_x_max, float local_y_min,
                            float local_y_max, float z_min, float z_max,
                            int kind) {

  // Sweep across the local x range
  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * CUP_RENDER_SPACING;
       local_x += CUP_RENDER_SPACING) {

    // Sweep across the local y range
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * CUP_RENDER_SPACING;
         local_y += CUP_RENDER_SPACING) {

      // Sweep across the depth range
      for (float z = z_min; z <= z_max + 0.5f * CUP_RENDER_SPACING;
           z += CUP_RENDER_SPACING) {

        // Convert the local point to world space
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the render boundary particle
        add_boundary_particle(world_x, world_y, z, kind);
      }
    }
  }
}

// Add all render wall particles for one cup
static void add_cup_render_particles(const Cup &cup, int kind) {

  // Build several outer wall layers for rendering
  int num_layers = 4; // Fills the H=0.04 radius

  for (int layer = 0; layer < num_layers; layer++) {

    // Expand the cup geometry for this render layer
    float offset = layer * CUP_RENDER_SPACING;
    float current_width = cup.width + (2.0f * offset);
    float current_depth = cup.depth + (2.0f * offset);
    float current_y_min = 0.0f - offset;

    float z_min = cup.center_z - 0.5f * current_depth;
    float z_max = cup.center_z + 0.5f * current_depth;

    // Add the bottom wall
    add_cup_surface(cup, -0.5f * current_width, 0.5f * current_width,
                    current_y_min, current_y_min, z_min, z_max, kind);

    // Add the left wall
    add_cup_surface(cup, -0.5f * current_width, -0.5f * current_width,
                    current_y_min, cup.height, z_min, z_max, kind);

    // Add the right wall
    add_cup_surface(cup, 0.5f * current_width, 0.5f * current_width,
                    current_y_min, cup.height, z_min, z_max, kind);

    // Add the front wall as a sheet at z min
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * CUP_RENDER_SPACING;
         local_x += CUP_RENDER_SPACING) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
           local_y += CUP_RENDER_SPACING) {
        float world_x, world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);
        add_boundary_particle(world_x, world_y, z_min, kind);
      }
    }

    // Add the back wall as a sheet at z max
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * CUP_RENDER_SPACING;
         local_x += CUP_RENDER_SPACING) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
           local_y += CUP_RENDER_SPACING) {
        float world_x, world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);
        add_boundary_particle(world_x, world_y, z_max, kind);
      }
    }
  }
}

// Add all render particles for one cup
// static void add_cup_render_particles(const Cup &cup, int kind) {
//
//     // Build the depth limits
//     float z_min = cup.center_z - 0.5f * cup.depth;
//     float z_max = cup.center_z + 0.5f * cup.depth;
//
//     // Add the bottom surface
//     add_cup_surface(cup, -0.5f * cup.width, 0.5f * cup.width, 0.0f, 0.0f,
//     z_min, z_max, kind);
//
//     // Add the left wall surface
//     add_cup_surface(cup, -0.5f * cup.width, -0.5f * cup.width, 0.0f,
//     cup.height, z_min, z_max, kind);
//
//     // Add the right wall surface
//     add_cup_surface(cup, 0.5f * cup.width, 0.5f * cup.width, 0.0f,
//     cup.height, z_min, z_max, kind);
//
//     // Add the front wall surface
//     for (float local_x = -0.5f * cup.width; local_x <= 0.5f * cup.width +
//     0.5f * CUP_RENDER_SPACING; local_x += CUP_RENDER_SPACING) {
//         for (float local_y = 0.0f; local_y <= cup.height + 0.5f *
//         CUP_RENDER_SPACING; local_y += CUP_RENDER_SPACING) {
//             float world_x;
//             float world_y;
//             cup_local_to_world(cup, local_x, local_y, world_x, world_y);
//             add_boundary_particle(world_x, world_y, z_min, kind);
//         }
//     }
//
//     // Add the back wall surface
//     for (float local_x = -0.5f * cup.width; local_x <= 0.5f * cup.width +
//     0.5f * CUP_RENDER_SPACING; local_x += CUP_RENDER_SPACING) {
//         for (float local_y = 0.0f; local_y <= cup.height + 0.5f *
//         CUP_RENDER_SPACING; local_y += CUP_RENDER_SPACING) {
//             float world_x;
//             float world_y;
//             cup_local_to_world(cup, local_x, local_y, world_x, world_y);
//             add_boundary_particle(world_x, world_y, z_max, kind);
//         }
//     }
// }

// Fill the source cup interior with fluid particles
static void add_fluid_in_source_cup() {

  // Build the inner fluid fill limits in the source cup frame
  float local_x_min = -0.5f * source_cup.width + source_cup.wall_thickness +
                      INITIAL_PARTICLE_SPACING;
  float local_x_max = 0.5f * source_cup.width - source_cup.wall_thickness -
                      INITIAL_PARTICLE_SPACING;
  float local_y_min = source_cup.wall_thickness + INITIAL_PARTICLE_SPACING;
  float local_y_max =
      source_cup.wall_thickness +
      SOURCE_FILL_RATIO * (source_cup.height - source_cup.wall_thickness -
                           INITIAL_PARTICLE_SPACING);
  float z_min = source_cup.center_z - 0.5f * source_cup.depth +
                source_cup.wall_thickness + INITIAL_PARTICLE_SPACING;
  float z_max = source_cup.center_z + 0.5f * source_cup.depth -
                source_cup.wall_thickness - INITIAL_PARTICLE_SPACING;

  // Sweep across the cup interior volume
  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * INITIAL_PARTICLE_SPACING;
       local_x += INITIAL_PARTICLE_SPACING) {
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * INITIAL_PARTICLE_SPACING;
         local_y += INITIAL_PARTICLE_SPACING) {
      for (float z = z_min; z <= z_max + 0.5f * INITIAL_PARTICLE_SPACING;
           z += INITIAL_PARTICLE_SPACING) {

        // Convert the local fluid point to world space
        float world_x;
        float world_y;
        cup_local_to_world(source_cup, local_x, local_y, world_x, world_y);

        // Add the fluid particle
        add_fluid_particle(world_x, world_y, z);
      }
    }
  }
}

// Rebuild the boundary particles used for export
void rebuild_boundary_particles_for_export() {

  // Clear the old render boundary
  boundary_particles.clear();

  // Rebuild the source cup render boundary
  add_cup_render_particles(source_cup, 1);

  // Rebuild the receiver cup render boundary
  add_cup_render_particles(receiver_cup, 2);
}

// Compute the source cup tilt for one frame or substep
static float get_source_tilt_for_frame(float frame_index,
                                       float target_tilt_deg) {

  // Hold the source cup fixed upright
  if (DEBUG_NO_TILT) {
    return 0.0f;
  }

  // Force the source cup to the final angle immediately
  if (DEBUG_STATIC_TILT) {
    return target_tilt_deg;
  }

  // Keep the cup upright during settling
  if (frame_index < SETTLE_FRAMES) {
    return 0.0f;

  // Ramp the tilt linearly during the tilt interval
  } else if (frame_index < (SETTLE_FRAMES + TILT_FRAMES)) {
    float alpha = (frame_index - static_cast<float>(SETTLE_FRAMES)) /
                  static_cast<float>(TILT_FRAMES);
    return alpha * target_tilt_deg;

  // Hold the cup at the final target tilt
  } else {
    return target_tilt_deg;
  }
}

// Build the full starting scene
void initialize_scene(float target_tilt_deg) {

  // Clear all particle arrays
  fluid_particles.clear();
  boundary_particles.clear();

  // Reserve storage for expected particle counts
  fluid_particles.reserve(12000);
  boundary_particles.reserve(80000);

  // Choose the starting source tilt
  float initial_source_tilt_deg = 0.0f;

  if (DEBUG_STATIC_TILT) {
    initial_source_tilt_deg = target_tilt_deg;
  }

  if (DEBUG_NO_TILT) {
    initial_source_tilt_deg = 0.0f;
  }

  // Build the two cups at their starting orientations
  source_cup =
      make_cup(SOURCE_CUP_CENTER_X, SOURCE_CUP_CENTER_Y, SOURCE_CUP_CENTER_Z,
               SOURCE_CUP_WIDTH, SOURCE_CUP_HEIGHT, SOURCE_CUP_DEPTH,
               CUP_WALL_THICKNESS, initial_source_tilt_deg);
  receiver_cup =
      make_cup(RECEIVER_CUP_CENTER_X, RECEIVER_CUP_CENTER_Y,
               RECEIVER_CUP_CENTER_Z, RECEIVER_CUP_WIDTH, RECEIVER_CUP_HEIGHT,
               RECEIVER_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);

  // Fill the source cup with the starting fluid
  add_fluid_in_source_cup();

  // Build the first render boundary particles
  rebuild_boundary_particles_for_export();
}

// Update the cup state for one substep
void update_scene_for_frame(float frame_index, float target_tilt_deg) {

  // Compute the current source tilt
  float current_tilt_deg =
      get_source_tilt_for_frame(frame_index, target_tilt_deg);

  // Compute the previous source tilt for angular velocity
  float previous_frame_index =
      frame_index - 1.0f / static_cast<float>(SUBSTEPS_PER_FRAME);

  if (previous_frame_index < 0.0f) {
    previous_frame_index = 0.0f;
  }

  float previous_tilt_deg =
      get_source_tilt_for_frame(previous_frame_index, target_tilt_deg);

  // Compute angular velocity from tilt change
  float delta_theta_rad = (current_tilt_deg - previous_tilt_deg) * PI / 180.0f;
  source_cup.angular_velocity = delta_theta_rad / DT;
  receiver_cup.angular_velocity = 0.0f;

  // Update the two cup rotations
  update_cup_rotation(source_cup, current_tilt_deg);
  update_cup_rotation(receiver_cup, 0.0f);
}

// Reset all penetration counters for a new frame
void reset_penetration_stats() {

  // Clear max penetration values
  max_source_penetration = 0.0f;
  max_receiver_penetration = 0.0f;

  // Clear count based debug values
  below_source_bottom_count = 0;
  below_receiver_bottom_count = 0;
  inside_source_wall_count = 0;
  inside_receiver_wall_count = 0;
}

// Print penetration counters for the current frame
void print_penetration_stats(int frame_index) {

  // Print the current debug values
  std::cout << "Frame " << frame_index
            << " | max_source_penetration = " << max_source_penetration
            << " | max_receiver_penetration = " << max_receiver_penetration
            << " | below_source_bottom_count = " << below_source_bottom_count
            << " | below_receiver_bottom_count = "
            << below_receiver_bottom_count
            << " | inside_source_wall_count = " << inside_source_wall_count
            << " | inside_receiver_wall_count = " << inside_receiver_wall_count
            << std::endl;
}

// Check that the source cup starts with fluid inside its valid bounds
void print_source_cup_setup_stats() {

  // Build the interior source cup limits
  float inner_left = -0.5f * source_cup.width + source_cup.wall_thickness;
  float inner_right = 0.5f * source_cup.width - source_cup.wall_thickness;
  float inner_bottom = source_cup.wall_thickness;
  float inner_front =
      source_cup.center_z - 0.5f * source_cup.depth + source_cup.wall_thickness;
  float inner_back =
      source_cup.center_z + 0.5f * source_cup.depth - source_cup.wall_thickness;

  // Start the invalid particle count
  int outside_count = 0;

  // Check every fluid particle against the source cup interior
  for (const Particle &particle : fluid_particles) {
    float local_x;
    float local_y;
    world_to_cup_local(source_cup, particle.x, particle.y, local_x, local_y);

    bool outside_x = (local_x < inner_left) || (local_x > inner_right);
    bool outside_y = (local_y < inner_bottom) || (local_y > source_cup.height);
    bool outside_z = (particle.z < inner_front) || (particle.z > inner_back);

    if (outside_x || outside_y || outside_z) {
      outside_count++;
    }
  }

  // Print the initial source cup setup result
  std::cout << "Initial source cup check"
            << " | fluid_count = " << fluid_particles.size()
            << " | outside_count = " << outside_count << std::endl;
}

// Resolve one particle against one cup interior
void resolve_cup_collision(Particle &particle, const Cup &cup) {

  // Skip boundary particles
  if (particle.is_boundary) {
    return;
  }

  // Pick the matching debug counters for this cup
  float *max_penetration = &max_receiver_penetration;
  int *below_bottom_count = &below_receiver_bottom_count;
  int *inside_wall_count = &inside_receiver_wall_count;

  if (&cup == &source_cup) {
    max_penetration = &max_source_penetration;
    below_bottom_count = &below_source_bottom_count;
    inside_wall_count = &inside_source_wall_count;
  }

  // Move the particle position into the cup frame
  float local_x;
  float local_y;
  world_to_cup_local(cup, particle.x, particle.y, local_x, local_y);

  // Move the particle velocity into the cup frame
  float local_vx;
  float local_vy;
  world_velocity_to_cup_local(cup, particle.vx, particle.vy, local_vx,
                              local_vy);

  // Build the cup box limits
  float outer_left = -0.5f * cup.width;
  float outer_right = 0.5f * cup.width;
  float outer_bottom = 0.0f;
  float outer_top = cup.height;
  float outer_front = cup.center_z - 0.5f * cup.depth;
  float outer_back = cup.center_z + 0.5f * cup.depth;

  float inner_left = outer_left;
  float inner_right = outer_right;
  float inner_bottom = outer_bottom;
  float inner_front = outer_front;
  float inner_back = outer_back;

  // Count particles below the cup bottom before correction
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom - 1e-4f)) {
    (*below_bottom_count)++;
  }

  // Track the closest wall violation
  float best_penetration = 1e30f;
  int best_wall = 0;

  // Check penetration against the bottom wall
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom)) {
    float penetration = inner_bottom - local_y;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 1;
    }
  }

  // Check penetration against the left wall
  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x < inner_left)) {
    float penetration = inner_left - local_x;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 2;
    }
  }

  // Check penetration against the right wall
  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x > inner_right)) {
    float penetration = local_x - inner_right;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 3;
    }
  }

  // Check penetration against the front wall
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z < inner_front)) {
    float penetration = inner_front - particle.z;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 4;
    }
  }

  // Check penetration against the back wall
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z > inner_back)) {
    float penetration = particle.z - inner_back;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 5;
    }
  }

  // Update debug counts if the particle was inside wall material
  if (best_wall != 0) {
    (*inside_wall_count)++;
    *max_penetration = std::max(*max_penetration, best_penetration);
  }

  // Resolve bottom wall penetration
  if (best_wall == 1) {

    // Push the particle above the bottom wall
    local_y = inner_bottom + WALL_EPS;

    // Build the local wall velocity
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect inward normal motion
    float relative_vy = local_vy - wall_vy;

    if (relative_vy < 0.0f) {
      local_vy = wall_vy - WALL_RESTITUTION * relative_vy;
    }

    // Dampen the tangential relative velocity
    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;

  // Resolve left wall penetration
  } else if (best_wall == 2) {

    // Push the particle inside the valid left bound
    local_x = inner_left + WALL_EPS;

    // Build the local wall velocity
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect inward normal motion
    float relative_vx = local_vx - wall_vx;

    if (relative_vx < 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }

    // Dampen the tangential relative velocity
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve right wall penetration
  } else if (best_wall == 3) {

    // Push the particle inside the valid right bound
    local_x = inner_right - WALL_EPS;

    // Build the local wall velocity
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect inward normal motion
    float relative_vx = local_vx - wall_vx;

    if (relative_vx > 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }

    // Dampen the tangential relative velocity
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve front wall penetration
  } else if (best_wall == 4) {

    // Push the particle inside the valid front bound
    particle.z = inner_front + WALL_EPS;

    // Reflect z velocity if needed
    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    // Build the wall velocity for consistency
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve back wall penetration
  } else if (best_wall == 5) {

    // Push the particle inside the valid back bound
    particle.z = inner_back - WALL_EPS;

    // Reflect z velocity if needed
    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    // Build the wall velocity for consistency
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;
  }

  // Move the corrected particle position back to world space
  cup_local_to_world(cup, local_x, local_y, particle.x, particle.y);

  // Move the corrected particle velocity back to world space
  cup_velocity_to_world(cup, local_vx, local_vy, particle.vx, particle.vy);
}