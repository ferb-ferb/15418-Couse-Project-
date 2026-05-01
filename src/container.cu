#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "container.h"
#include "sph_baseline.h"

Cup source_cup;
Cup receiver_cup;

Particle *boundary_particles;
int num_boundary_particles = 0;

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

  // Precompute rotation values for local world transforms
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);

  return cup;
}

// Update one cup rotation after the tilt changes
void update_cup_rotation(Cup &cup, float tilt_deg) {

  // Store the new tilt angle
  cup.tilt_deg = tilt_deg;

  // Recompute rotation values for local world transforms
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);
}

// Convert one cup local point into world coordinates
static void cup_local_to_world(const Cup &cup, float local_x, float local_y,
                               float &world_x, float &world_y) {

  // Rotate and shift the local point into world space
  world_x = cup.center_x + local_x * cup.cos_t + local_y * cup.sin_t;
  world_y = cup.center_y - local_x * cup.sin_t + local_y * cup.cos_t;
}

// Convert one world point into cup local coordinates
static void world_to_cup_local(const Cup &cup, float world_x, float world_y,
                               float &local_x, float &local_y) {

  // Shift the point into the cup frame
  float dx = world_x - cup.center_x;
  float dy = world_y - cup.center_y;

  // Rotate the shifted point into local space
  local_x = dx * cup.cos_t - dy * cup.sin_t;
  local_y = dx * cup.sin_t + dy * cup.cos_t;
}

// Convert one world velocity into cup local velocity
static void world_velocity_to_cup_local(const Cup &cup, float world_vx,
                                        float world_vy, float &local_vx,
                                        float &local_vy) {

  // Rotate the velocity into the cup frame
  local_vx = world_vx * cup.cos_t - world_vy * cup.sin_t;
  local_vy = world_vx * cup.sin_t + world_vy * cup.cos_t;
}

// Convert one cup local velocity into world velocity
static void cup_velocity_to_world(const Cup &cup, float local_vx,
                                  float local_vy, float &world_vx,
                                  float &world_vy) {

  // Rotate the velocity back into world space
  world_vx = local_vx * cup.cos_t + local_vy * cup.sin_t;
  world_vy = -local_vx * cup.sin_t + local_vy * cup.cos_t;
}

// Compute local wall velocity from cup rotation
static void get_local_wall_velocity(const Cup &cup, float local_x,
                                    float local_y, float &wall_vx,
                                    float &wall_vy) {

  // Build rigid body wall velocity in the cup frame
  wall_vx = -cup.angular_velocity * local_y;
  wall_vy = cup.angular_velocity * local_x;
}

// Add one boundary particle to the boundary array
static void add_boundary_particle(float x, float y, float z, int kind) {

  // Fill the boundary particle state
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

  // Store the boundary particle if space remains
  if (num_boundary_particles < MAX_BOUNDARY_PARTICLES) {
    boundary_particles[num_boundary_particles] = particle;
    num_boundary_particles++;

  // Warn when the fixed boundary array is full
  } else {
    std::cout << "Warning: Exceeded MAX_BOUNDARY_PARTICLES!" << std::endl;
  }
}

// Add one fluid particle to the fluid array
static void add_fluid_particle(float x, float y, float z) {

  // Build small random position jitter
  float jitter_scale = POS_JITTER_CONST * INITIAL_PARTICLE_SPACING;
  float jitter_x =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_y =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_z =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

  // Fill the fluid particle state
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

  // Store the fluid particle if space remains
  if (num_fluid_particles < MAX_FLUID_PARTICLES) {
    fluid_particles[num_fluid_particles] = particle;
    num_fluid_particles++;

  // Warn when the fixed fluid array is full
  } else {
    std::cout << "Warning: Exceeded MAX_FLUID_PARTICLES!" << std::endl;
  }
}

// Add one rectangular cup surface as boundary particles
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

      // Sweep across the z range
      for (float z = z_min; z <= z_max + 0.5f * CUP_RENDER_SPACING;
           z += CUP_RENDER_SPACING) {

        // Convert the cup local point into world space
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the boundary particle
        add_boundary_particle(world_x, world_y, z, kind);
      }
    }
  }
}

// Add all boundary particles for one cup using multiple wall layers
static void add_cup_render_particles(const Cup &cup, int kind) {

  // Set number of wall layers
  int num_layers = 4; // Fills the H=0.04 radius

  // Build each layer outward from the cup
  for (int layer = 0; layer < num_layers; layer++) {

    // Expand the cup size for this wall layer
    float offset = layer * CUP_RENDER_SPACING;
    float current_width = cup.width + (2.0f * offset);
    float current_depth = cup.depth + (2.0f * offset);
    float current_y_min = 0.0f - offset;

    // Build the depth limits for this layer
    float z_min = cup.center_z - 0.5f * current_depth;
    float z_max = cup.center_z + 0.5f * current_depth;

    // Add the bottom surface
    add_cup_surface(cup, -0.5f * current_width, 0.5f * current_width,
                    current_y_min, current_y_min, z_min, z_max, kind);

    // Add the left wall
    add_cup_surface(cup, -0.5f * current_width, -0.5f * current_width,
                    current_y_min, cup.height, z_min, z_max, kind);

    // Add the right wall
    add_cup_surface(cup, 0.5f * current_width, 0.5f * current_width,
                    current_y_min, cup.height, z_min, z_max, kind);

    // Add the front wall
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * CUP_RENDER_SPACING;
         local_x += CUP_RENDER_SPACING) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
           local_y += CUP_RENDER_SPACING) {

        // Convert the front wall point into world space
        float world_x, world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the front wall particle
        add_boundary_particle(world_x, world_y, z_min, kind);
      }
    }

    // Add the back wall
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * CUP_RENDER_SPACING;
         local_x += CUP_RENDER_SPACING) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
           local_y += CUP_RENDER_SPACING) {

        // Convert the back wall point into world space
        float world_x, world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the back wall particle
        add_boundary_particle(world_x, world_y, z_max, kind);
      }
    }
  }
}

// Add all render particles for one cup
// static void add_cup_render_particles(const Cup &cup, int kind) {
//
//   // Build the depth limits
//   float z_min = cup.center_z - 0.5f * cup.depth;
//   float z_max = cup.center_z + 0.5f * cup.depth;
//
//   // Add the bottom surface
//   add_cup_surface(cup, -0.5f * cup.width, 0.5f * cup.width, 0.0f, 0.0f,
//   z_min,
//                   z_max, kind);
//
//   // Add the left wall surface
//   add_cup_surface(cup, -0.5f * cup.width, -0.5f * cup.width, 0.0f,
//   cup.height,
//                   z_min, z_max, kind);
//
//   // Add the right wall surface
//   add_cup_surface(cup, 0.5f * cup.width, 0.5f * cup.width, 0.0f, cup.height,
//                   z_min, z_max, kind);
//
//   // Add the front wall surface
//   for (float local_x = -0.5f * cup.width;
//        local_x <= 0.5f * cup.width + 0.5f * CUP_RENDER_SPACING;
//        local_x += CUP_RENDER_SPACING) {
//     for (float local_y = 0.0f;
//          local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
//          local_y += CUP_RENDER_SPACING) {
//       float world_x;
//       float world_y;
//       cup_local_to_world(cup, local_x, local_y, world_x, world_y);
//       add_boundary_particle(world_x, world_y, z_min, kind);
//     }
//   }
//
//   // Add the back wall surface
//   for (float local_x = -0.5f * cup.width;
//        local_x <= 0.5f * cup.width + 0.5f * CUP_RENDER_SPACING;
//        local_x += CUP_RENDER_SPACING) {
//     for (float local_y = 0.0f;
//          local_y <= cup.height + 0.5f * CUP_RENDER_SPACING;
//          local_y += CUP_RENDER_SPACING) {
//       float world_x;
//       float world_y;
//       cup_local_to_world(cup, local_x, local_y, world_x, world_y);
//       add_boundary_particle(world_x, world_y, z_max, kind);
//     }
//   }
// }

// Add starting fluid inside the source cup
static void add_fluid_in_source_cup() {

  // Build the inner source cup limits
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

  // Sweep through the cup interior
  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * INITIAL_PARTICLE_SPACING;
       local_x += INITIAL_PARTICLE_SPACING) {
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * INITIAL_PARTICLE_SPACING;
         local_y += INITIAL_PARTICLE_SPACING) {
      for (float z = z_min; z <= z_max + 0.5f * INITIAL_PARTICLE_SPACING;
           z += INITIAL_PARTICLE_SPACING) {

        // Convert this local fluid point into world space
        float world_x;
        float world_y;
        cup_local_to_world(source_cup, local_x, local_y, world_x, world_y);

        // Add the fluid particle
        add_fluid_particle(world_x, world_y, z);
      }
    }
  }
}

// Rebuild all boundary particles for the current cup poses
void rebuild_boundary_particles_for_export() {

  // Clear the old boundary particles
  num_boundary_particles = 0;

  // Rebuild the source cup boundary particles
  add_cup_render_particles(source_cup, 1);

  // Rebuild the receiver cup boundary particles
  add_cup_render_particles(receiver_cup, 2);
}

// Compute the source cup tilt for one frame or substep
static float get_source_tilt_for_frame(float frame_index,
                                       float target_tilt_deg) {

  // Return zero tilt in no tilt debug mode
  if (DEBUG_NO_TILT) {
    return 0.0f;
  }

  // Return target tilt in static tilt debug mode
  if (DEBUG_STATIC_TILT) {
    return target_tilt_deg;
  }

  // Keep cup upright during the settling frames
  if (frame_index < SETTLE_FRAMES) {
    return 0.0f;

  // Linearly ramp the cup tilt during the tilt frames
  } else if (frame_index < (SETTLE_FRAMES + TILT_FRAMES)) {
    float alpha = (frame_index - static_cast<float>(SETTLE_FRAMES)) /
                  static_cast<float>(TILT_FRAMES);
    return alpha * target_tilt_deg;

  // Hold the cup at the target tilt after the ramp
  } else {
    return target_tilt_deg;
  }
}

// Build the full starting scene
void initialize_scene(float target_tilt_deg) {

  // Clear particle counts
  num_fluid_particles = 0;
  num_boundary_particles = 0;

  // Pick the starting source cup tilt
  float initial_source_tilt_deg = 0.0f;

  if (DEBUG_STATIC_TILT) {
    initial_source_tilt_deg = target_tilt_deg;
  }

  if (DEBUG_NO_TILT) {
    initial_source_tilt_deg = 0.0f;
  }

  // Build the source cup at its starting angle
  source_cup =
      make_cup(SOURCE_CUP_CENTER_X, SOURCE_CUP_CENTER_Y, SOURCE_CUP_CENTER_Z,
               SOURCE_CUP_WIDTH, SOURCE_CUP_HEIGHT, SOURCE_CUP_DEPTH,
               CUP_WALL_THICKNESS, initial_source_tilt_deg);

  // Build the receiver cup upright
  receiver_cup =
      make_cup(RECEIVER_CUP_CENTER_X, RECEIVER_CUP_CENTER_Y,
               RECEIVER_CUP_CENTER_Z, RECEIVER_CUP_WIDTH, RECEIVER_CUP_HEIGHT,
               RECEIVER_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);

  // Add starting fluid into the source cup
  add_fluid_in_source_cup();

  // Build the first boundary particles
  rebuild_boundary_particles_for_export();
}

// Update the cup state for one frame or substep
void update_scene_for_frame(float frame_index, float target_tilt_deg) {

  // Get the current source cup tilt
  float current_tilt_deg =
      get_source_tilt_for_frame(frame_index, target_tilt_deg);

  // Get the previous source cup tilt
  float previous_frame_index =
      frame_index - 1.0f / static_cast<float>(SUBSTEPS_PER_FRAME);

  if (previous_frame_index < 0.0f) {
    previous_frame_index = 0.0f;
  }

  float previous_tilt_deg =
      get_source_tilt_for_frame(previous_frame_index, target_tilt_deg);

  // Compute the source cup angular velocity
  float delta_theta_rad = (current_tilt_deg - previous_tilt_deg) * PI / 180.0f;
  source_cup.angular_velocity = delta_theta_rad / DT;
  receiver_cup.angular_velocity = 0.0f;

  // Update the source cup rotation
  update_cup_rotation(source_cup, current_tilt_deg);

  // Keep the receiver cup upright
  update_cup_rotation(receiver_cup, 0.0f);
}

// Reset penetration debug counters
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

// Print penetration debug counters
void print_penetration_stats(int frame_index) {

  // Print all penetration stats for this frame
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

// Print source cup setup debug stats
void print_source_cup_setup_stats() {

  // Build the source cup interior limits
  float inner_left = -0.5f * source_cup.width + source_cup.wall_thickness;
  float inner_right = 0.5f * source_cup.width - source_cup.wall_thickness;
  float inner_bottom = source_cup.wall_thickness;
  float inner_front =
      source_cup.center_z - 0.5f * source_cup.depth + source_cup.wall_thickness;
  float inner_back =
      source_cup.center_z + 0.5f * source_cup.depth - source_cup.wall_thickness;

  // Start the outside count
  int outside_count = 0;

  // Check every fluid particle against the source cup bounds
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];

    // Convert particle position into the source cup frame
    float local_x;
    float local_y;
    world_to_cup_local(source_cup, particle.x, particle.y, local_x, local_y);

    // Check each interior bound
    bool outside_x = (local_x < inner_left) || (local_x > inner_right);
    bool outside_y = (local_y < inner_bottom) || (local_y > source_cup.height);
    bool outside_z = (particle.z < inner_front) || (particle.z > inner_back);

    // Count particles outside any bound
    if (outside_x || outside_y || outside_z) {
      outside_count++;
    }
  }

  // Print the source cup setup result
  std::cout << "Initial source cup check"
            << " | fluid_count = " << num_fluid_particles
            << " | outside_count = " << outside_count << std::endl;
}

// Push one fluid particle out of one cup wall
void resolve_cup_collision(Particle &particle, const Cup &cup) {

  // Skip boundary particles
  if (particle.is_boundary) {
    return;
  }

  // Pick receiver debug counters by default
  float *max_penetration = &max_receiver_penetration;
  int *below_bottom_count = &below_receiver_bottom_count;
  int *inside_wall_count = &inside_receiver_wall_count;

  // Switch to source debug counters for the source cup
  if (&cup == &source_cup) {
    max_penetration = &max_source_penetration;
    below_bottom_count = &below_source_bottom_count;
    inside_wall_count = &inside_source_wall_count;
  }

  // Move particle position into the cup frame
  float local_x;
  float local_y;
  world_to_cup_local(cup, particle.x, particle.y, local_x, local_y);

  // Move particle velocity into the cup frame
  float local_vx;
  float local_vy;
  world_velocity_to_cup_local(cup, particle.vx, particle.vy, local_vx,
                              local_vy);

  // Build the outer cup bounds
  float outer_left = -0.5f * cup.width;
  float outer_right = 0.5f * cup.width;
  float outer_bottom = 0.0f;
  float outer_top = cup.height;
  float outer_front = cup.center_z - 0.5f * cup.depth;
  float outer_back = cup.center_z + 0.5f * cup.depth;

  // Build the inner collision bounds
  float inner_left = outer_left;
  float inner_right = outer_right;
  float inner_bottom = outer_bottom;
  float inner_front = outer_front;
  float inner_back = outer_back;

  // Count particles below the bottom slab before correction
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom - 1e-4f)) {
    (*below_bottom_count)++;
  }

  // Start the closest wall candidate
  float best_penetration = 1e30f;
  int best_wall = 0;

  // Check bottom wall penetration
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom)) {
    float penetration = inner_bottom - local_y;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 1;
    }
  }

  // Check left wall penetration
  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x < inner_left)) {
    float penetration = inner_left - local_x;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 2;
    }
  }

  // Check right wall penetration
  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x > inner_right)) {
    float penetration = local_x - inner_right;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 3;
    }
  }

  // Check front wall penetration
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z < inner_front)) {
    float penetration = inner_front - particle.z;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 4;
    }
  }

  // Check back wall penetration
  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z > inner_back)) {
    float penetration = particle.z - inner_back;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 5;
    }
  }

  // Count particles that entered wall material
  if (best_wall != 0) {
    (*inside_wall_count)++;
    *max_penetration = std::max(*max_penetration, best_penetration);
  }

  // Resolve bottom wall collision
  if (best_wall == 1) {

    // Push particle above the bottom wall
    local_y = inner_bottom + WALL_EPS;

    // Compute wall velocity at the contact point
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect relative normal velocity
    float relative_vy = local_vy - wall_vy;

    if (relative_vy < 0.0f) {
      local_vy = wall_vy - WALL_RESTITUTION * relative_vy;
    }

    // Dampen tangential motion if enabled
    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;

  // Resolve left wall collision
  } else if (best_wall == 2) {

    // Push particle inside the left wall
    local_x = inner_left + WALL_EPS;

    // Compute wall velocity at the contact point
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect relative normal velocity
    float relative_vx = local_vx - wall_vx;

    if (relative_vx < 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }

    // Dampen tangential motion if enabled
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve right wall collision
  } else if (best_wall == 3) {

    // Push particle inside the right wall
    local_x = inner_right - WALL_EPS;

    // Compute wall velocity at the contact point
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Reflect relative normal velocity
    float relative_vx = local_vx - wall_vx;

    if (relative_vx > 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }

    // Dampen tangential motion if enabled
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve front wall collision
  } else if (best_wall == 4) {

    // Push particle behind the front wall
    particle.z = inner_front + WALL_EPS;

    // Reflect z velocity if moving into the wall
    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    // Compute wall velocity at the contact point
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Dampen tangential motion if enabled
    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;

  // Resolve back wall collision
  } else if (best_wall == 5) {

    // Push particle in front of the back wall
    particle.z = inner_back - WALL_EPS;

    // Reflect z velocity if moving into the wall
    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    // Compute wall velocity at the contact point
    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    // Dampen tangential motion if enabled
    // local_vx = wall_vx + (local_vx - wall_vx) * WALL_TANGENTIAL_DAMPING;
    // local_vy = wall_vy + (local_vy - wall_vy) * WALL_TANGENTIAL_DAMPING;
  }

  // Move corrected particle position back to world space
  cup_local_to_world(cup, local_x, local_y, particle.x, particle.y);

  // Move corrected particle velocity back to world space
  cup_velocity_to_world(cup, local_vx, local_vy, particle.vx, particle.vy);
}