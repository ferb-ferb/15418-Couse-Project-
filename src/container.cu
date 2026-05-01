#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "container.h"
#include "sph_baseline.h"

Cup source_cup;
Cup receiver_cup;

// Share the dense render boundary
Particle *boundary_particles;
int num_boundary_particles = 0;

// Share the source compute boundary
Particle *source_compute_boundary_particles;
int num_source_compute_boundary_particles = 0;

// Share the receiver compute boundary
Particle *receiver_compute_boundary_particles;
int num_receiver_compute_boundary_particles = 0;

// Track penetration stats
static float max_source_penetration = 0.0f;
static float max_receiver_penetration = 0.0f;
static int below_source_bottom_count = 0;
static int below_receiver_bottom_count = 0;
static int inside_source_wall_count = 0;
static int inside_receiver_wall_count = 0;

// Build one cup from geometry and tilt inputs
Cup make_cup(float center_x, float center_y, float center_z, float width,
             float height, float depth, float wall_thickness, float tilt_deg) {

  // Fill the cup values
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

  // Store the rotation values
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);

  return cup;
}

// Update one cup rotation from a new tilt angle
void update_cup_rotation(Cup &cup, float tilt_deg) {

  // Store the new angle
  cup.tilt_deg = tilt_deg;

  // Store the new rotation values
  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);
}

// Convert cup local coordinates to world coordinates
static void cup_local_to_world(const Cup &cup, float local_x, float local_y,
                               float &world_x, float &world_y) {

  // Rotate and shift the point
  world_x = cup.center_x + local_x * cup.cos_t + local_y * cup.sin_t;
  world_y = cup.center_y - local_x * cup.sin_t + local_y * cup.cos_t;
}

// Convert world coordinates to cup local coordinates
static void world_to_cup_local(const Cup &cup, float world_x, float world_y,
                               float &local_x, float &local_y) {

  // Shift into the cup frame
  float dx = world_x - cup.center_x;
  float dy = world_y - cup.center_y;

  // Rotate into the cup frame
  local_x = dx * cup.cos_t - dy * cup.sin_t;
  local_y = dx * cup.sin_t + dy * cup.cos_t;
}

// Convert world velocity to cup local velocity
static void world_velocity_to_cup_local(const Cup &cup, float world_vx,
                                        float world_vy, float &local_vx,
                                        float &local_vy) {

  // Rotate the velocity into the cup frame
  local_vx = world_vx * cup.cos_t - world_vy * cup.sin_t;
  local_vy = world_vx * cup.sin_t + world_vy * cup.cos_t;
}

// Convert cup local velocity to world velocity
static void cup_velocity_to_world(const Cup &cup, float local_vx,
                                  float local_vy, float &world_vx,
                                  float &world_vy) {

  // Rotate the velocity back to world space
  world_vx = local_vx * cup.cos_t + local_vy * cup.sin_t;
  world_vy = -local_vx * cup.sin_t + local_vy * cup.cos_t;
}

// Build the wall velocity in the cup frame
static void get_local_wall_velocity(const Cup &cup, float local_x,
                                    float local_y, float &wall_vx,
                                    float &wall_vy) {

  // Build rigid wall motion
  wall_vx = -cup.angular_velocity * local_y;
  wall_vy = cup.angular_velocity * local_x;
}

// Add one particle to any boundary array
static void add_boundary_particle_to_array(Particle *particle_array,
                                           int &particle_count, float x, float y,
                                           float z, int kind) {

  // Fill the boundary particle
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

  // Store the particle if space remains
  if (particle_count < MAX_BOUNDARY_PARTICLES) {
    particle_array[particle_count] = particle;
    particle_count++;

  // Warn if the boundary array is full
  } else {
    std::cout << "Warning: Exceeded MAX_BOUNDARY_PARTICLES" << std::endl;
  }
}

// Add one fluid particle to the fluid array
static void add_fluid_particle(float x, float y, float z) {

  // Build the position jitter
  float jitter_scale = POS_JITTER_CONST * INITIAL_PARTICLE_SPACING;
  float jitter_x =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_y =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_z =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

  // Fill the fluid particle
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

  // Warn if the fluid array is full
  } else {
    std::cout << "Warning: Exceeded MAX_FLUID_PARTICLES" << std::endl;
  }
}

// Add one cup wall surface with a selected spacing
static void add_cup_surface_with_spacing(const Cup &cup, float local_x_min,
                                         float local_x_max, float local_y_min,
                                         float local_y_max, float z_min,
                                         float z_max, int kind, float spacing,
                                         Particle *particle_array,
                                         int &particle_count) {

  // Sweep across the local x range
  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * spacing;
       local_x += spacing) {

    // Sweep across the local y range
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * spacing;
         local_y += spacing) {

      // Sweep across the z range
      for (float z = z_min; z <= z_max + 0.5f * spacing; z += spacing) {

        // Convert the local point to world space
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the boundary particle to the selected array
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z, kind);
      }
    }
  }
}

// Add all cup particles at a chosen spacing and layer count
static void add_cup_particles_with_spacing(const Cup &cup, int kind,
                                           float spacing, int num_layers,
                                           Particle *particle_array,
                                           int &particle_count) {

  // Build each wall layer
  for (int layer = 0; layer < num_layers; layer++) {

    // Expand the cup for this layer
    float offset = layer * spacing;
    float current_width = cup.width + (2.0f * offset);
    float current_depth = cup.depth + (2.0f * offset);
    float current_y_min = 0.0f - offset;

    // Build the z limits
    float z_min = cup.center_z - 0.5f * current_depth;
    float z_max = cup.center_z + 0.5f * current_depth;

    // Add the bottom wall
    add_cup_surface_with_spacing(
        cup, -0.5f * current_width, 0.5f * current_width, current_y_min,
        current_y_min, z_min, z_max, kind, spacing, particle_array,
        particle_count);

    // Add the left wall
    add_cup_surface_with_spacing(
        cup, -0.5f * current_width, -0.5f * current_width, current_y_min,
        cup.height, z_min, z_max, kind, spacing, particle_array, particle_count);

    // Add the right wall
    add_cup_surface_with_spacing(
        cup, 0.5f * current_width, 0.5f * current_width, current_y_min,
        cup.height, z_min, z_max, kind, spacing, particle_array, particle_count);

    // Add the front wall
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * spacing;
         local_x += spacing) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * spacing;
           local_y += spacing) {

        // Convert the front wall point to world space
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the front wall particle
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z_min, kind);
      }
    }

    // Add the back wall
    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * spacing;
         local_x += spacing) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * spacing;
           local_y += spacing) {

        // Convert the back wall point to world space
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);

        // Add the back wall particle
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z_max, kind);
      }
    }
  }
}

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

        // Convert this local fluid point to world space
        float world_x;
        float world_y;
        cup_local_to_world(source_cup, local_x, local_y, world_x, world_y);

        // Add the fluid particle
        add_fluid_particle(world_x, world_y, z);
      }
    }
  }
}

// Rebuild the source compute boundary
void rebuild_source_compute_boundary_particles() {

  // Clear the old source compute boundary
  num_source_compute_boundary_particles = 0;

  // Rebuild the source wall with compute spacing
  add_cup_particles_with_spacing(source_cup, 1, COMPUTE_BOUNDARY_SPACING,
                                 COMPUTE_BOUNDARY_LAYERS,
                                 source_compute_boundary_particles,
                                 num_source_compute_boundary_particles);
}

// Rebuild the receiver compute boundary
void rebuild_receiver_compute_boundary_particles() {

  // Clear the old receiver compute boundary
  num_receiver_compute_boundary_particles = 0;

  // Rebuild the receiver wall with compute spacing
  add_cup_particles_with_spacing(receiver_cup, 2, COMPUTE_BOUNDARY_SPACING,
                                 COMPUTE_BOUNDARY_LAYERS,
                                 receiver_compute_boundary_particles,
                                 num_receiver_compute_boundary_particles);
}

// Rebuild the dense render boundary
void rebuild_boundary_particles_for_export() {

  // Clear the old render boundary
  num_boundary_particles = 0;

  // Rebuild the source render wall
  add_cup_particles_with_spacing(source_cup, 1, CUP_RENDER_SPACING,
                                 RENDER_BOUNDARY_LAYERS, boundary_particles,
                                 num_boundary_particles);

  // Rebuild the receiver render wall
  add_cup_particles_with_spacing(receiver_cup, 2, CUP_RENDER_SPACING,
                                 RENDER_BOUNDARY_LAYERS, boundary_particles,
                                 num_boundary_particles);
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

  // Keep the source cup upright during settling
  if (frame_index < SETTLE_FRAMES) {
    return 0.0f;

  // Linearly ramp the source cup tilt
  } else if (frame_index < (SETTLE_FRAMES + TILT_FRAMES)) {
    float alpha = (frame_index - static_cast<float>(SETTLE_FRAMES)) /
                  static_cast<float>(TILT_FRAMES);
    return alpha * target_tilt_deg;

  // Hold the source cup at the target tilt
  } else {
    return target_tilt_deg;
  }
}

// Build the full starting scene
void initialize_scene(float target_tilt_deg) {

  // Clear all particle counts
  num_fluid_particles = 0;
  num_boundary_particles = 0;
  num_source_compute_boundary_particles = 0;
  num_receiver_compute_boundary_particles = 0;

  // Pick the starting source angle
  float initial_source_tilt_deg = 0.0f;

  // Use target angle when static tilt mode is enabled
  if (DEBUG_STATIC_TILT) {
    initial_source_tilt_deg = target_tilt_deg;
  }

  // Force upright when no tilt mode is enabled
  if (DEBUG_NO_TILT) {
    initial_source_tilt_deg = 0.0f;
  }

  // Build the source cup
  source_cup =
      make_cup(SOURCE_CUP_CENTER_X, SOURCE_CUP_CENTER_Y, SOURCE_CUP_CENTER_Z,
               SOURCE_CUP_WIDTH, SOURCE_CUP_HEIGHT, SOURCE_CUP_DEPTH,
               CUP_WALL_THICKNESS, initial_source_tilt_deg);

  // Build the receiver cup
  receiver_cup =
      make_cup(RECEIVER_CUP_CENTER_X, RECEIVER_CUP_CENTER_Y,
               RECEIVER_CUP_CENTER_Z, RECEIVER_CUP_WIDTH, RECEIVER_CUP_HEIGHT,
               RECEIVER_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);

  // Add the starting fluid
  add_fluid_in_source_cup();

  // Build the source compute boundary
  rebuild_source_compute_boundary_particles();

  // Build the receiver compute boundary
  rebuild_receiver_compute_boundary_particles();

  // Build the dense render boundary
  rebuild_boundary_particles_for_export();
}

// Update the cups for one frame or substep
void update_scene_for_frame(float frame_index, float target_tilt_deg) {

  // Get the current source tilt
  float current_tilt_deg =
      get_source_tilt_for_frame(frame_index, target_tilt_deg);

  // Build the previous frame value
  float previous_frame_index =
      frame_index - 1.0f / static_cast<float>(SUBSTEPS_PER_FRAME);

  // Clamp the previous frame value at startup
  if (previous_frame_index < 0.0f) {
    previous_frame_index = 0.0f;
  }

  // Get the previous source tilt
  float previous_tilt_deg =
      get_source_tilt_for_frame(previous_frame_index, target_tilt_deg);

  // Build the angular velocity
  float delta_theta_rad = (current_tilt_deg - previous_tilt_deg) * PI / 180.0f;
  source_cup.angular_velocity = delta_theta_rad / DT;
  receiver_cup.angular_velocity = 0.0f;

  // Update the source cup rotation
  update_cup_rotation(source_cup, current_tilt_deg);

  // Keep the receiver cup upright
  update_cup_rotation(receiver_cup, 0.0f);
}

// Reset the penetration stats
void reset_penetration_stats() {

  // Reset max penetration values
  max_source_penetration = 0.0f;
  max_receiver_penetration = 0.0f;

  // Reset penetration counts
  below_source_bottom_count = 0;
  below_receiver_bottom_count = 0;
  inside_source_wall_count = 0;
  inside_receiver_wall_count = 0;
}

// Print the penetration stats
void print_penetration_stats(int frame_index) {

  // Print the values
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

// Print the source cup setup stats
void print_source_cup_setup_stats() {

  // Build the inner source cup limits
  float inner_left = -0.5f * source_cup.width + source_cup.wall_thickness;
  float inner_right = 0.5f * source_cup.width - source_cup.wall_thickness;
  float inner_bottom = source_cup.wall_thickness;
  float inner_front =
      source_cup.center_z - 0.5f * source_cup.depth + source_cup.wall_thickness;
  float inner_back =
      source_cup.center_z + 0.5f * source_cup.depth - source_cup.wall_thickness;

  // Start outside count
  int outside_count = 0;

  // Check all fluid particles against the source cup bounds
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];

    // Convert particle position into source cup space
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

  // Print the values
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

  // Pick receiver debug values by default
  float *max_penetration = &max_receiver_penetration;
  int *below_bottom_count = &below_receiver_bottom_count;
  int *inside_wall_count = &inside_receiver_wall_count;

  // Switch to source debug values for the source cup
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
  }

  // Move corrected particle position back to world space
  cup_local_to_world(cup, local_x, local_y, particle.x, particle.y);

  // Move corrected particle velocity back to world space
  cup_velocity_to_world(cup, local_vx, local_vy, particle.vx, particle.vy);
}