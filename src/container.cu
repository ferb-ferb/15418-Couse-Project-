#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "container.h"
#include "sph_baseline.h"

Cup source_cup;
Cup receiver_cup;

// Dense render boundary
Particle *boundary_particles;
int num_boundary_particles = 0;

// Compute boundary for the moving source cup
Particle *source_compute_boundary_particles;
int num_source_compute_boundary_particles = 0;

// Compute boundary for the static receiver cup
Particle *receiver_compute_boundary_particles;
int num_receiver_compute_boundary_particles = 0;

static float max_source_penetration = 0.0f;
static float max_receiver_penetration = 0.0f;
static int below_source_bottom_count = 0;
static int below_receiver_bottom_count = 0;
static int inside_source_wall_count = 0;
static int inside_receiver_wall_count = 0;

// Build one cup
Cup make_cup(float center_x, float center_y, float center_z, float width,
             float height, float depth, float wall_thickness, float tilt_deg) {

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

  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);

  return cup;
}

// Update one cup rotation
void update_cup_rotation(Cup &cup, float tilt_deg) {
  cup.tilt_deg = tilt_deg;

  float theta = tilt_deg * PI / 180.0f;
  cup.cos_t = std::cos(theta);
  cup.sin_t = std::sin(theta);
}

// Convert cup local coordinates to world coordinates
static void cup_local_to_world(const Cup &cup, float local_x, float local_y,
                               float &world_x, float &world_y) {
  world_x = cup.center_x + local_x * cup.cos_t + local_y * cup.sin_t;
  world_y = cup.center_y - local_x * cup.sin_t + local_y * cup.cos_t;
}

// Convert world coordinates to cup local coordinates
static void world_to_cup_local(const Cup &cup, float world_x, float world_y,
                               float &local_x, float &local_y) {
  float dx = world_x - cup.center_x;
  float dy = world_y - cup.center_y;

  local_x = dx * cup.cos_t - dy * cup.sin_t;
  local_y = dx * cup.sin_t + dy * cup.cos_t;
}

// Convert world velocity to cup local velocity
static void world_velocity_to_cup_local(const Cup &cup, float world_vx,
                                        float world_vy, float &local_vx,
                                        float &local_vy) {
  local_vx = world_vx * cup.cos_t - world_vy * cup.sin_t;
  local_vy = world_vx * cup.sin_t + world_vy * cup.cos_t;
}

// Convert cup local velocity to world velocity
static void cup_velocity_to_world(const Cup &cup, float local_vx,
                                  float local_vy, float &world_vx,
                                  float &world_vy) {
  world_vx = local_vx * cup.cos_t + local_vy * cup.sin_t;
  world_vy = -local_vx * cup.sin_t + local_vy * cup.cos_t;
}

// Build the wall velocity in the cup frame
static void get_local_wall_velocity(const Cup &cup, float local_x,
                                    float local_y, float &wall_vx,
                                    float &wall_vy) {
  wall_vx = -cup.angular_velocity * local_y;
  wall_vy = cup.angular_velocity * local_x;
}

// Add one particle to a boundary array
static void add_boundary_particle_to_array(Particle *particle_array,
                                           int &particle_count, float x, float y,
                                           float z, int kind) {
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

  if (particle_count < MAX_BOUNDARY_PARTICLES) {
    particle_array[particle_count] = particle;
    particle_count++;
  } else {
    std::cout << "Warning: Exceeded MAX_BOUNDARY_PARTICLES!" << std::endl;
  }
}

// Add one fluid particle
static void add_fluid_particle(float x, float y, float z) {
  float jitter_scale = POS_JITTER_CONST * INITIAL_PARTICLE_SPACING;
  float jitter_x =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_y =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
  float jitter_z =
      (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

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

  if (num_fluid_particles < MAX_FLUID_PARTICLES) {
    fluid_particles[num_fluid_particles] = particle;
    num_fluid_particles++;
  } else {
    std::cout << "Warning: Exceeded MAX_FLUID_PARTICLES!" << std::endl;
  }
}

// Add one cup wall surface at a chosen spacing
static void add_cup_surface_with_spacing(const Cup &cup, float local_x_min,
                                         float local_x_max, float local_y_min,
                                         float local_y_max, float z_min,
                                         float z_max, int kind, float spacing,
                                         Particle *particle_array,
                                         int &particle_count) {
  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * spacing; local_x += spacing) {
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * spacing; local_y += spacing) {
      for (float z = z_min; z <= z_max + 0.5f * spacing; z += spacing) {
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z, kind);
      }
    }
  }
}

// Add cup particles at a chosen spacing
static void add_cup_particles_with_spacing(const Cup &cup, int kind,
                                           float spacing,
                                           Particle *particle_array,
                                           int &particle_count) {
  int num_layers = 4;

  for (int layer = 0; layer < num_layers; layer++) {
    float offset = layer * spacing;
    float current_width = cup.width + (2.0f * offset);
    float current_depth = cup.depth + (2.0f * offset);
    float current_y_min = 0.0f - offset;

    float z_min = cup.center_z - 0.5f * current_depth;
    float z_max = cup.center_z + 0.5f * current_depth;

    add_cup_surface_with_spacing(
        cup, -0.5f * current_width, 0.5f * current_width, current_y_min,
        current_y_min, z_min, z_max, kind, spacing, particle_array,
        particle_count);

    add_cup_surface_with_spacing(
        cup, -0.5f * current_width, -0.5f * current_width, current_y_min,
        cup.height, z_min, z_max, kind, spacing, particle_array, particle_count);

    add_cup_surface_with_spacing(
        cup, 0.5f * current_width, 0.5f * current_width, current_y_min,
        cup.height, z_min, z_max, kind, spacing, particle_array, particle_count);

    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * spacing;
         local_x += spacing) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * spacing; local_y += spacing) {
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z_min, kind);
      }
    }

    for (float local_x = -0.5f * current_width;
         local_x <= 0.5f * current_width + 0.5f * spacing;
         local_x += spacing) {
      for (float local_y = current_y_min;
           local_y <= cup.height + 0.5f * spacing; local_y += spacing) {
        float world_x;
        float world_y;
        cup_local_to_world(cup, local_x, local_y, world_x, world_y);
        add_boundary_particle_to_array(particle_array, particle_count, world_x,
                                       world_y, z_max, kind);
      }
    }
  }
}

// Add fluid inside the source cup
static void add_fluid_in_source_cup() {
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

  for (float local_x = local_x_min;
       local_x <= local_x_max + 0.5f * INITIAL_PARTICLE_SPACING;
       local_x += INITIAL_PARTICLE_SPACING) {
    for (float local_y = local_y_min;
         local_y <= local_y_max + 0.5f * INITIAL_PARTICLE_SPACING;
         local_y += INITIAL_PARTICLE_SPACING) {
      for (float z = z_min; z <= z_max + 0.5f * INITIAL_PARTICLE_SPACING;
           z += INITIAL_PARTICLE_SPACING) {
        float world_x;
        float world_y;
        cup_local_to_world(source_cup, local_x, local_y, world_x, world_y);
        add_fluid_particle(world_x, world_y, z);
      }
    }
  }
}

// Rebuild only the moving source compute boundary
void rebuild_source_compute_boundary_particles() {
  num_source_compute_boundary_particles = 0;

  add_cup_particles_with_spacing(source_cup, 1, COMPUTE_BOUNDARY_SPACING,
                                 source_compute_boundary_particles,
                                 num_source_compute_boundary_particles);
}

// Rebuild the static receiver compute boundary
void rebuild_receiver_compute_boundary_particles() {
  num_receiver_compute_boundary_particles = 0;

  add_cup_particles_with_spacing(receiver_cup, 2, COMPUTE_BOUNDARY_SPACING,
                                 receiver_compute_boundary_particles,
                                 num_receiver_compute_boundary_particles);
}

// Rebuild the dense render boundary
void rebuild_boundary_particles_for_export() {
  num_boundary_particles = 0;

  add_cup_particles_with_spacing(source_cup, 1, CUP_RENDER_SPACING,
                                 boundary_particles, num_boundary_particles);

  add_cup_particles_with_spacing(receiver_cup, 2, CUP_RENDER_SPACING,
                                 boundary_particles, num_boundary_particles);
}

// Get the source tilt for one frame
static float get_source_tilt_for_frame(float frame_index,
                                       float target_tilt_deg) {
  if (DEBUG_NO_TILT) {
    return 0.0f;
  } else if (DEBUG_STATIC_TILT) {
    return target_tilt_deg;
  }

  if (frame_index < SETTLE_FRAMES) {
    return 0.0f;
  } else if (frame_index < (SETTLE_FRAMES + TILT_FRAMES)) {
    float alpha = (frame_index - static_cast<float>(SETTLE_FRAMES)) /
                  static_cast<float>(TILT_FRAMES);
    return alpha * target_tilt_deg;
  } else {
    return target_tilt_deg;
  }
}

// Build the whole scene
void initialize_scene(float target_tilt_deg) {
  num_fluid_particles = 0;
  num_boundary_particles = 0;
  num_source_compute_boundary_particles = 0;
  num_receiver_compute_boundary_particles = 0;

  float initial_source_tilt_deg = 0.0f;

  if (DEBUG_STATIC_TILT) {
    initial_source_tilt_deg = target_tilt_deg;
  }

  if (DEBUG_NO_TILT) {
    initial_source_tilt_deg = 0.0f;
  }

  source_cup =
      make_cup(SOURCE_CUP_CENTER_X, SOURCE_CUP_CENTER_Y, SOURCE_CUP_CENTER_Z,
               SOURCE_CUP_WIDTH, SOURCE_CUP_HEIGHT, SOURCE_CUP_DEPTH,
               CUP_WALL_THICKNESS, initial_source_tilt_deg);

  receiver_cup =
      make_cup(RECEIVER_CUP_CENTER_X, RECEIVER_CUP_CENTER_Y,
               RECEIVER_CUP_CENTER_Z, RECEIVER_CUP_WIDTH, RECEIVER_CUP_HEIGHT,
               RECEIVER_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);

  add_fluid_in_source_cup();

  rebuild_source_compute_boundary_particles();
  rebuild_receiver_compute_boundary_particles();
  rebuild_boundary_particles_for_export();
}

// Update the cups for one frame
void update_scene_for_frame(float frame_index, float target_tilt_deg) {
  float current_tilt_deg =
      get_source_tilt_for_frame(frame_index, target_tilt_deg);

  float previous_frame_index =
      frame_index - 1.0f / static_cast<float>(SUBSTEPS_PER_FRAME);

  if (previous_frame_index < 0.0f) {
    previous_frame_index = 0.0f;
  }

  float previous_tilt_deg =
      get_source_tilt_for_frame(previous_frame_index, target_tilt_deg);

  float delta_theta_rad = (current_tilt_deg - previous_tilt_deg) * PI / 180.0f;
  source_cup.angular_velocity = delta_theta_rad / DT;
  receiver_cup.angular_velocity = 0.0f;

  update_cup_rotation(source_cup, current_tilt_deg);
  update_cup_rotation(receiver_cup, 0.0f);
}

// Reset the penetration stats
void reset_penetration_stats() {
  max_source_penetration = 0.0f;
  max_receiver_penetration = 0.0f;
  below_source_bottom_count = 0;
  below_receiver_bottom_count = 0;
  inside_source_wall_count = 0;
  inside_receiver_wall_count = 0;
}

// Print the penetration stats
void print_penetration_stats(int frame_index) {
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
  float inner_left = -0.5f * source_cup.width + source_cup.wall_thickness;
  float inner_right = 0.5f * source_cup.width - source_cup.wall_thickness;
  float inner_bottom = source_cup.wall_thickness;
  float inner_front =
      source_cup.center_z - 0.5f * source_cup.depth + source_cup.wall_thickness;
  float inner_back =
      source_cup.center_z + 0.5f * source_cup.depth - source_cup.wall_thickness;

  int outside_count = 0;

  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
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

  std::cout << "Initial source cup check"
            << " | fluid_count = " << num_fluid_particles
            << " | outside_count = " << outside_count << std::endl;
}

// Push one fluid particle out of one cup wall
void resolve_cup_collision(Particle &particle, const Cup &cup) {
  if (particle.is_boundary) {
    return;
  }

  float *max_penetration = &max_receiver_penetration;
  int *below_bottom_count = &below_receiver_bottom_count;
  int *inside_wall_count = &inside_receiver_wall_count;

  if (&cup == &source_cup) {
    max_penetration = &max_source_penetration;
    below_bottom_count = &below_source_bottom_count;
    inside_wall_count = &inside_source_wall_count;
  }

  float local_x;
  float local_y;
  world_to_cup_local(cup, particle.x, particle.y, local_x, local_y);

  float local_vx;
  float local_vy;
  world_velocity_to_cup_local(cup, particle.vx, particle.vy, local_vx,
                              local_vy);

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

  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom - 1e-4f)) {
    (*below_bottom_count)++;
  }

  float best_penetration = 1e30f;
  int best_wall = 0;

  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_y < inner_bottom)) {
    float penetration = inner_bottom - local_y;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 1;
    }
  }

  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x < inner_left)) {
    float penetration = inner_left - local_x;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 2;
    }
  }

  if ((local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z >= outer_front) && (particle.z <= outer_back) &&
      (local_x > inner_right)) {
    float penetration = local_x - inner_right;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 3;
    }
  }

  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z < inner_front)) {
    float penetration = inner_front - particle.z;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 4;
    }
  }

  if ((local_x >= outer_left) && (local_x <= outer_right) &&
      (local_y >= outer_bottom) && (local_y <= outer_top) &&
      (particle.z > inner_back)) {
    float penetration = particle.z - inner_back;

    if (penetration < best_penetration) {
      best_penetration = penetration;
      best_wall = 5;
    }
  }

  if (best_wall != 0) {
    (*inside_wall_count)++;
    *max_penetration = std::max(*max_penetration, best_penetration);
  }

  if (best_wall == 1) {
    local_y = inner_bottom + WALL_EPS;

    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    float relative_vy = local_vy - wall_vy;

    if (relative_vy < 0.0f) {
      local_vy = wall_vy - WALL_RESTITUTION * relative_vy;
    }
  } else if (best_wall == 2) {
    local_x = inner_left + WALL_EPS;

    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    float relative_vx = local_vx - wall_vx;

    if (relative_vx < 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }
  } else if (best_wall == 3) {
    local_x = inner_right - WALL_EPS;

    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);

    float relative_vx = local_vx - wall_vx;

    if (relative_vx > 0.0f) {
      local_vx = wall_vx - WALL_RESTITUTION * relative_vx;
    }
  } else if (best_wall == 4) {
    particle.z = inner_front + WALL_EPS;

    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);
  } else if (best_wall == 5) {
    particle.z = inner_back - WALL_EPS;

    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    float wall_vx;
    float wall_vy;
    get_local_wall_velocity(cup, local_x, local_y, wall_vx, wall_vy);
  }

  cup_local_to_world(cup, local_x, local_y, particle.x, particle.y);
  cup_velocity_to_world(cup, local_vx, local_vy, particle.vx, particle.vy);
}