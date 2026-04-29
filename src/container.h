#ifndef CONTAINER_H
#define CONTAINER_H

#include <vector>

#include "sph_baseline.h"

struct Cup {
  float center_x;
  float center_y;
  float center_z;
  float width;
  float height;
  float depth;
  float wall_thickness;
  float tilt_deg;
  float cos_t;
  float sin_t;
  float angular_velocity;
};

// Set the cup sizes
const float SOURCE_CUP_WIDTH = 0.14f;
const float SOURCE_CUP_HEIGHT = 0.20f;
const float SOURCE_CUP_DEPTH = 0.18f;
const float RECEIVER_CUP_WIDTH = 0.18f;
const float RECEIVER_CUP_HEIGHT = 0.16f;
const float RECEIVER_CUP_DEPTH = 0.20f;
// const float SOURCE_CUP_WIDTH = 0.105f;
// const float SOURCE_CUP_HEIGHT = 0.15f;
// const float SOURCE_CUP_DEPTH = 0.135f;
// const float RECEIVER_CUP_WIDTH = 0.135f;
// const float RECEIVER_CUP_HEIGHT = 0.12f;
// const float RECEIVER_CUP_DEPTH = 0.15f;
const float CUP_WALL_THICKNESS = 0.025f;

// Set the cup positions
const float SOURCE_CUP_CENTER_X = 0.18f;
const float SOURCE_CUP_CENTER_Y = 0.44f;
const float SOURCE_CUP_CENTER_Z = 0.35f;
const float RECEIVER_CUP_CENTER_X = 0.43f;
const float RECEIVER_CUP_CENTER_Y = 0.08f;
const float RECEIVER_CUP_CENTER_Z = 0.35f;

// Set the fluid fill and render spacing
const float SOURCE_FILL_RATIO = 0.80f;
const float INITIAL_PARTICLE_SPACING = 0.005f;
const float CUP_RENDER_SPACING = 0.005f;

// Set the tilt schedule
const int SETTLE_FRAMES = 100;
const int TILT_FRAMES = 100;

// Set the tilt debug modes
const bool DEBUG_NO_TILT = false;
const bool DEBUG_STATIC_TILT = false;

// Share the cups and boundary particles
extern Cup source_cup;
extern Cup receiver_cup;
extern Particle *boundary_particles;
extern int num_boundary_particles;
// Build one cup
Cup make_cup(float center_x, float center_y, float center_z, float width,
             float height, float depth, float wall_thickness, float tilt_deg);

// Update one cup rotation
void update_cup_rotation(Cup &cup, float tilt_deg);

// Build the whole scene
void initialize_scene(float target_tilt_deg);

// Update the cups for one frame
void update_scene_for_frame(float frame_index, float target_tilt_deg);

// Rebuild the render particles
void rebuild_boundary_particles_for_export();

// Push one fluid particle out of one cup wall
void resolve_cup_collision(Particle &particle, const Cup &cup);

// Reset the penetration stats
void reset_penetration_stats();

// Print the penetration stats
void print_penetration_stats(int frame_index);

// Print the source cup setup stats
void print_source_cup_setup_stats();

#endif
