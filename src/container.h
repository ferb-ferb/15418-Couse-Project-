#ifndef CONTAINER_H
#define CONTAINER_H

#include <vector>

#include "sph_baseline.h"

// Store one cup state and its current rotation values
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
const float INITIAL_PARTICLE_SPACING = 0.01f;
const float CUP_RENDER_SPACING = 0.01f;

// Set the tilt schedule
const int SETTLE_FRAMES = 100;
const int TILT_FRAMES = 500;

// Set the tilt debug modes
const bool DEBUG_NO_TILT = false;
const bool DEBUG_STATIC_TILT = false;

// Share the cup states and render boundary particles
extern Cup source_cup;
extern Cup receiver_cup;
extern std::vector<Particle> boundary_particles;

// Build one cup with its starting rotation values
Cup make_cup(float center_x, float center_y, float center_z, float width,
             float height, float depth, float wall_thickness, float tilt_deg);

// Update one cup rotation for the current tilt angle
void update_cup_rotation(Cup &cup, float tilt_deg);

// Build the full starting scene with cups fluid and boundary particles
void initialize_scene(float target_tilt_deg);

// Update cup motion values for one substep
void update_scene_for_frame(float frame_index, float target_tilt_deg);

// Rebuild the render boundary particles for export
void rebuild_boundary_particles_for_export();

// Push one fluid particle back out of cup walls if it penetrates
void resolve_cup_collision(Particle &particle, const Cup &cup);

// Reset all penetration counters before a new frame
void reset_penetration_stats();

// Print penetration counters for debugging
void print_penetration_stats(int frame_index);

// Check that the source cup starts with fluid inside valid bounds
void print_source_cup_setup_stats();

#endif