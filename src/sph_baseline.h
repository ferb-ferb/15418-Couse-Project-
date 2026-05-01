#ifndef SPH_BASELINE_H
#define SPH_BASELINE_H

#include <string>

// Store one fluid or boundary particle
struct Particle {
  float x;
  float y;
  float z;
  float vx;
  float vy;
  float vz;
  float fx;
  float fy;
  float fz;
  float rho;
  float p;
  bool is_boundary;
  int kind;
};

// Select which simulation path to run
enum SimulationMode {
  SIM_MODE_CPU_SEQUENTIAL = 0,
  SIM_MODE_GPU_BRUTE_FORCE = 1,
  SIM_MODE_GPU_SPATIAL_HASH = 2,
};

const int MAX_FLUID_PARTICLES = 20000;
const int MAX_BOUNDARY_PARTICLES = 20000;

// Set the main simulation length
const int FRAME_COUNT = 1000;
const int SUBSTEPS_PER_FRAME = 20;
const int EXPORT_EVERY = 1;

// Set the random jitter scales
const float POS_JITTER_CONST = 0.05f;
const float VEL_JITTER_CONST = 0.0f;

// Set the fluid physics values
const float H = 0.025f;
const float H_DENS = H;
const float H_FORCE = H;
const float GRAVITY = -9.81f;
const float DT = 0.0002f;
const float PI = 3.1415926535f;
const float MASS = 0.0009f;
const float REST_DENS = 900.0f;
const float GAS_CONST = 1500.0f;
const float VISCOSITY = 1.0f;
const float EPS = 1e-6f;
const float VELOCITY_DAMPING = 0.999f;

// Set the kernel values
const float POLY6 = 315.0f / (64.0f * PI * H_DENS * H_DENS * H_DENS * H_DENS *
                              H_DENS * H_DENS * H_DENS * H_DENS * H_DENS);
const float SPIKY_GRAD =
    -45.0f / (PI * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE);
const float VISC_LAP =
    45.0f / (PI * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE);

// Set the world box
const float BOX_X_MIN = 0.0f;
const float BOX_X_MAX = 0.7f;
const float BOX_Y_MIN = 0.0f;
const float BOX_Y_MAX = 2.0f;
const float BOX_Z_MIN = 0.0f;
const float BOX_Z_MAX = 0.7f;
const float WALL_EPS = 0.002f;
const float WALL_RESTITUTION = 0.05f;

// Set the spatial hash grid values
const float HASH_CELL_SIZE = H;
const int HASH_GRID_SIZE_X = 28;
const int HASH_GRID_SIZE_Y = 80;
const int HASH_GRID_SIZE_Z = 28;
const int HASH_GRID_CELL_COUNT =
    HASH_GRID_SIZE_X * HASH_GRID_SIZE_Y * HASH_GRID_SIZE_Z;

// Share the main fluid particle array
extern Particle *fluid_particles;
extern int num_fluid_particles;
extern SimulationMode simulation_mode;

// Compute density and pressure for every fluid particle
void compute_density_pressure();

// Compute pressure viscosity and gravity forces for every fluid particle
void compute_forces();

// Build the sorted spatial grid for fluid particles
void build_spatial_grid();

// Integrate velocity and position for every fluid particle
void integrate_fluid_particles();

// Export the current fluid and boundary state to csv
void export_csv(int frame_index);

// Print frame level density pressure and speed stats
void print_stats(int frame_index);

// Print the first density and pressure stats after setup
void print_initial_density_stats();

// Print fluid only stats with a custom label
void print_fluid_only_stats(const std::string &label);

#endif