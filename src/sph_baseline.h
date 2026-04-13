#ifndef SPH_BASELINE_H
#define SPH_BASELINE_H

#include <string>
#include <vector>

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

// Set the main simulation length
const int FRAME_COUNT = 1000;
const int SUBSTEPS_PER_FRAME = 20;
const int EXPORT_EVERY = 1;

// Set the random jitter scales
const float POS_JITTER_CONST = 0.0f;
const float VEL_JITTER_CONST = 0.0f;

//const float POS_JITTER_CONST = 0.3f;
//const float VEL_JITTER_CONST = 0.05f;

// Set the fluid physics values
const float H_DENS = 0.13f;
const float H_FORCE = 0.05f;
const float GRAVITY = -9.81f;
const float DT = 0.0003f;
const float PI = 3.1415926535f;
const float MASS = 0.0025f;
const float REST_DENS = 900.0f;
const float GAS_CONST = 60.0f;
const float VISCOSITY = 30.0f;
const float EPS = 1e-6f;
const float VELOCITY_DAMPING = 0.9995f;

// Set the kernel values
const float POLY6 = 315.0f / (64.0f * PI * H_DENS * H_DENS * H_DENS * H_DENS * H_DENS * H_DENS * H_DENS * H_DENS * H_DENS);
const float SPIKY_GRAD = -45.0f / (PI * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE);
const float VISC_LAP = 45.0f / (PI * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE * H_FORCE);

// Set the world box
const float BOX_X_MIN = 0.0f;
const float BOX_X_MAX = 0.7f;
const float BOX_Y_MIN = 0.0f;
const float BOX_Y_MAX = 2.0f;
const float BOX_Z_MIN = 0.0f;
const float BOX_Z_MAX = 0.7f;
const float WALL_EPS = 0.002f;
const float WALL_RESTITUTION = 0.0f;
const float WALL_TANGENTIAL_DAMPING = 0.98f;

// Share the fluid particles
extern std::vector<Particle> fluid_particles;

// Run the density pass
void compute_density_pressure();

// Run the force pass
void compute_forces();

// Move the fluid particles
void integrate_fluid_particles();

// Export the scene to csv
void export_csv(int frame_index);

// Print the frame stats
void print_stats(int frame_index);

// Print the initial stats
void print_initial_density_stats();

// Print fluid only stats
void print_fluid_only_stats(const std::string &label);

#endif