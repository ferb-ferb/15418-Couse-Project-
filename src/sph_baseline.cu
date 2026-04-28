#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include "container.h"
#include "sph_baseline.h"

Particle *fluid_particles;
int num_fluid_particles;
SimulationMode simulation_mode = SIM_MODE_GPU_NEIGHBOR_LIST;

// Share the fluid neighbor list
static int *fluid_neighbor_count;
static int *fluid_neighbor_list;

// Share the source boundary neighbor list
static int *source_boundary_neighbor_count;
static int *source_boundary_neighbor_list;

// Share the receiver boundary neighbor list
static int *receiver_boundary_neighbor_count;
static int *receiver_boundary_neighbor_list;

// Track neighbor overflow
static int *fluid_neighbor_overflow_particles;
static int *source_boundary_neighbor_overflow_particles;
static int *receiver_boundary_neighbor_overflow_particles;

static long long total_fluid_neighbor_overflow_particles = 0;
static long long total_source_boundary_neighbor_overflow_particles = 0;
static long long total_receiver_boundary_neighbor_overflow_particles = 0;

// Convert the mode to text
static const char *mode_to_string(SimulationMode mode) {
  if (mode == SIM_MODE_GPU_BRUTE_FORCE) {
    return "gpu_bruteforce";
  } else {
    return "gpu_neighbor_list";
  }
}

// Stop when a CUDA call fails
static void cuda_check(cudaError_t code, const char *label) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error in " << label << " "
              << cudaGetErrorString(code) << std::endl;
    std::exit(1);
  }
}

// Read the mode from the command line
static SimulationMode parse_mode(int argc, char **argv) {
  if (argc < 3) {
    return SIM_MODE_GPU_NEIGHBOR_LIST;
  }

  std::string mode_string = argv[2];

  if ((mode_string == "brute") || (mode_string == "gpu_brute") ||
      (mode_string == "all_pairs")) {
    return SIM_MODE_GPU_BRUTE_FORCE;
  }

  if ((mode_string == "neighbor") || (mode_string == "neighbor_list") ||
      (mode_string == "nlist")) {
    return SIM_MODE_GPU_NEIGHBOR_LIST;
  }

  std::cerr << "Unknown mode " << mode_string << std::endl;
  std::cerr << "Use one of brute neighbor" << std::endl;
  std::exit(1);
}

// Build all neighbor lists
__global__ void build_neighbor_lists_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles, int *fluid_neighbor_count,
    int *fluid_neighbor_list, int *source_boundary_neighbor_count,
    int *source_boundary_neighbor_list, int *receiver_boundary_neighbor_count,
    int *receiver_boundary_neighbor_list,
    int *fluid_neighbor_overflow_particles,
    int *source_boundary_neighbor_overflow_particles,
    int *receiver_boundary_neighbor_overflow_particles) {

  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip extra threads
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the particle position
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  // Start the list counts
  int local_fluid_count = 0;
  int local_source_boundary_count = 0;
  int local_receiver_boundary_count = 0;

  // Track overflow for this particle
  bool fluid_overflow = false;
  bool source_boundary_overflow = false;
  bool receiver_boundary_overflow = false;

  // Build the fluid neighbor list
  for (int j = 0; j < num_fluid_particles; j++) {
    if (i == j) {
      continue;
    }

    float dx = fluid_particles[j].x - p_i_x;
    float dy = fluid_particles[j].y - p_i_y;
    float dz = fluid_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_FORCE * H_FORCE) {
      if (local_fluid_count < MAX_FLUID_NEIGHBORS) {
        fluid_neighbor_list[i * MAX_FLUID_NEIGHBORS + local_fluid_count] = j;
        local_fluid_count++;
      } else {
        fluid_overflow = true;
      }
    }
  }

  // Build the source boundary neighbor list
  for (int j = 0; j < num_source_compute_boundary_particles; j++) {
    float dx = source_compute_boundary_particles[j].x - p_i_x;
    float dy = source_compute_boundary_particles[j].y - p_i_y;
    float dz = source_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_FORCE * H_FORCE) {
      if (local_source_boundary_count < MAX_SOURCE_BOUNDARY_NEIGHBORS) {
        source_boundary_neighbor_list
            [i * MAX_SOURCE_BOUNDARY_NEIGHBORS + local_source_boundary_count] =
                j;
        local_source_boundary_count++;
      } else {
        source_boundary_overflow = true;
      }
    }
  }

  // Build the receiver boundary neighbor list
  for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
    float dx = receiver_compute_boundary_particles[j].x - p_i_x;
    float dy = receiver_compute_boundary_particles[j].y - p_i_y;
    float dz = receiver_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_FORCE * H_FORCE) {
      if (local_receiver_boundary_count < MAX_RECEIVER_BOUNDARY_NEIGHBORS) {
        receiver_boundary_neighbor_list
            [i * MAX_RECEIVER_BOUNDARY_NEIGHBORS +
             local_receiver_boundary_count] = j;
        local_receiver_boundary_count++;
      } else {
        receiver_boundary_overflow = true;
      }
    }
  }

  // Store the list counts
  fluid_neighbor_count[i] = local_fluid_count;
  source_boundary_neighbor_count[i] = local_source_boundary_count;
  receiver_boundary_neighbor_count[i] = local_receiver_boundary_count;

  // Count overflow particles
  if (fluid_overflow) {
    atomicAdd(fluid_neighbor_overflow_particles, 1);
  }

  if (source_boundary_overflow) {
    atomicAdd(source_boundary_neighbor_overflow_particles, 1);
  }

  if (receiver_boundary_overflow) {
    atomicAdd(receiver_boundary_neighbor_overflow_particles, 1);
  }
}

// Compute density with all pairs
__global__ void compute_density_pressure_bruteforce_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {

  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip extra threads
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the particle position
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  // Start the density
  float density = 0.0f;

  // Sum fluid density
  for (int j = 0; j < num_fluid_particles; j++) {
    float dx = fluid_particles[j].x - p_i_x;
    float dy = fluid_particles[j].y - p_i_y;
    float dz = fluid_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum source boundary density
  for (int j = 0; j < num_source_compute_boundary_particles; j++) {
    float dx = source_compute_boundary_particles[j].x - p_i_x;
    float dy = source_compute_boundary_particles[j].y - p_i_y;
    float dz = source_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum receiver boundary density
  for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
    float dx = receiver_compute_boundary_particles[j].x - p_i_x;
    float dy = receiver_compute_boundary_particles[j].y - p_i_y;
    float dz = receiver_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Store density and pressure
  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Compute density with neighbor lists
__global__ void compute_density_pressure_neighbor_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles, int *fluid_neighbor_count,
    int *fluid_neighbor_list, int *source_boundary_neighbor_count,
    int *source_boundary_neighbor_list, int *receiver_boundary_neighbor_count,
    int *receiver_boundary_neighbor_list) {

  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip extra threads
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the particle position
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  // Start with self density
  float self_weight = H_DENS * H_DENS;
  float density = MASS * POLY6 * self_weight * self_weight * self_weight;

  // Sum fluid neighbor density
  int local_fluid_neighbor_count = fluid_neighbor_count[i];

  for (int n = 0; n < local_fluid_neighbor_count; n++) {
    int j = fluid_neighbor_list[i * MAX_FLUID_NEIGHBORS + n];

    float dx = fluid_particles[j].x - p_i_x;
    float dy = fluid_particles[j].y - p_i_y;
    float dz = fluid_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum source boundary neighbor density
  int local_source_boundary_neighbor_count = source_boundary_neighbor_count[i];

  for (int n = 0; n < local_source_boundary_neighbor_count; n++) {
    int j =
        source_boundary_neighbor_list[i * MAX_SOURCE_BOUNDARY_NEIGHBORS + n];

    float dx = source_compute_boundary_particles[j].x - p_i_x;
    float dy = source_compute_boundary_particles[j].y - p_i_y;
    float dz = source_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum receiver boundary neighbor density
  int local_receiver_boundary_neighbor_count =
      receiver_boundary_neighbor_count[i];

  for (int n = 0; n < local_receiver_boundary_neighbor_count; n++) {
    int j = receiver_boundary_neighbor_list
        [i * MAX_RECEIVER_BOUNDARY_NEIGHBORS + n];

    float dx = receiver_compute_boundary_particles[j].x - p_i_x;
    float dy = receiver_compute_boundary_particles[j].y - p_i_y;
    float dz = receiver_compute_boundary_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Store density and pressure
  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Compute forces with all pairs
__global__ void compute_forces_bruteforce_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {

  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip extra threads
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the particle state
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  // Start the force sums
  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Sum fluid forces
  for (int j = 0; j < num_fluid_particles; j++) {
    if (i == j) {
      continue;
    }

    float dx = p_i_x - fluid_particles[j].x;
    float dy = p_i_y - fluid_particles[j].y;
    float dz = p_i_z - fluid_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float p_term =
          -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                   fluid_particles[j].p /
                       fmaxf(fluid_particles[j].rho * fluid_particles[j].rho, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / fmaxf(fluid_particles[j].rho, EPS);

      viscosity_fx += VISCOSITY * MASS * (fluid_particles[j].vx - p_i_vx) *
                      inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (fluid_particles[j].vy - p_i_vy) *
                      inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (fluid_particles[j].vz - p_i_vz) *
                      inv_rho_j * visc_coeff;
    }
  }

  // Sum source boundary forces
  for (int j = 0; j < num_source_compute_boundary_particles; j++) {
    float dx = p_i_x - source_compute_boundary_particles[j].x;
    float dy = p_i_y - source_compute_boundary_particles[j].y;
    float dz = p_i_z - source_compute_boundary_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float pb = p_i_p;
      float rhob = source_compute_boundary_particles[j].rho;

      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Sum receiver boundary forces
  for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
    float dx = p_i_x - receiver_compute_boundary_particles[j].x;
    float dy = p_i_y - receiver_compute_boundary_particles[j].y;
    float dz = p_i_z - receiver_compute_boundary_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float pb = p_i_p;
      float rhob = receiver_compute_boundary_particles[j].rho;

      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Store the final force
  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Compute forces with neighbor lists
__global__ void compute_forces_neighbor_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles, int *fluid_neighbor_count,
    int *fluid_neighbor_list, int *source_boundary_neighbor_count,
    int *source_boundary_neighbor_list, int *receiver_boundary_neighbor_count,
    int *receiver_boundary_neighbor_list) {

  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip extra threads
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the particle state
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  // Start the force sums
  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Sum fluid neighbor forces
  int local_fluid_neighbor_count = fluid_neighbor_count[i];

  for (int n = 0; n < local_fluid_neighbor_count; n++) {
    int j = fluid_neighbor_list[i * MAX_FLUID_NEIGHBORS + n];

    float dx = p_i_x - fluid_particles[j].x;
    float dy = p_i_y - fluid_particles[j].y;
    float dz = p_i_z - fluid_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float p_term =
          -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                   fluid_particles[j].p /
                       fmaxf(fluid_particles[j].rho * fluid_particles[j].rho, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / fmaxf(fluid_particles[j].rho, EPS);

      viscosity_fx += VISCOSITY * MASS * (fluid_particles[j].vx - p_i_vx) *
                      inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (fluid_particles[j].vy - p_i_vy) *
                      inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (fluid_particles[j].vz - p_i_vz) *
                      inv_rho_j * visc_coeff;
    }
  }

  // Sum source boundary neighbor forces
  int local_source_boundary_neighbor_count = source_boundary_neighbor_count[i];

  for (int n = 0; n < local_source_boundary_neighbor_count; n++) {
    int j =
        source_boundary_neighbor_list[i * MAX_SOURCE_BOUNDARY_NEIGHBORS + n];

    float dx = p_i_x - source_compute_boundary_particles[j].x;
    float dy = p_i_y - source_compute_boundary_particles[j].y;
    float dz = p_i_z - source_compute_boundary_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float pb = p_i_p;
      float rhob = source_compute_boundary_particles[j].rho;

      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Sum receiver boundary neighbor forces
  int local_receiver_boundary_neighbor_count =
      receiver_boundary_neighbor_count[i];

  for (int n = 0; n < local_receiver_boundary_neighbor_count; n++) {
    int j = receiver_boundary_neighbor_list
        [i * MAX_RECEIVER_BOUNDARY_NEIGHBORS + n];

    float dx = p_i_x - receiver_compute_boundary_particles[j].x;
    float dy = p_i_y - receiver_compute_boundary_particles[j].y;
    float dz = p_i_z - receiver_compute_boundary_particles[j].z;
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      float pb = p_i_p;
      float rhob = receiver_compute_boundary_particles[j].rho;

      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Store the final force
  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Build the neighbor lists
void build_neighbor_lists() {

  // Skip this in brute mode
  if (simulation_mode != SIM_MODE_GPU_NEIGHBOR_LIST) {
    return;
  }

  // Reset the overflow counters
  cuda_check(cudaMemset(fluid_neighbor_overflow_particles, 0, sizeof(int)),
             "cudaMemset(fluid_neighbor_overflow_particles)");
  cuda_check(
      cudaMemset(source_boundary_neighbor_overflow_particles, 0, sizeof(int)),
      "cudaMemset(source_boundary_neighbor_overflow_particles)");
  cuda_check(
      cudaMemset(receiver_boundary_neighbor_overflow_particles, 0, sizeof(int)),
      "cudaMemset(receiver_boundary_neighbor_overflow_particles)");

  // Build the launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  // Build the lists
  build_neighbor_lists_kernel<<<blocks_per_grid, threads_per_block>>>(
      fluid_particles, num_fluid_particles, source_compute_boundary_particles,
      num_source_compute_boundary_particles, receiver_compute_boundary_particles,
      num_receiver_compute_boundary_particles, fluid_neighbor_count,
      fluid_neighbor_list, source_boundary_neighbor_count,
      source_boundary_neighbor_list, receiver_boundary_neighbor_count,
      receiver_boundary_neighbor_list, fluid_neighbor_overflow_particles,
      source_boundary_neighbor_overflow_particles,
      receiver_boundary_neighbor_overflow_particles);
  cuda_check(cudaDeviceSynchronize(), "build_neighbor_lists_kernel");

  // Accumulate the overflow totals
  total_fluid_neighbor_overflow_particles +=
      *fluid_neighbor_overflow_particles;
  total_source_boundary_neighbor_overflow_particles +=
      *source_boundary_neighbor_overflow_particles;
  total_receiver_boundary_neighbor_overflow_particles +=
      *receiver_boundary_neighbor_overflow_particles;
}

// Run the density pass
void compute_density_pressure() {

  // Build the launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  // Run the chosen mode
  if (simulation_mode == SIM_MODE_GPU_BRUTE_FORCE) {
    compute_density_pressure_bruteforce_kernel<<<blocks_per_grid,
                                                 threads_per_block>>>(
        fluid_particles, num_fluid_particles,
        source_compute_boundary_particles, num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(),
               "compute_density_pressure_bruteforce_kernel");
  } else {
    compute_density_pressure_neighbor_kernel<<<blocks_per_grid,
                                               threads_per_block>>>(
        fluid_particles, num_fluid_particles, source_compute_boundary_particles,
        receiver_compute_boundary_particles, fluid_neighbor_count,
        fluid_neighbor_list, source_boundary_neighbor_count,
        source_boundary_neighbor_list, receiver_boundary_neighbor_count,
        receiver_boundary_neighbor_list);
    cuda_check(cudaDeviceSynchronize(),
               "compute_density_pressure_neighbor_kernel");
  }
}

// Run the force pass
void compute_forces() {

  // Build the launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  // Run the chosen mode
  if (simulation_mode == SIM_MODE_GPU_BRUTE_FORCE) {
    compute_forces_bruteforce_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles,
        source_compute_boundary_particles, num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_bruteforce_kernel");
  } else {
    compute_forces_neighbor_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles, source_compute_boundary_particles,
        receiver_compute_boundary_particles, fluid_neighbor_count,
        fluid_neighbor_list, source_boundary_neighbor_count,
        source_boundary_neighbor_list, receiver_boundary_neighbor_count,
        receiver_boundary_neighbor_list);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_neighbor_kernel");
  }
}

// Push one fluid particle back into the world box
static void apply_world_box_collision(Particle &particle) {

  // Resolve the left wall
  if (particle.x < BOX_X_MIN) {
    particle.x = BOX_X_MIN + WALL_EPS;

    if (particle.vx < 0.0f) {
      particle.vx = -particle.vx * WALL_RESTITUTION;
    }
  } else if (particle.x > BOX_X_MAX) {
    particle.x = BOX_X_MAX - WALL_EPS;

    if (particle.vx > 0.0f) {
      particle.vx = -particle.vx * WALL_RESTITUTION;
    }
  }

  // Resolve the floor and ceiling
  if (particle.y < BOX_Y_MIN) {
    particle.y = BOX_Y_MIN + WALL_EPS;

    if (particle.vy < 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }
  } else if (particle.y > BOX_Y_MAX) {
    particle.y = BOX_Y_MAX - WALL_EPS;

    if (particle.vy > 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }
  }

  // Resolve the front and back walls
  if (particle.z < BOX_Z_MIN) {
    particle.z = BOX_Z_MIN + WALL_EPS;

    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }
  } else if (particle.z > BOX_Z_MAX) {
    particle.z = BOX_Z_MAX - WALL_EPS;

    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }
  }
}

// Move the fluid particles
void integrate_fluid_particles() {

  // Loop over the fluid particles
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];

    // Build the acceleration
    float ax = particle.fx / std::fmax(particle.rho, EPS);
    float ay = particle.fy / std::fmax(particle.rho, EPS);
    float az = particle.fz / std::fmax(particle.rho, EPS);

    // Update the velocity
    particle.vx += ax * DT;
    particle.vy += ay * DT;
    particle.vz += az * DT;

    // Dampen the velocity
    particle.vx *= VELOCITY_DAMPING;
    particle.vy *= VELOCITY_DAMPING;
    particle.vz *= VELOCITY_DAMPING;

    // Update the position
    particle.x += particle.vx * DT;
    particle.y += particle.vy * DT;
    particle.z += particle.vz * DT;

    // Resolve the cup collisions
    resolve_cup_collision(particle, source_cup);
    resolve_cup_collision(particle, receiver_cup);

    // Resolve the world box collision
    apply_world_box_collision(particle);
  }
}

// Export the scene to csv
void export_csv(int frame_index) {

  // Build the current render boundary
  rebuild_boundary_particles_for_export();

  // Build the output file name
  std::string frame_string = std::to_string(frame_index);
  frame_string = std::string(4 - frame_string.length(), '0') + frame_string;
  std::string file_name = "output/frame_" + frame_string + ".csv";

  // Open the output file
  std::ofstream file(file_name);

  // Write the header
  file << "x,y,z,rho,p,is_boundary,kind\n";

  // Write the fluid particles
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }

  // Write the boundary particles
  for (int i = 0; i < num_boundary_particles; i++) {
    Particle &particle = boundary_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }
}

// Print the frame stats
void print_stats(int frame_index) {

  // Skip the stats when there is no fluid
  if (num_fluid_particles == 0) {
    return;
  }

  // Start the stat values
  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_speed = 0.0f;
  float max_speed = 0.0f;

  // Accumulate the stats
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    min_rho = std::min(min_rho, particle.rho);
    max_rho = std::max(max_rho, particle.rho);
    min_p = std::min(min_p, particle.p);
    max_p = std::max(max_p, particle.p);
    sum_rho += particle.rho;

    float speed =
        std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy +
                  particle.vz * particle.vz);
    sum_speed += speed;
    max_speed = std::max(max_speed, speed);
  }

  // Build the averages
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  // Print the values
  std::cout << "Frame " << frame_index << " | rho: [" << min_rho << ", "
            << max_rho << "] avg=" << avg_rho << " | p: [" << min_p << ", "
            << max_p << "]"
            << " | speed avg=" << avg_speed << " max=" << max_speed
            << std::endl;
}

// Print the initial stats
void print_initial_density_stats() {

  // Skip the stats when there is no fluid
  if (num_fluid_particles == 0) {
    return;
  }

  // Build the first neighbor lists when needed
  if (simulation_mode == SIM_MODE_GPU_NEIGHBOR_LIST) {
    build_neighbor_lists();
  }

  // Run the first density pass
  compute_density_pressure();

  // Start the stat values
  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_p = 0.0f;

  // Accumulate the stats
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    min_rho = std::min(min_rho, particle.rho);
    max_rho = std::max(max_rho, particle.rho);
    min_p = std::min(min_p, particle.p);
    max_p = std::max(max_p, particle.p);
    sum_rho += particle.rho;
    sum_p += particle.p;
  }

  // Build the averages
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);

  // Print the values
  std::cout << "Initial density stats | rho: [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p: [" << min_p << ", " << max_p
            << "] avg = " << avg_p << std::endl;
}

// Print fluid only stats
void print_fluid_only_stats(const std::string &label) {

  // Skip the stats when there is no fluid
  if (num_fluid_particles == 0) {
    std::cout << label << " | no fluid particles" << std::endl;
    return;
  }

  // Start the stat values
  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float min_speed = std::sqrt(fluid_particles[0].vx * fluid_particles[0].vx +
                              fluid_particles[0].vy * fluid_particles[0].vy +
                              fluid_particles[0].vz * fluid_particles[0].vz);
  float max_speed = min_speed;
  float sum_rho = 0.0f;
  float sum_p = 0.0f;
  float sum_speed = 0.0f;
  int rho_floor_count = 0;
  int zero_pressure_count = 0;

  // Accumulate the stats
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    float speed =
        std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy +
                  particle.vz * particle.vz);

    min_rho = std::min(min_rho, particle.rho);
    max_rho = std::max(max_rho, particle.rho);
    min_p = std::min(min_p, particle.p);
    max_p = std::max(max_p, particle.p);
    min_speed = std::min(min_speed, speed);
    max_speed = std::max(max_speed, speed);

    sum_rho += particle.rho;
    sum_p += particle.p;
    sum_speed += speed;

    if (particle.rho <= (REST_DENS * 0.1f + 1e-4f)) {
      rho_floor_count++;
    }

    if (particle.p <= 1e-4f) {
      zero_pressure_count++;
    }
  }

  // Build the averages
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  // Print the values
  std::cout << label << " | fluid_count = " << num_fluid_particles
            << " | rho = [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p = [" << min_p << ", " << max_p
            << "] avg = " << avg_p << " | speed = [" << min_speed << ", "
            << max_speed << "] avg = " << avg_speed
            << " | rho_floor_count = " << rho_floor_count
            << " | zero_pressure_count = " << zero_pressure_count << std::endl;
}

// Run the full simulation
int main(int argc, char **argv) {
  using clock_type = std::chrono::high_resolution_clock;

  // Seed the random generator
  srand(0);

  // Read the target tilt angle
  float target_tilt_deg = 0.0f;

  if (argc >= 2) {
    target_tilt_deg = std::stof(argv[1]);
  }

  // Clamp the target tilt angle
  target_tilt_deg = std::max(0.0f, std::min(180.0f, target_tilt_deg));

  // Read the simulation mode
  simulation_mode = parse_mode(argc, argv);

  // Allocate the fluid array
  cuda_check(cudaMallocManaged(&fluid_particles,
                               MAX_FLUID_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(fluid_particles)");

  // Allocate the render boundary array
  cuda_check(cudaMallocManaged(&boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(boundary_particles)");

  // Allocate the source compute boundary array
  cuda_check(cudaMallocManaged(&source_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(source_compute_boundary_particles)");

  // Allocate the receiver compute boundary array
  cuda_check(cudaMallocManaged(&receiver_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(receiver_compute_boundary_particles)");

  // Allocate the fluid neighbor arrays
  cuda_check(cudaMallocManaged(&fluid_neighbor_count,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(fluid_neighbor_count)");
  cuda_check(
      cudaMallocManaged(&fluid_neighbor_list,
                        static_cast<size_t>(MAX_FLUID_PARTICLES) *
                            static_cast<size_t>(MAX_FLUID_NEIGHBORS) *
                            sizeof(int)),
      "cudaMallocManaged(fluid_neighbor_list)");

  // Allocate the source boundary neighbor arrays
  cuda_check(cudaMallocManaged(&source_boundary_neighbor_count,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(source_boundary_neighbor_count)");
  cuda_check(
      cudaMallocManaged(&source_boundary_neighbor_list,
                        static_cast<size_t>(MAX_FLUID_PARTICLES) *
                            static_cast<size_t>(MAX_SOURCE_BOUNDARY_NEIGHBORS) *
                            sizeof(int)),
      "cudaMallocManaged(source_boundary_neighbor_list)");

  // Allocate the receiver boundary neighbor arrays
  cuda_check(cudaMallocManaged(&receiver_boundary_neighbor_count,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(receiver_boundary_neighbor_count)");
  cuda_check(
      cudaMallocManaged(
          &receiver_boundary_neighbor_list,
          static_cast<size_t>(MAX_FLUID_PARTICLES) *
              static_cast<size_t>(MAX_RECEIVER_BOUNDARY_NEIGHBORS) * sizeof(int)),
      "cudaMallocManaged(receiver_boundary_neighbor_list)");

  // Allocate the overflow counters
  cuda_check(cudaMallocManaged(&fluid_neighbor_overflow_particles, sizeof(int)),
             "cudaMallocManaged(fluid_neighbor_overflow_particles)");
  cuda_check(
      cudaMallocManaged(&source_boundary_neighbor_overflow_particles,
                        sizeof(int)),
      "cudaMallocManaged(source_boundary_neighbor_overflow_particles)");
  cuda_check(
      cudaMallocManaged(&receiver_boundary_neighbor_overflow_particles,
                        sizeof(int)),
      "cudaMallocManaged(receiver_boundary_neighbor_overflow_particles)");

  // Build the initial scene
  std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg
            << std::endl;
  std::cout << "Simulation mode = " << mode_to_string(simulation_mode)
            << std::endl;
  initialize_scene(target_tilt_deg);

  // Print the initial stats
  print_initial_density_stats();
  print_fluid_only_stats("Frame 0 fluid stats");
  print_source_cup_setup_stats();

  // Print the particle counts
  std::cout << "Fluid particles = " << num_fluid_particles << std::endl;
  std::cout << "Source compute boundary particles = "
            << num_source_compute_boundary_particles << std::endl;
  std::cout << "Receiver compute boundary particles = "
            << num_receiver_compute_boundary_particles << std::endl;
  std::cout << "Render boundary particles = " << num_boundary_particles
            << std::endl;

  // Export the first frame
  export_csv(0);

  // Start the timing totals
  double total_compute_ms = 0.0;
  double min_frame_ms = std::numeric_limits<double>::max();
  double max_frame_ms = 0.0;
  double total_boundary_rebuild_ms = 0.0;
  double total_neighbor_build_ms = 0.0;
  double total_density_ms = 0.0;
  double total_force_ms = 0.0;
  double total_integration_ms = 0.0;

  // Run the frame loop
  for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {

    // Reset the penetration stats
    reset_penetration_stats();

    // Start the frame timer
    auto frame_start = clock_type::now();

    // Run the substeps
    for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {

      // Update the scene for this substep
      float substep_frame_index = static_cast<float>(frame_index - 1) +
                                  static_cast<float>(step_index + 1) /
                                      static_cast<float>(SUBSTEPS_PER_FRAME);
      update_scene_for_frame(substep_frame_index, target_tilt_deg);

      // Rebuild only the source compute boundary
      auto boundary_start = clock_type::now();
      rebuild_source_compute_boundary_particles();
      auto boundary_end = clock_type::now();
      total_boundary_rebuild_ms +=
          std::chrono::duration<double, std::milli>(boundary_end - boundary_start)
              .count();

      // Build neighbor lists when needed
      if (simulation_mode == SIM_MODE_GPU_NEIGHBOR_LIST) {
        auto neighbor_start = clock_type::now();
        build_neighbor_lists();
        auto neighbor_end = clock_type::now();
        total_neighbor_build_ms +=
            std::chrono::duration<double, std::milli>(neighbor_end -
                                                      neighbor_start)
                .count();
      }

      // Run the density pass
      auto density_start = clock_type::now();
      compute_density_pressure();
      auto density_end = clock_type::now();
      total_density_ms +=
          std::chrono::duration<double, std::milli>(density_end - density_start)
              .count();

      // Run the force pass
      auto force_start = clock_type::now();
      compute_forces();
      auto force_end = clock_type::now();
      total_force_ms +=
          std::chrono::duration<double, std::milli>(force_end - force_start)
              .count();

      // Run the integration pass
      auto integration_start = clock_type::now();
      integrate_fluid_particles();
      auto integration_end = clock_type::now();
      total_integration_ms += std::chrono::duration<double, std::milli>(
                                  integration_end - integration_start)
                                  .count();
    }

    // Stop the frame timer
    auto frame_end = clock_type::now();
    double frame_ms =
        std::chrono::duration<double, std::milli>(frame_end - frame_start)
            .count();

    // Update the frame timing stats
    total_compute_ms += frame_ms;
    min_frame_ms = std::min(min_frame_ms, frame_ms);
    max_frame_ms = std::max(max_frame_ms, frame_ms);

    // Print the frame timing
    std::cout << std::fixed << std::setprecision(3)
              << "FrameTiming " << frame_index << " " << frame_ms << std::endl;

    // Export this frame when needed
    if (frame_index % EXPORT_EVERY == 0) {
      export_csv(frame_index / EXPORT_EVERY);
    }

    // Print the stats sometimes
    if (frame_index % 20 == 0) {
      print_stats(frame_index);
      print_fluid_only_stats("Frame " + std::to_string(frame_index) +
                             " fluid stats");
      print_penetration_stats(frame_index);
    }
  }

  // Build the average frame time
  double avg_frame_ms = total_compute_ms / static_cast<double>(FRAME_COUNT);

  // Print the timing summary
  std::cout << std::fixed << std::setprecision(3)
            << "TimingSummary mode=" << mode_to_string(simulation_mode)
            << " total_ms=" << total_compute_ms
            << " avg_ms=" << avg_frame_ms
            << " min_ms=" << min_frame_ms
            << " max_ms=" << max_frame_ms
            << " boundary_rebuild_ms=" << total_boundary_rebuild_ms
            << " neighbor_build_ms=" << total_neighbor_build_ms
            << " density_ms=" << total_density_ms
            << " force_ms=" << total_force_ms
            << " integration_ms=" << total_integration_ms << std::endl;

  // Print the overflow summary
  std::cout << "NeighborOverflowSummary"
            << " fluid_particles=" << total_fluid_neighbor_overflow_particles
            << " source_boundary_particles="
            << total_source_boundary_neighbor_overflow_particles
            << " receiver_boundary_particles="
            << total_receiver_boundary_neighbor_overflow_particles << std::endl;

  std::cout << "Done" << std::endl;

  // Free the overflow counters
  cuda_check(cudaFree(receiver_boundary_neighbor_overflow_particles),
             "cudaFree(receiver_boundary_neighbor_overflow_particles)");
  cuda_check(cudaFree(source_boundary_neighbor_overflow_particles),
             "cudaFree(source_boundary_neighbor_overflow_particles)");
  cuda_check(cudaFree(fluid_neighbor_overflow_particles),
             "cudaFree(fluid_neighbor_overflow_particles)");

  // Free the receiver boundary neighbor arrays
  cuda_check(cudaFree(receiver_boundary_neighbor_list),
             "cudaFree(receiver_boundary_neighbor_list)");
  cuda_check(cudaFree(receiver_boundary_neighbor_count),
             "cudaFree(receiver_boundary_neighbor_count)");

  // Free the source boundary neighbor arrays
  cuda_check(cudaFree(source_boundary_neighbor_list),
             "cudaFree(source_boundary_neighbor_list)");
  cuda_check(cudaFree(source_boundary_neighbor_count),
             "cudaFree(source_boundary_neighbor_count)");

  // Free the fluid neighbor arrays
  cuda_check(cudaFree(fluid_neighbor_list), "cudaFree(fluid_neighbor_list)");
  cuda_check(cudaFree(fluid_neighbor_count), "cudaFree(fluid_neighbor_count)");

  // Free the boundary arrays
  cuda_check(cudaFree(receiver_compute_boundary_particles),
             "cudaFree(receiver_compute_boundary_particles)");
  cuda_check(cudaFree(source_compute_boundary_particles),
             "cudaFree(source_compute_boundary_particles)");
  cuda_check(cudaFree(boundary_particles), "cudaFree(boundary_particles)");

  // Free the fluid array
  cuda_check(cudaFree(fluid_particles), "cudaFree(fluid_particles)");

  return 0;
}