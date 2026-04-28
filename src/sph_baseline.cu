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
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "container.h"
#include "sph_baseline.h"

Particle *fluid_particles;
int num_fluid_particles;
SimulationMode simulation_mode = SIM_MODE_GPU_SPATIAL_HASH;

static int *particle_cell_id;
static int *particle_sorted_index;
static int *cell_start;
static int *cell_end;

// Convert the mode to text
static const char *mode_to_string(SimulationMode mode) {
  if (mode == SIM_MODE_CPU_SEQUENTIAL) {
    return "cpu_sequential";
  } else if (mode == SIM_MODE_GPU_BRUTE_FORCE) {
    return "gpu_bruteforce";
  } else {
    return "gpu_spatial_hash";
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

// Clamp one integer
__device__ __forceinline__ static int clamp_int(int value, int low, int high) {
  if (value < low) {
    return low;
  }

  if (value > high) {
    return high;
  }

  return value;
}

// Convert grid coordinates to one flat index
__device__ __forceinline__ static int grid_coords_to_index(int gx, int gy,
                                                           int gz) {
  return gx + HASH_GRID_SIZE_X * (gy + HASH_GRID_SIZE_Y * gz);
}

// Convert one particle position to grid coordinates
__device__ __forceinline__ static void position_to_grid_coords(float x, float y,
                                                               float z, int &gx,
                                                               int &gy,
                                                               int &gz) {
  gx = static_cast<int>(floorf((x - BOX_X_MIN) / HASH_CELL_SIZE));
  gy = static_cast<int>(floorf((y - BOX_Y_MIN) / HASH_CELL_SIZE));
  gz = static_cast<int>(floorf((z - BOX_Z_MIN) / HASH_CELL_SIZE));

  gx = clamp_int(gx, 0, HASH_GRID_SIZE_X - 1);
  gy = clamp_int(gy, 0, HASH_GRID_SIZE_Y - 1);
  gz = clamp_int(gz, 0, HASH_GRID_SIZE_Z - 1);
}

// Build the particle cell ids
__global__ void compute_particle_cell_id_kernel(Particle *fluid_particles,
                                                int num_fluid_particles,
                                                int *particle_cell_id,
                                                int *particle_sorted_index) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  int gx;
  int gy;
  int gz;
  position_to_grid_coords(fluid_particles[i].x, fluid_particles[i].y,
                          fluid_particles[i].z, gx, gy, gz);

  particle_cell_id[i] = grid_coords_to_index(gx, gy, gz);
  particle_sorted_index[i] = i;
}

// Build the cell ranges
__global__ void build_cell_ranges_kernel(int *particle_cell_id,
                                         int num_fluid_particles,
                                         int *cell_start, int *cell_end) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  int current_cell = particle_cell_id[i];

  if ((i == 0) || (particle_cell_id[i - 1] != current_cell)) {
    cell_start[current_cell] = i;
  }

  if ((i == num_fluid_particles - 1) ||
      (particle_cell_id[i + 1] != current_cell)) {
    cell_end[current_cell] = i + 1;
  }
}

// Compute density with all pairs
__global__ void compute_density_pressure_bruteforce_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  float density = 0.0f;

  // Fluid fluid density
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

  // Source boundary density
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

  // Receiver boundary density
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

  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Compute density with sorted grid
__global__ void compute_density_pressure_hash_kernel(
    Particle *fluid_particles, int num_fluid_particles, int *particle_sorted_index,
    int *cell_start, int *cell_end,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  int gx;
  int gy;
  int gz;
  position_to_grid_coords(p_i_x, p_i_y, p_i_z, gx, gy, gz);

  float density = 0.0f;

  // Fluid fluid density
  for (int dz_off = -1; dz_off <= 1; dz_off++) {
    for (int dy_off = -1; dy_off <= 1; dy_off++) {
      for (int dx_off = -1; dx_off <= 1; dx_off++) {
        int ngx = gx + dx_off;
        int ngy = gy + dy_off;
        int ngz = gz + dz_off;

        if ((ngx < 0) || (ngx >= HASH_GRID_SIZE_X) || (ngy < 0) ||
            (ngy >= HASH_GRID_SIZE_Y) || (ngz < 0) ||
            (ngz >= HASH_GRID_SIZE_Z)) {
          continue;
        }

        int neighbor_cell = grid_coords_to_index(ngx, ngy, ngz);
        int start = cell_start[neighbor_cell];

        if (start == -1) {
          continue;
        }

        int end = cell_end[neighbor_cell];

        for (int s = start; s < end; s++) {
          int j = particle_sorted_index[s];

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
      }
    }
  }

  // Source boundary density
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

  // Receiver boundary density
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Fluid fluid forces
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

  // Source boundary forces
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

  // Receiver boundary forces
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

  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Compute forces with sorted grid
__global__ void compute_forces_hash_kernel(
    Particle *fluid_particles, int num_fluid_particles, int *particle_sorted_index,
    int *cell_start, int *cell_end,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_fluid_particles) {
    return;
  }

  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  int gx;
  int gy;
  int gz;
  position_to_grid_coords(p_i_x, p_i_y, p_i_z, gx, gy, gz);

  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Fluid fluid forces
  for (int dz_off = -1; dz_off <= 1; dz_off++) {
    for (int dy_off = -1; dy_off <= 1; dy_off++) {
      for (int dx_off = -1; dx_off <= 1; dx_off++) {
        int ngx = gx + dx_off;
        int ngy = gy + dy_off;
        int ngz = gz + dz_off;

        if ((ngx < 0) || (ngx >= HASH_GRID_SIZE_X) || (ngy < 0) ||
            (ngy >= HASH_GRID_SIZE_Y) || (ngz < 0) ||
            (ngz >= HASH_GRID_SIZE_Z)) {
          continue;
        }

        int neighbor_cell = grid_coords_to_index(ngx, ngy, ngz);
        int start = cell_start[neighbor_cell];

        if (start == -1) {
          continue;
        }

        int end = cell_end[neighbor_cell];

        for (int s = start; s < end; s++) {
          int j = particle_sorted_index[s];

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
                         fluid_particles[j].p / fmaxf(fluid_particles[j].rho *
                                                          fluid_particles[j].rho,
                                                      EPS));

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
      }
    }
  }

  // Source boundary forces
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

  // Receiver boundary forces
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

  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Run the density pass on the CPU
static void compute_density_pressure_cpu() {
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];
    float density = 0.0f;

    // Fluid fluid density
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;
      float r2 = dx * dx + dy * dy + dz * dz;

      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    // Source boundary density
    for (int j = 0; j < num_source_compute_boundary_particles; j++) {
      Particle &particle_j = source_compute_boundary_particles[j];
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;
      float r2 = dx * dx + dy * dy + dz * dz;

      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    // Receiver boundary density
    for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
      Particle &particle_j = receiver_compute_boundary_particles[j];
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;
      float r2 = dx * dx + dy * dy + dz * dz;

      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    particle_i.rho = std::max(density, REST_DENS * 0.1f);
    particle_i.p = std::max(GAS_CONST * (density - REST_DENS), 0.0f);
  }
}

// Run the force pass on the CPU
static void compute_forces_cpu() {
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];

    float pressure_fx = 0.0f;
    float pressure_fy = 0.0f;
    float pressure_fz = 0.0f;
    float viscosity_fx = 0.0f;
    float viscosity_fy = 0.0f;
    float viscosity_fz = 0.0f;

    // Fluid fluid forces
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      if (i == j) {
        continue;
      }

      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      if (r2 < H_FORCE * H_FORCE && r >= EPS) {
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        float p_term =
            -MASS * (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
                     particle_j.p / std::max(particle_j.rho * particle_j.rho, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;

        float visc_coeff = VISC_LAP * h_minus_r;
        float inv_rho_j = 1.0f / std::max(particle_j.rho, EPS);

        viscosity_fx += VISCOSITY * MASS * (particle_j.vx - particle_i.vx) *
                        inv_rho_j * visc_coeff;
        viscosity_fy += VISCOSITY * MASS * (particle_j.vy - particle_i.vy) *
                        inv_rho_j * visc_coeff;
        viscosity_fz += VISCOSITY * MASS * (particle_j.vz - particle_i.vz) *
                        inv_rho_j * visc_coeff;
      }
    }

    // Source boundary forces
    for (int j = 0; j < num_source_compute_boundary_particles; j++) {
      Particle &particle_j = source_compute_boundary_particles[j];
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      if (r2 < H_FORCE * H_FORCE && r >= EPS) {
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        float pb = particle_i.p;
        float rhob = particle_j.rho;

        float p_term =
            -MASS * (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
                     pb / std::max(rhob * rhob, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;
      }
    }

    // Receiver boundary forces
    for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
      Particle &particle_j = receiver_compute_boundary_particles[j];
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      if (r2 < H_FORCE * H_FORCE && r >= EPS) {
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        float pb = particle_i.p;
        float rhob = particle_j.rho;

        float p_term =
            -MASS * (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
                     pb / std::max(rhob * rhob, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;
      }
    }

    particle_i.fx = pressure_fx + viscosity_fx;
    particle_i.fy = pressure_fy + viscosity_fy + particle_i.rho * GRAVITY;
    particle_i.fz = pressure_fz + viscosity_fz;
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
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &p = fluid_particles[i];

    float ax = p.fx / std::fmaxf(p.rho, EPS);
    float ay = p.fy / std::fmaxf(p.rho, EPS);
    float az = p.fz / std::fmaxf(p.rho, EPS);

    p.vx += ax * DT;
    p.vy += ay * DT;
    p.vz += az * DT;

    p.vx *= VELOCITY_DAMPING;
    p.vy *= VELOCITY_DAMPING;
    p.vz *= VELOCITY_DAMPING;

    p.x += p.vx * DT;
    p.y += p.vy * DT;
    p.z += p.vz * DT;

    resolve_cup_collision(p, source_cup);
    resolve_cup_collision(p, receiver_cup);
    apply_world_box_collision(p);
  }
}

// Build the sorted spatial grid
void build_spatial_grid() {
  if (simulation_mode != SIM_MODE_GPU_SPATIAL_HASH) {
    return;
  }

  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  compute_particle_cell_id_kernel<<<blocks_per_grid, threads_per_block>>>(
      fluid_particles, num_fluid_particles, particle_cell_id,
      particle_sorted_index);
  cuda_check(cudaDeviceSynchronize(), "compute_particle_cell_id_kernel");

  thrust::device_ptr<int> key_begin =
      thrust::device_pointer_cast(particle_cell_id);
  thrust::device_ptr<int> value_begin =
      thrust::device_pointer_cast(particle_sorted_index);

  thrust::sort_by_key(thrust::device, key_begin, key_begin + num_fluid_particles,
                      value_begin);
  cuda_check(cudaDeviceSynchronize(), "thrust_sort_by_key");

  cuda_check(cudaMemset(cell_start, 0xFF,
                        HASH_GRID_CELL_COUNT * static_cast<int>(sizeof(int))),
             "cudaMemset(cell_start)");
  cuda_check(cudaMemset(cell_end, 0xFF,
                        HASH_GRID_CELL_COUNT * static_cast<int>(sizeof(int))),
             "cudaMemset(cell_end)");

  build_cell_ranges_kernel<<<blocks_per_grid, threads_per_block>>>(
      particle_cell_id, num_fluid_particles, cell_start, cell_end);
  cuda_check(cudaDeviceSynchronize(), "build_cell_ranges_kernel");
}

// Run the density pass
void compute_density_pressure() {
  if (simulation_mode == SIM_MODE_CPU_SEQUENTIAL) {
    compute_density_pressure_cpu();
    return;
  }

  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

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
    compute_density_pressure_hash_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles, particle_sorted_index, cell_start,
        cell_end,
        source_compute_boundary_particles, num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(), "compute_density_pressure_hash_kernel");
  }
}

// Run the force pass
void compute_forces() {
  if (simulation_mode == SIM_MODE_CPU_SEQUENTIAL) {
    compute_forces_cpu();
    return;
  }

  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  if (simulation_mode == SIM_MODE_GPU_BRUTE_FORCE) {
    compute_forces_bruteforce_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles,
        source_compute_boundary_particles, num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_bruteforce_kernel");
  } else {
    compute_forces_hash_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles, particle_sorted_index, cell_start,
        cell_end,
        source_compute_boundary_particles, num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_hash_kernel");
  }
}

// Export the scene to csv
void export_csv(int frame_index) {
  rebuild_boundary_particles_for_export();

  std::string frame_string = std::to_string(frame_index);
  frame_string = std::string(4 - frame_string.length(), '0') + frame_string;
  std::string file_name = "output/frame_" + frame_string + ".csv";

  std::ofstream file(file_name);

  file << "x,y,z,rho,p,is_boundary,kind\n";

  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }

  for (int i = 0; i < num_boundary_particles; i++) {
    Particle &particle = boundary_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }
}

// Print the frame stats
void print_stats(int frame_index) {
  if (num_fluid_particles == 0) {
    return;
  }

  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_speed = 0.0f;
  float max_speed = 0.0f;

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

  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  std::cout << "Frame " << frame_index << " | rho: [" << min_rho << ", "
            << max_rho << "] avg=" << avg_rho << " | p: [" << min_p << ", "
            << max_p << "]"
            << " | speed avg=" << avg_speed << " max=" << max_speed
            << std::endl;
}

// Print the initial stats
void print_initial_density_stats() {
  if (num_fluid_particles == 0) {
    return;
  }

  if (simulation_mode == SIM_MODE_GPU_SPATIAL_HASH) {
    build_spatial_grid();
  }

  compute_density_pressure();

  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_p = 0.0f;

  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    min_rho = std::min(min_rho, particle.rho);
    max_rho = std::max(max_rho, particle.rho);
    min_p = std::min(min_p, particle.p);
    max_p = std::max(max_p, particle.p);
    sum_rho += particle.rho;
    sum_p += particle.p;
  }

  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);

  std::cout << "Initial density stats | rho: [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p: [" << min_p << ", " << max_p
            << "] avg = " << avg_p << std::endl;
}

// Print fluid only stats
void print_fluid_only_stats(const std::string &label) {
  if (num_fluid_particles == 0) {
    std::cout << label << " | no fluid particles" << std::endl;
    return;
  }

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

  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  std::cout << label << " | fluid_count = " << num_fluid_particles
            << " | rho = [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p = [" << min_p << ", " << max_p
            << "] avg = " << avg_p << " | speed = [" << min_speed << ", "
            << max_speed << "] avg = " << avg_speed
            << " | rho_floor_count = " << rho_floor_count
            << " | zero_pressure_count = " << zero_pressure_count << std::endl;
}

// Read the mode from the command line
static SimulationMode parse_mode(int argc, char **argv) {
  if (argc < 3) {
    return SIM_MODE_GPU_SPATIAL_HASH;
  }

  std::string mode_string = argv[2];

  if ((mode_string == "sequential") || (mode_string == "cpu")) {
    return SIM_MODE_CPU_SEQUENTIAL;
  }

  if ((mode_string == "brute") || (mode_string == "gpu_brute") ||
      (mode_string == "all_pairs")) {
    return SIM_MODE_GPU_BRUTE_FORCE;
  }

  if ((mode_string == "hash") || (mode_string == "spatial_hash") ||
      (mode_string == "grid")) {
    return SIM_MODE_GPU_SPATIAL_HASH;
  }

  std::cerr << "Unknown mode " << mode_string << std::endl;
  std::cerr << "Use one of sequential brute hash" << std::endl;
  std::exit(1);
}

// Run the full simulation
int main(int argc, char **argv) {
  using clock_type = std::chrono::high_resolution_clock;

  srand(0);

  float target_tilt_deg = 0.0f;

  if (argc >= 2) {
    target_tilt_deg = std::stof(argv[1]);
  }

  target_tilt_deg = std::max(0.0f, std::min(180.0f, target_tilt_deg));
  simulation_mode = parse_mode(argc, argv);

  cuda_check(cudaMallocManaged(&fluid_particles,
                               MAX_FLUID_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(fluid_particles)");
  cuda_check(cudaMallocManaged(&boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(boundary_particles)");
  cuda_check(cudaMallocManaged(&source_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(source_compute_boundary_particles)");
  cuda_check(cudaMallocManaged(&receiver_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(receiver_compute_boundary_particles)");
  cuda_check(cudaMallocManaged(&particle_cell_id,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(particle_cell_id)");
  cuda_check(cudaMallocManaged(&particle_sorted_index,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(particle_sorted_index)");
  cuda_check(cudaMallocManaged(&cell_start,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(cell_start)");
  cuda_check(cudaMallocManaged(&cell_end,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(cell_end)");

  std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg
            << std::endl;
  std::cout << "Simulation mode = " << mode_to_string(simulation_mode)
            << std::endl;

  initialize_scene(target_tilt_deg);

  print_initial_density_stats();
  print_fluid_only_stats("Frame 0 fluid stats");
  print_source_cup_setup_stats();

  std::cout << "Fluid particles = " << num_fluid_particles << std::endl;
  std::cout << "Source compute boundary particles = "
            << num_source_compute_boundary_particles << std::endl;
  std::cout << "Receiver compute boundary particles = "
            << num_receiver_compute_boundary_particles << std::endl;
  std::cout << "Total compute boundary particles = "
            << (num_source_compute_boundary_particles +
                num_receiver_compute_boundary_particles)
            << std::endl;
  std::cout << "Render boundary particles = " << num_boundary_particles
            << std::endl;

  export_csv(0);

  double total_compute_ms = 0.0;
  double min_frame_ms = std::numeric_limits<double>::max();
  double max_frame_ms = 0.0;

  for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {
    reset_penetration_stats();

    auto frame_start = clock_type::now();

    for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {
      float substep_frame_index = static_cast<float>(frame_index - 1) +
                                  static_cast<float>(step_index + 1) /
                                      static_cast<float>(SUBSTEPS_PER_FRAME);

      update_scene_for_frame(substep_frame_index, target_tilt_deg);

      // Rebuild only the moving source compute boundary
      rebuild_source_compute_boundary_particles();

      if (simulation_mode == SIM_MODE_GPU_SPATIAL_HASH) {
        build_spatial_grid();
      }

      compute_density_pressure();
      compute_forces();
      integrate_fluid_particles();
    }

    auto frame_end = clock_type::now();
    double frame_ms =
        std::chrono::duration<double, std::milli>(frame_end - frame_start)
            .count();

    total_compute_ms += frame_ms;
    min_frame_ms = std::min(min_frame_ms, frame_ms);
    max_frame_ms = std::max(max_frame_ms, frame_ms);

    std::cout << std::fixed << std::setprecision(3)
              << "FrameTiming " << frame_index << " " << frame_ms
              << std::endl;

    if (frame_index % EXPORT_EVERY == 0) {
      export_csv(frame_index / EXPORT_EVERY);
    }

    if (frame_index % 20 == 0) {
      print_stats(frame_index);
      print_fluid_only_stats("Frame " + std::to_string(frame_index) +
                             " fluid stats");
      print_penetration_stats(frame_index);
    }
  }

  double avg_frame_ms = total_compute_ms / static_cast<double>(FRAME_COUNT);

  std::cout << std::fixed << std::setprecision(3)
            << "TimingSummary mode=" << mode_to_string(simulation_mode)
            << " total_ms=" << total_compute_ms
            << " avg_ms=" << avg_frame_ms
            << " min_ms=" << min_frame_ms
            << " max_ms=" << max_frame_ms << std::endl;

  std::cout << "Done" << std::endl;

  cuda_check(cudaFree(cell_end), "cudaFree(cell_end)");
  cuda_check(cudaFree(cell_start), "cudaFree(cell_start)");
  cuda_check(cudaFree(particle_sorted_index), "cudaFree(particle_sorted_index)");
  cuda_check(cudaFree(particle_cell_id), "cudaFree(particle_cell_id)");
  cuda_check(cudaFree(receiver_compute_boundary_particles),
             "cudaFree(receiver_compute_boundary_particles)");
  cuda_check(cudaFree(source_compute_boundary_particles),
             "cudaFree(source_compute_boundary_particles)");
  cuda_check(cudaFree(boundary_particles), "cudaFree(boundary_particles)");
  cuda_check(cudaFree(fluid_particles), "cudaFree(fluid_particles)");

  return 0;
}