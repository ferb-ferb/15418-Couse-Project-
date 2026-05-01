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

// Store the fluid particle spatial hash arrays
static int *fluid_particle_cell_id;
static int *fluid_particle_sorted_index;
static int *fluid_cell_start;
static int *fluid_cell_end;

// Store the moving source boundary spatial hash arrays
static int *source_boundary_cell_id;
static int *source_boundary_sorted_index;
static int *source_boundary_cell_start;
static int *source_boundary_cell_end;

// Store the static receiver boundary spatial hash arrays
static int *receiver_boundary_cell_id;
static int *receiver_boundary_sorted_index;
static int *receiver_boundary_cell_start;
static int *receiver_boundary_cell_end;

// Convert the selected simulation mode to text
static const char *mode_to_string(SimulationMode mode) {

  // Return the CPU mode label
  if (mode == SIM_MODE_CPU_SEQUENTIAL) {
    return "cpu_sequential";

  // Return the GPU brute force mode label
  } else if (mode == SIM_MODE_GPU_BRUTE_FORCE) {
    return "gpu_bruteforce";

  // Return the spatial hash mode label
  } else {
    return "gpu_spatial_hash";
  }
}

// Stop the program when a CUDA call fails
static void cuda_check(cudaError_t code, const char *label) {

  // Check the CUDA return code
  if (code != cudaSuccess) {

    // Print the failing operation and CUDA error
    std::cerr << "CUDA error in " << label << " " << cudaGetErrorString(code)
              << std::endl;

    // Exit immediately after a CUDA failure
    std::exit(1);
  }
}

// Clamp one integer to a valid range
__device__ __forceinline__ static int clamp_int(int value, int low, int high) {

  // Clamp values below the lower bound
  if (value < low) {
    return low;
  }

  // Clamp values above the upper bound
  if (value > high) {
    return high;
  }

  // Return values already inside the range
  return value;
}

// Convert 3D grid coordinates into one flat cell index
__device__ __forceinline__ static int grid_coords_to_index(int gx, int gy,
                                                           int gz) {

  // Flatten the 3D grid cell into a 1D array index
  return gx + HASH_GRID_SIZE_X * (gy + HASH_GRID_SIZE_Y * gz);
}

// Convert one world position into spatial hash grid coordinates
__device__ __forceinline__ static void
position_to_grid_coords(float x, float y, float z, int &gx, int &gy, int &gz) {

  // Convert world position into raw grid coordinates
  gx = static_cast<int>(floorf((x - BOX_X_MIN) / HASH_CELL_SIZE));
  gy = static_cast<int>(floorf((y - BOX_Y_MIN) / HASH_CELL_SIZE));
  gz = static_cast<int>(floorf((z - BOX_Z_MIN) / HASH_CELL_SIZE));

  // Clamp grid coordinates to valid grid bounds
  gx = clamp_int(gx, 0, HASH_GRID_SIZE_X - 1);
  gy = clamp_int(gy, 0, HASH_GRID_SIZE_Y - 1);
  gz = clamp_int(gz, 0, HASH_GRID_SIZE_Z - 1);
}

// Build one cell id and sorted index entry for every particle in one array
__global__ void compute_particle_cell_id_kernel(Particle *particles,
                                                int num_particles,
                                                int *particle_cell_id,
                                                int *particle_sorted_index) {
  // Map this CUDA thread to one particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the particle range
  if (i >= num_particles) {
    return;
  }

  // Convert this particle position to grid coordinates
  int gx;
  int gy;
  int gz;
  position_to_grid_coords(particles[i].x, particles[i].y, particles[i].z, gx,
                          gy, gz);

  // Store the flat cell id for sorting
  particle_cell_id[i] = grid_coords_to_index(gx, gy, gz);

  // Store the original particle index
  particle_sorted_index[i] = i;
}

// Build start and end ranges for every occupied grid cell
__global__ void build_cell_ranges_kernel(int *particle_cell_id,
                                         int num_particles, int *cell_start,
                                         int *cell_end) {
  // Map this CUDA thread to one sorted particle entry
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the sorted particle range
  if (i >= num_particles) {
    return;
  }

  // Read this sorted particle cell
  int current_cell = particle_cell_id[i];

  // Mark the start of a new cell range
  if ((i == 0) || (particle_cell_id[i - 1] != current_cell)) {
    cell_start[current_cell] = i;
  }

  // Mark the end of the current cell range
  if ((i == num_particles - 1) || (particle_cell_id[i + 1] != current_cell)) {
    cell_end[current_cell] = i + 1;
  }
}

// Build one sorted spatial grid for one particle array
static void build_sorted_grid_for_particles(Particle *particles,
                                            int num_particles,
                                            int *particle_cell_id,
                                            int *particle_sorted_index,
                                            int *cell_start, int *cell_end,
                                            const std::string &label) {

  // Clear all cell start markers
  cuda_check(cudaMemset(cell_start, 0xFF,
                        HASH_GRID_CELL_COUNT * static_cast<int>(sizeof(int))),
             ("cudaMemset(cell_start) " + label).c_str());

  // Clear all cell end markers
  cuda_check(cudaMemset(cell_end, 0xFF,
                        HASH_GRID_CELL_COUNT * static_cast<int>(sizeof(int))),
             ("cudaMemset(cell_end) " + label).c_str());

  // Stop early when this particle set is empty
  if (num_particles == 0) {
    return;
  }

  // Build the CUDA launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_particles + threads_per_block - 1) / threads_per_block;

  // Compute one cell id for every particle
  compute_particle_cell_id_kernel<<<blocks_per_grid, threads_per_block>>>(
      particles, num_particles, particle_cell_id, particle_sorted_index);
  cuda_check(cudaDeviceSynchronize(),
             ("compute_particle_cell_id_kernel " + label).c_str());

  // Wrap raw device pointers for thrust sorting
  thrust::device_ptr<int> key_begin =
      thrust::device_pointer_cast(particle_cell_id);
  thrust::device_ptr<int> value_begin =
      thrust::device_pointer_cast(particle_sorted_index);

  // Sort particle indices by cell id
  thrust::sort_by_key(thrust::device, key_begin, key_begin + num_particles,
                      value_begin);
  cuda_check(cudaDeviceSynchronize(), ("thrust_sort_by_key " + label).c_str());

  // Build start and end ranges for every occupied cell
  build_cell_ranges_kernel<<<blocks_per_grid, threads_per_block>>>(
      particle_cell_id, num_particles, cell_start, cell_end);
  cuda_check(cudaDeviceSynchronize(),
             ("build_cell_ranges_kernel " + label).c_str());
}

// Build the fluid particle spatial grid
void build_spatial_grid() {

  // Skip grid work outside spatial hash mode
  if (simulation_mode != SIM_MODE_GPU_SPATIAL_HASH) {
    return;
  }

  // Build the sorted grid for fluid particles
  build_sorted_grid_for_particles(
      fluid_particles, num_fluid_particles, fluid_particle_cell_id,
      fluid_particle_sorted_index, fluid_cell_start, fluid_cell_end, "fluid");
}

// Build the moving source boundary spatial grid
static void build_source_boundary_grid() {

  // Skip grid work outside spatial hash mode
  if (simulation_mode != SIM_MODE_GPU_SPATIAL_HASH) {
    return;
  }

  // Build the sorted grid for source compute boundary particles
  build_sorted_grid_for_particles(
      source_compute_boundary_particles, num_source_compute_boundary_particles,
      source_boundary_cell_id, source_boundary_sorted_index,
      source_boundary_cell_start, source_boundary_cell_end, "source_boundary");
}

// Build the static receiver boundary spatial grid
static void build_receiver_boundary_grid() {

  // Skip grid work outside spatial hash mode
  if (simulation_mode != SIM_MODE_GPU_SPATIAL_HASH) {
    return;
  }

  // Build the sorted grid for receiver compute boundary particles
  build_sorted_grid_for_particles(
      receiver_compute_boundary_particles,
      num_receiver_compute_boundary_particles, receiver_boundary_cell_id,
      receiver_boundary_sorted_index, receiver_boundary_cell_start,
      receiver_boundary_cell_end, "receiver_boundary");
}

// Compute density and pressure with brute force GPU search
// One CUDA thread owns one fluid particle
// Each thread scans fluid source boundary and receiver boundary arrays
__global__ void compute_density_pressure_bruteforce_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    Particle *source_compute_boundary_particles,
    int num_source_compute_boundary_particles,
    Particle *receiver_compute_boundary_particles,
    int num_receiver_compute_boundary_particles) {

  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the query particle position
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  // Start the density accumulator
  float density = 0.0f;

  // Sum density from all fluid particles
  for (int j = 0; j < num_fluid_particles; j++) {

    // Build the fluid to fluid offset
    float dx = fluid_particles[j].x - p_i_x;
    float dy = fluid_particles[j].y - p_i_y;
    float dz = fluid_particles[j].z - p_i_z;

    // Build squared distance
    float r2 = dx * dx + dy * dy + dz * dz;

    // Add contribution if inside the density radius
    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum density from the source compute boundary
  for (int j = 0; j < num_source_compute_boundary_particles; j++) {

    // Build the fluid to source boundary offset
    float dx = source_compute_boundary_particles[j].x - p_i_x;
    float dy = source_compute_boundary_particles[j].y - p_i_y;
    float dz = source_compute_boundary_particles[j].z - p_i_z;

    // Build squared distance
    float r2 = dx * dx + dy * dy + dz * dz;

    // Add contribution if inside the density radius
    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Sum density from the receiver compute boundary
  for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {

    // Build the fluid to receiver boundary offset
    float dx = receiver_compute_boundary_particles[j].x - p_i_x;
    float dy = receiver_compute_boundary_particles[j].y - p_i_y;
    float dz = receiver_compute_boundary_particles[j].z - p_i_z;

    // Build squared distance
    float r2 = dx * dx + dy * dy + dz * dz;

    // Add contribution if inside the density radius
    if (r2 < H_DENS * H_DENS) {
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      density += MASS * POLY6 * weight;
    }
  }

  // Store the final density and pressure
  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Compute density and pressure with three sorted spatial grids
// One CUDA thread owns one fluid particle
// Each thread checks nearby fluid source boundary and receiver boundary cells
__global__ void compute_density_pressure_hash_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    int *fluid_particle_sorted_index, int *fluid_cell_start,
    int *fluid_cell_end, Particle *source_compute_boundary_particles,
    int *source_boundary_sorted_index, int *source_boundary_cell_start,
    int *source_boundary_cell_end,
    Particle *receiver_compute_boundary_particles,
    int *receiver_boundary_sorted_index, int *receiver_boundary_cell_start,
    int *receiver_boundary_cell_end) {

  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the query particle position
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  // Convert query particle position to grid coordinates
  int gx;
  int gy;
  int gz;
  position_to_grid_coords(p_i_x, p_i_y, p_i_z, gx, gy, gz);

  // Start the density accumulator
  float density = 0.0f;

  // Visit neighboring cells around the query particle
  for (int dz_off = -1; dz_off <= 1; dz_off++) {
    for (int dy_off = -1; dy_off <= 1; dy_off++) {
      for (int dx_off = -1; dx_off <= 1; dx_off++) {

        // Build neighboring grid coordinates
        int ngx = gx + dx_off;
        int ngy = gy + dy_off;
        int ngz = gz + dz_off;

        // Skip neighbor cells outside the grid
        if ((ngx < 0) || (ngx >= HASH_GRID_SIZE_X) || (ngy < 0) ||
            (ngy >= HASH_GRID_SIZE_Y) || (ngz < 0) ||
            (ngz >= HASH_GRID_SIZE_Z)) {
          continue;
        }

        // Convert neighbor cell coordinates to a flat index
        int neighbor_cell = grid_coords_to_index(ngx, ngy, ngz);

        // Read the fluid cell range
        int fluid_start = fluid_cell_start[neighbor_cell];

        // Sum density from nearby fluid particles
        if (fluid_start != -1) {
          int fluid_end = fluid_cell_end[neighbor_cell];

          for (int s = fluid_start; s < fluid_end; s++) {
            int j = fluid_particle_sorted_index[s];

            // Build the fluid to fluid offset
            float dx = fluid_particles[j].x - p_i_x;
            float dy = fluid_particles[j].y - p_i_y;
            float dz = fluid_particles[j].z - p_i_z;

            // Build squared distance
            float r2 = dx * dx + dy * dy + dz * dz;

            // Add contribution if inside the density radius
            if (r2 < H_DENS * H_DENS) {
              float h2_minus_r2 = H_DENS * H_DENS - r2;
              float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
              density += MASS * POLY6 * weight;
            }
          }
        }

        // Read the source boundary cell range
        int source_start = source_boundary_cell_start[neighbor_cell];

        // Sum density from nearby source boundary particles
        if (source_start != -1) {
          int source_end = source_boundary_cell_end[neighbor_cell];

          for (int s = source_start; s < source_end; s++) {
            int j = source_boundary_sorted_index[s];

            // Build the fluid to source boundary offset
            float dx = source_compute_boundary_particles[j].x - p_i_x;
            float dy = source_compute_boundary_particles[j].y - p_i_y;
            float dz = source_compute_boundary_particles[j].z - p_i_z;

            // Build squared distance
            float r2 = dx * dx + dy * dy + dz * dz;

            // Add contribution if inside the density radius
            if (r2 < H_DENS * H_DENS) {
              float h2_minus_r2 = H_DENS * H_DENS - r2;
              float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
              density += MASS * POLY6 * weight;
            }
          }
        }

        // Read the receiver boundary cell range
        int receiver_start = receiver_boundary_cell_start[neighbor_cell];

        // Sum density from nearby receiver boundary particles
        if (receiver_start != -1) {
          int receiver_end = receiver_boundary_cell_end[neighbor_cell];

          for (int s = receiver_start; s < receiver_end; s++) {
            int j = receiver_boundary_sorted_index[s];

            // Build the fluid to receiver boundary offset
            float dx = receiver_compute_boundary_particles[j].x - p_i_x;
            float dy = receiver_compute_boundary_particles[j].y - p_i_y;
            float dz = receiver_compute_boundary_particles[j].z - p_i_z;

            // Build squared distance
            float r2 = dx * dx + dy * dy + dz * dz;

            // Add contribution if inside the density radius
            if (r2 < H_DENS * H_DENS) {
              float h2_minus_r2 = H_DENS * H_DENS - r2;
              float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
              density += MASS * POLY6 * weight;
            }
          }
        }
      }
    }
  }

  // Store the final density and pressure
  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Compute forces with brute force GPU search
// One CUDA thread owns one fluid particle
// Each thread scans fluid source boundary and receiver boundary arrays
__global__ void
compute_forces_bruteforce_kernel(Particle *fluid_particles,
                                 int num_fluid_particles,
                                 Particle *source_compute_boundary_particles,
                                 int num_source_compute_boundary_particles,
                                 Particle *receiver_compute_boundary_particles,
                                 int num_receiver_compute_boundary_particles) {

  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the query particle state
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  // Start force accumulators
  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Sum pressure and viscosity from all fluid particles
  for (int j = 0; j < num_fluid_particles; j++) {

    // Skip self interaction
    if (i == j) {
      continue;
    }

    // Build the fluid to fluid offset
    float dx = p_i_x - fluid_particles[j].x;
    float dy = p_i_y - fluid_particles[j].y;
    float dz = p_i_z - fluid_particles[j].z;

    // Build squared distance and true distance
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    // Add force contribution if inside the force radius
    if (r2 < H_FORCE * H_FORCE && r >= EPS) {

      // Build smoothing values
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      // Accumulate pressure force
      float p_term =
          -MASS *
          (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
           fluid_particles[j].p /
               fmaxf(fluid_particles[j].rho * fluid_particles[j].rho, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      // Accumulate viscosity force
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

  // Sum pressure force from the source compute boundary
  for (int j = 0; j < num_source_compute_boundary_particles; j++) {

    // Build the fluid to source boundary offset
    float dx = p_i_x - source_compute_boundary_particles[j].x;
    float dy = p_i_y - source_compute_boundary_particles[j].y;
    float dz = p_i_z - source_compute_boundary_particles[j].z;

    // Build squared distance and true distance
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    // Add wall pressure if inside the force radius
    if (r2 < H_FORCE * H_FORCE && r >= EPS) {

      // Build smoothing values
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      // Use the fluid pressure as the wall pressure response
      float pb = p_i_p;
      float rhob = source_compute_boundary_particles[j].rho;

      // Accumulate boundary pressure force
      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Sum pressure force from the receiver compute boundary
  for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {

    // Build the fluid to receiver boundary offset
    float dx = p_i_x - receiver_compute_boundary_particles[j].x;
    float dy = p_i_y - receiver_compute_boundary_particles[j].y;
    float dz = p_i_z - receiver_compute_boundary_particles[j].z;

    // Build squared distance and true distance
    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    // Add wall pressure if inside the force radius
    if (r2 < H_FORCE * H_FORCE && r >= EPS) {

      // Build smoothing values
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

      // Use the fluid pressure as the wall pressure response
      float pb = p_i_p;
      float rhob = receiver_compute_boundary_particles[j].rho;

      // Accumulate boundary pressure force
      float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;
    }
  }

  // Store the final force with gravity
  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Compute forces with three sorted spatial grids
// One CUDA thread owns one fluid particle
// Each thread checks nearby fluid source boundary and receiver boundary cells
__global__ void compute_forces_hash_kernel(
    Particle *fluid_particles, int num_fluid_particles,
    int *fluid_particle_sorted_index, int *fluid_cell_start,
    int *fluid_cell_end, Particle *source_compute_boundary_particles,
    int *source_boundary_sorted_index, int *source_boundary_cell_start,
    int *source_boundary_cell_end,
    Particle *receiver_compute_boundary_particles,
    int *receiver_boundary_sorted_index, int *receiver_boundary_cell_start,
    int *receiver_boundary_cell_end) {

  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles) {
    return;
  }

  // Cache the query particle state
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;
  float p_i_vx = fluid_particles[i].vx;
  float p_i_vy = fluid_particles[i].vy;
  float p_i_vz = fluid_particles[i].vz;
  float p_i_rho = fluid_particles[i].rho;
  float p_i_p = fluid_particles[i].p;

  // Convert query particle position to grid coordinates
  int gx;
  int gy;
  int gz;
  position_to_grid_coords(p_i_x, p_i_y, p_i_z, gx, gy, gz);

  // Start force accumulators
  float pressure_fx = 0.0f;
  float pressure_fy = 0.0f;
  float pressure_fz = 0.0f;
  float viscosity_fx = 0.0f;
  float viscosity_fy = 0.0f;
  float viscosity_fz = 0.0f;

  // Visit neighboring cells around the query particle
  for (int dz_off = -1; dz_off <= 1; dz_off++) {
    for (int dy_off = -1; dy_off <= 1; dy_off++) {
      for (int dx_off = -1; dx_off <= 1; dx_off++) {

        // Build neighboring grid coordinates
        int ngx = gx + dx_off;
        int ngy = gy + dy_off;
        int ngz = gz + dz_off;

        // Skip neighbor cells outside the grid
        if ((ngx < 0) || (ngx >= HASH_GRID_SIZE_X) || (ngy < 0) ||
            (ngy >= HASH_GRID_SIZE_Y) || (ngz < 0) ||
            (ngz >= HASH_GRID_SIZE_Z)) {
          continue;
        }

        // Convert neighbor cell coordinates to a flat index
        int neighbor_cell = grid_coords_to_index(ngx, ngy, ngz);

        // Read the fluid cell range
        int fluid_start = fluid_cell_start[neighbor_cell];

        // Sum forces from nearby fluid particles
        if (fluid_start != -1) {
          int fluid_end = fluid_cell_end[neighbor_cell];

          for (int s = fluid_start; s < fluid_end; s++) {
            int j = fluid_particle_sorted_index[s];

            // Skip self interaction
            if (i == j) {
              continue;
            }

            // Build the fluid to fluid offset
            float dx = p_i_x - fluid_particles[j].x;
            float dy = p_i_y - fluid_particles[j].y;
            float dz = p_i_z - fluid_particles[j].z;

            // Build squared distance and true distance
            float r2 = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(r2);

            // Add force contribution if inside the force radius
            if (r2 < H_FORCE * H_FORCE && r >= EPS) {

              // Build smoothing values
              float h_minus_r = H_FORCE - r;
              float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

              // Accumulate pressure force
              float p_term =
                  -MASS *
                  (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                   fluid_particles[j].p /
                       fmaxf(fluid_particles[j].rho * fluid_particles[j].rho,
                             EPS));

              pressure_fx += p_term * grad_coeff * dx / r;
              pressure_fy += p_term * grad_coeff * dy / r;
              pressure_fz += p_term * grad_coeff * dz / r;

              // Accumulate viscosity force
              float visc_coeff = VISC_LAP * h_minus_r;
              float inv_rho_j = 1.0f / fmaxf(fluid_particles[j].rho, EPS);

              viscosity_fx += VISCOSITY * MASS *
                              (fluid_particles[j].vx - p_i_vx) * inv_rho_j *
                              visc_coeff;
              viscosity_fy += VISCOSITY * MASS *
                              (fluid_particles[j].vy - p_i_vy) * inv_rho_j *
                              visc_coeff;
              viscosity_fz += VISCOSITY * MASS *
                              (fluid_particles[j].vz - p_i_vz) * inv_rho_j *
                              visc_coeff;
            }
          }
        }

        // Read the source boundary cell range
        int source_start = source_boundary_cell_start[neighbor_cell];

        // Sum forces from nearby source boundary particles
        if (source_start != -1) {
          int source_end = source_boundary_cell_end[neighbor_cell];

          for (int s = source_start; s < source_end; s++) {
            int j = source_boundary_sorted_index[s];

            // Build the fluid to source boundary offset
            float dx = p_i_x - source_compute_boundary_particles[j].x;
            float dy = p_i_y - source_compute_boundary_particles[j].y;
            float dz = p_i_z - source_compute_boundary_particles[j].z;

            // Build squared distance and true distance
            float r2 = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(r2);

            // Add wall pressure if inside the force radius
            if (r2 < H_FORCE * H_FORCE && r >= EPS) {

              // Build smoothing values
              float h_minus_r = H_FORCE - r;
              float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

              // Use the fluid pressure as the wall pressure response
              float pb = p_i_p;
              float rhob = source_compute_boundary_particles[j].rho;

              // Accumulate boundary pressure force
              float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                                      pb / fmaxf(rhob * rhob, EPS));

              pressure_fx += p_term * grad_coeff * dx / r;
              pressure_fy += p_term * grad_coeff * dy / r;
              pressure_fz += p_term * grad_coeff * dz / r;
            }
          }
        }

        // Read the receiver boundary cell range
        int receiver_start = receiver_boundary_cell_start[neighbor_cell];

        // Sum forces from nearby receiver boundary particles
        if (receiver_start != -1) {
          int receiver_end = receiver_boundary_cell_end[neighbor_cell];

          for (int s = receiver_start; s < receiver_end; s++) {
            int j = receiver_boundary_sorted_index[s];

            // Build the fluid to receiver boundary offset
            float dx = p_i_x - receiver_compute_boundary_particles[j].x;
            float dy = p_i_y - receiver_compute_boundary_particles[j].y;
            float dz = p_i_z - receiver_compute_boundary_particles[j].z;

            // Build squared distance and true distance
            float r2 = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(r2);

            // Add wall pressure if inside the force radius
            if (r2 < H_FORCE * H_FORCE && r >= EPS) {

              // Build smoothing values
              float h_minus_r = H_FORCE - r;
              float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

              // Use the fluid pressure as the wall pressure response
              float pb = p_i_p;
              float rhob = receiver_compute_boundary_particles[j].rho;

              // Accumulate boundary pressure force
              float p_term = -MASS * (p_i_p / fmaxf(p_i_rho * p_i_rho, EPS) +
                                      pb / fmaxf(rhob * rhob, EPS));

              pressure_fx += p_term * grad_coeff * dx / r;
              pressure_fy += p_term * grad_coeff * dy / r;
              pressure_fz += p_term * grad_coeff * dz / r;
            }
          }
        }
      }
    }
  }

  // Store the final force with gravity
  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Compute density and pressure on the CPU
// This is kept for the sequential comparison mode
static void compute_density_pressure_cpu() {

  // Loop over every fluid particle
  for (int i = 0; i < num_fluid_particles; i++) {

    // Get the query particle
    Particle &particle_i = fluid_particles[i];

    // Start the density accumulator
    float density = 0.0f;

    // Sum density from all fluid particles
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      // Build the fluid to fluid offset
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;

      // Build squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Add contribution if inside the density radius
      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    // Sum density from the source compute boundary
    for (int j = 0; j < num_source_compute_boundary_particles; j++) {
      Particle &particle_j = source_compute_boundary_particles[j];

      // Build the fluid to source boundary offset
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;

      // Build squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Add contribution if inside the density radius
      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    // Sum density from the receiver compute boundary
    for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
      Particle &particle_j = receiver_compute_boundary_particles[j];

      // Build the fluid to receiver boundary offset
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;

      // Build squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Add contribution if inside the density radius
      if (r2 < H_DENS * H_DENS) {
        float h2_minus_r2 = H_DENS * H_DENS - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        density += MASS * POLY6 * weight;
      }
    }

    // Store density and pressure
    particle_i.rho = std::max(density, REST_DENS * 0.1f);
    particle_i.p = std::max(GAS_CONST * (density - REST_DENS), 0.0f);
  }
}

// Compute forces on the CPU
// This is kept for the sequential comparison mode
static void compute_forces_cpu() {

  // Loop over every fluid particle
  for (int i = 0; i < num_fluid_particles; i++) {

    // Get the query particle
    Particle &particle_i = fluid_particles[i];

    // Start force accumulators
    float pressure_fx = 0.0f;
    float pressure_fy = 0.0f;
    float pressure_fz = 0.0f;
    float viscosity_fx = 0.0f;
    float viscosity_fy = 0.0f;
    float viscosity_fz = 0.0f;

    // Sum pressure and viscosity from all fluid particles
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      // Skip self interaction
      if (i == j) {
        continue;
      }

      // Build the fluid to fluid offset
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;

      // Build squared distance and true distance
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      // Add force contribution if inside the force radius
      if (r2 < H_FORCE * H_FORCE && r >= EPS) {

        // Build smoothing values
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        // Accumulate pressure force
        float p_term =
            -MASS *
            (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
             particle_j.p / std::max(particle_j.rho * particle_j.rho, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;

        // Accumulate viscosity force
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

    // Sum pressure force from the source compute boundary
    for (int j = 0; j < num_source_compute_boundary_particles; j++) {
      Particle &particle_j = source_compute_boundary_particles[j];

      // Build the fluid to source boundary offset
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;

      // Build squared distance and true distance
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      // Add wall pressure if inside the force radius
      if (r2 < H_FORCE * H_FORCE && r >= EPS) {

        // Build smoothing values
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        // Use the fluid pressure as the wall pressure response
        float pb = particle_i.p;
        float rhob = particle_j.rho;

        // Accumulate boundary pressure force
        float p_term =
            -MASS *
            (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
             pb / std::max(rhob * rhob, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;
      }
    }

    // Sum pressure force from the receiver compute boundary
    for (int j = 0; j < num_receiver_compute_boundary_particles; j++) {
      Particle &particle_j = receiver_compute_boundary_particles[j];

      // Build the fluid to receiver boundary offset
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;

      // Build squared distance and true distance
      float r2 = dx * dx + dy * dy + dz * dz;
      float r = std::sqrt(r2);

      // Add wall pressure if inside the force radius
      if (r2 < H_FORCE * H_FORCE && r >= EPS) {

        // Build smoothing values
        float h_minus_r = H_FORCE - r;
        float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;

        // Use the fluid pressure as the wall pressure response
        float pb = particle_i.p;
        float rhob = particle_j.rho;

        // Accumulate boundary pressure force
        float p_term =
            -MASS *
            (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
             pb / std::max(rhob * rhob, EPS));

        pressure_fx += p_term * grad_coeff * dx / r;
        pressure_fy += p_term * grad_coeff * dy / r;
        pressure_fz += p_term * grad_coeff * dz / r;
      }
    }

    // Store the final force with gravity
    particle_i.fx = pressure_fx + viscosity_fx;
    particle_i.fy = pressure_fy + viscosity_fy + particle_i.rho * GRAVITY;
    particle_i.fz = pressure_fz + viscosity_fz;
  }
}

// Push one fluid particle back into the world box
static void apply_world_box_collision(Particle &particle) {

  // Resolve the left world wall
  if (particle.x < BOX_X_MIN) {
    particle.x = BOX_X_MIN + WALL_EPS;

    if (particle.vx < 0.0f) {
      particle.vx = -particle.vx * WALL_RESTITUTION;
    }

  // Resolve the right world wall
  } else if (particle.x > BOX_X_MAX) {
    particle.x = BOX_X_MAX - WALL_EPS;

    if (particle.vx > 0.0f) {
      particle.vx = -particle.vx * WALL_RESTITUTION;
    }
  }

  // Resolve the floor
  if (particle.y < BOX_Y_MIN) {
    particle.y = BOX_Y_MIN + WALL_EPS;

    if (particle.vy < 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }

  // Resolve the ceiling
  } else if (particle.y > BOX_Y_MAX) {
    particle.y = BOX_Y_MAX - WALL_EPS;

    if (particle.vy > 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }
  }

  // Resolve the front world wall
  if (particle.z < BOX_Z_MIN) {
    particle.z = BOX_Z_MIN + WALL_EPS;

    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

  // Resolve the back world wall
  } else if (particle.z > BOX_Z_MAX) {
    particle.z = BOX_Z_MAX - WALL_EPS;

    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }
  }
}

// Integrate velocity and position on the CPU
void integrate_fluid_particles() {

  // Loop over every fluid particle
  for (int i = 0; i < num_fluid_particles; i++) {

    // Get a reference to the actual particle in memory
    Particle &p = fluid_particles[i];

    // Convert force into acceleration
    float ax = p.fx / std::fmaxf(p.rho, EPS);
    float ay = p.fy / std::fmaxf(p.rho, EPS);
    float az = p.fz / std::fmaxf(p.rho, EPS);

    // Update velocity from acceleration
    p.vx += ax * DT;
    p.vy += ay * DT;
    p.vz += az * DT;

    // Apply global damping to reduce instability
    p.vx *= VELOCITY_DAMPING;
    p.vy *= VELOCITY_DAMPING;
    p.vz *= VELOCITY_DAMPING;

    // Update particle position
    p.x += p.vx * DT;
    p.y += p.vy * DT;
    p.z += p.vz * DT;

    // Resolve collisions with cups and world bounds
    resolve_cup_collision(p, source_cup);
    resolve_cup_collision(p, receiver_cup);
    apply_world_box_collision(p);
  }
}

// Run the density pass using the selected mode
void compute_density_pressure() {

  // Run sequential CPU density when selected
  if (simulation_mode == SIM_MODE_CPU_SEQUENTIAL) {
    compute_density_pressure_cpu();
    return;
  }

  // Build the CUDA launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  // Run brute force GPU density when selected
  if (simulation_mode == SIM_MODE_GPU_BRUTE_FORCE) {
    compute_density_pressure_bruteforce_kernel<<<blocks_per_grid,
                                                 threads_per_block>>>(
        fluid_particles, num_fluid_particles, source_compute_boundary_particles,
        num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(),
               "compute_density_pressure_bruteforce_kernel");

  // Run spatial hash GPU density when selected
  } else {
    compute_density_pressure_hash_kernel<<<blocks_per_grid,
                                           threads_per_block>>>(
        fluid_particles, num_fluid_particles, fluid_particle_sorted_index,
        fluid_cell_start, fluid_cell_end, source_compute_boundary_particles,
        source_boundary_sorted_index, source_boundary_cell_start,
        source_boundary_cell_end, receiver_compute_boundary_particles,
        receiver_boundary_sorted_index, receiver_boundary_cell_start,
        receiver_boundary_cell_end);
    cuda_check(cudaDeviceSynchronize(), "compute_density_pressure_hash_kernel");
  }
}

// Run the force pass using the selected mode
void compute_forces() {

  // Run sequential CPU forces when selected
  if (simulation_mode == SIM_MODE_CPU_SEQUENTIAL) {
    compute_forces_cpu();
    return;
  }

  // Build the CUDA launch shape
  int threads_per_block = 256;
  int blocks_per_grid =
      (num_fluid_particles + threads_per_block - 1) / threads_per_block;

  // Run brute force GPU forces when selected
  if (simulation_mode == SIM_MODE_GPU_BRUTE_FORCE) {
    compute_forces_bruteforce_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles, source_compute_boundary_particles,
        num_source_compute_boundary_particles,
        receiver_compute_boundary_particles,
        num_receiver_compute_boundary_particles);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_bruteforce_kernel");

  // Run spatial hash GPU forces when selected
  } else {
    compute_forces_hash_kernel<<<blocks_per_grid, threads_per_block>>>(
        fluid_particles, num_fluid_particles, fluid_particle_sorted_index,
        fluid_cell_start, fluid_cell_end, source_compute_boundary_particles,
        source_boundary_sorted_index, source_boundary_cell_start,
        source_boundary_cell_end, receiver_compute_boundary_particles,
        receiver_boundary_sorted_index, receiver_boundary_cell_start,
        receiver_boundary_cell_end);
    cuda_check(cudaDeviceSynchronize(), "compute_forces_hash_kernel");
  }
}

// Export the current scene to csv
void export_csv(int frame_index) {

  // Rebuild dense render boundary for this exported frame
  rebuild_boundary_particles_for_export();

  // Build the output file name
  std::string frame_string = std::to_string(frame_index);
  frame_string = std::string(4 - frame_string.length(), '0') + frame_string;
  std::string file_name = "/tmp/athanf/frame_" + frame_string + ".csv";

  // Open the output file
  std::ofstream file(file_name);

  // Write the csv header
  file << "x,y,z,rho,p,is_boundary,kind\n";

  // Write all fluid particles
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }

  // Write all dense render boundary particles
  for (int i = 0; i < num_boundary_particles; i++) {
    Particle &particle = boundary_particles[i];
    file << particle.x << "," << particle.y << "," << particle.z << ","
         << particle.rho << "," << particle.p << "," << particle.is_boundary
         << "," << particle.kind << "\n";
  }
}

// Print summary stats for one frame
void print_stats(int frame_index) {

  // Skip stats when there is no fluid
  if (num_fluid_particles == 0) {
    return;
  }

  // Start stat accumulators
  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_speed = 0.0f;
  float max_speed = 0.0f;

  // Accumulate density pressure and speed values
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

  // Compute average stats
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  // Print frame stats
  std::cout << "Frame " << frame_index << " | rho: [" << min_rho << ", "
            << max_rho << "] avg=" << avg_rho << " | p: [" << min_p << ", "
            << max_p << "]"
            << " | speed avg=" << avg_speed << " max=" << max_speed
            << std::endl;
}

// Print density and pressure stats after setup
void print_initial_density_stats() {

  // Skip stats when there is no fluid
  if (num_fluid_particles == 0) {
    return;
  }

  // Build the first grids for spatial hash mode
  if (simulation_mode == SIM_MODE_GPU_SPATIAL_HASH) {
    build_spatial_grid();
    build_source_boundary_grid();
    build_receiver_boundary_grid();
  }

  // Run the first density pass
  compute_density_pressure();

  // Start stat accumulators
  float min_rho = fluid_particles[0].rho;
  float max_rho = fluid_particles[0].rho;
  float min_p = fluid_particles[0].p;
  float max_p = fluid_particles[0].p;
  float sum_rho = 0.0f;
  float sum_p = 0.0f;

  // Accumulate density and pressure values
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle = fluid_particles[i];
    min_rho = std::min(min_rho, particle.rho);
    max_rho = std::max(max_rho, particle.rho);
    min_p = std::min(min_p, particle.p);
    max_p = std::max(max_p, particle.p);
    sum_rho += particle.rho;
    sum_p += particle.p;
  }

  // Compute average stats
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);

  // Print initial density stats
  std::cout << "Initial density stats | rho: [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p: [" << min_p << ", " << max_p
            << "] avg = " << avg_p << std::endl;
}

// Print detailed fluid only stats with a label
void print_fluid_only_stats(const std::string &label) {

  // Skip stats when there is no fluid
  if (num_fluid_particles == 0) {
    std::cout << label << " | no fluid particles" << std::endl;
    return;
  }

  // Start detailed stat accumulators
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

  // Accumulate detailed fluid stats
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

  // Compute detailed averages
  float avg_rho = sum_rho / static_cast<float>(num_fluid_particles);
  float avg_p = sum_p / static_cast<float>(num_fluid_particles);
  float avg_speed = sum_speed / static_cast<float>(num_fluid_particles);

  // Print detailed fluid stats
  std::cout << label << " | fluid_count = " << num_fluid_particles
            << " | rho = [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho << " | p = [" << min_p << ", " << max_p
            << "] avg = " << avg_p << " | speed = [" << min_speed << ", "
            << max_speed << "] avg = " << avg_speed
            << " | rho_floor_count = " << rho_floor_count
            << " | zero_pressure_count = " << zero_pressure_count << std::endl;
}

// Read the simulation mode from the command line
static SimulationMode parse_mode(int argc, char **argv) {

  // Default to spatial hash mode
  if (argc < 3) {
    return SIM_MODE_GPU_SPATIAL_HASH;
  }

  // Read the mode string
  std::string mode_string = argv[2];

  // Select CPU sequential mode
  if ((mode_string == "sequential") || (mode_string == "cpu")) {
    return SIM_MODE_CPU_SEQUENTIAL;
  }

  // Select GPU brute force mode
  if ((mode_string == "brute") || (mode_string == "gpu_brute") ||
      (mode_string == "all_pairs")) {
    return SIM_MODE_GPU_BRUTE_FORCE;
  }

  // Select GPU spatial hash mode
  if ((mode_string == "hash") || (mode_string == "spatial_hash") ||
      (mode_string == "grid")) {
    return SIM_MODE_GPU_SPATIAL_HASH;
  }

  // Print valid mode options
  std::cerr << "Unknown mode " << mode_string << std::endl;
  std::cerr << "Use one of sequential brute hash" << std::endl;
  std::exit(1);
}

// Run the full simulation
int main(int argc, char **argv) {
  using clock_type = std::chrono::high_resolution_clock;

  // Seed random jitter for repeatable initialization
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

  // Default to rebuilding every substep
  int rebuild_freq = 1; // Default to rebuilding every 1 substep

  // Read the spatial hash rebuild frequency
  if (argc > 3) {
    rebuild_freq = std::stoi(argv[3]);
  }

  // Allocate unified memory for fluid particles
  cuda_check(cudaMallocManaged(&fluid_particles,
                               MAX_FLUID_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(fluid_particles)");

  // Allocate unified memory for dense render boundary particles
  cuda_check(cudaMallocManaged(&boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(boundary_particles)");

  // Allocate unified memory for source compute boundary particles
  cuda_check(cudaMallocManaged(&source_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(source_compute_boundary_particles)");

  // Allocate unified memory for receiver compute boundary particles
  cuda_check(cudaMallocManaged(&receiver_compute_boundary_particles,
                               MAX_BOUNDARY_PARTICLES * sizeof(Particle)),
             "cudaMallocManaged(receiver_compute_boundary_particles)");

  // Allocate unified memory for fluid cell ids
  cuda_check(cudaMallocManaged(&fluid_particle_cell_id,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(fluid_particle_cell_id)");

  // Allocate unified memory for sorted fluid indices
  cuda_check(cudaMallocManaged(&fluid_particle_sorted_index,
                               MAX_FLUID_PARTICLES * sizeof(int)),
             "cudaMallocManaged(fluid_particle_sorted_index)");

  // Allocate unified memory for fluid cell starts
  cuda_check(
      cudaMallocManaged(&fluid_cell_start, HASH_GRID_CELL_COUNT * sizeof(int)),
      "cudaMallocManaged(fluid_cell_start)");

  // Allocate unified memory for fluid cell ends
  cuda_check(
      cudaMallocManaged(&fluid_cell_end, HASH_GRID_CELL_COUNT * sizeof(int)),
      "cudaMallocManaged(fluid_cell_end)");

  // Allocate unified memory for source boundary cell ids
  cuda_check(cudaMallocManaged(&source_boundary_cell_id,
                               MAX_BOUNDARY_PARTICLES * sizeof(int)),
             "cudaMallocManaged(source_boundary_cell_id)");

  // Allocate unified memory for sorted source boundary indices
  cuda_check(cudaMallocManaged(&source_boundary_sorted_index,
                               MAX_BOUNDARY_PARTICLES * sizeof(int)),
             "cudaMallocManaged(source_boundary_sorted_index)");

  // Allocate unified memory for source boundary cell starts
  cuda_check(cudaMallocManaged(&source_boundary_cell_start,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(source_boundary_cell_start)");

  // Allocate unified memory for source boundary cell ends
  cuda_check(cudaMallocManaged(&source_boundary_cell_end,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(source_boundary_cell_end)");

  // Allocate unified memory for receiver boundary cell ids
  cuda_check(cudaMallocManaged(&receiver_boundary_cell_id,
                               MAX_BOUNDARY_PARTICLES * sizeof(int)),
             "cudaMallocManaged(receiver_boundary_cell_id)");

  // Allocate unified memory for sorted receiver boundary indices
  cuda_check(cudaMallocManaged(&receiver_boundary_sorted_index,
                               MAX_BOUNDARY_PARTICLES * sizeof(int)),
             "cudaMallocManaged(receiver_boundary_sorted_index)");

  // Allocate unified memory for receiver boundary cell starts
  cuda_check(cudaMallocManaged(&receiver_boundary_cell_start,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(receiver_boundary_cell_start)");

  // Allocate unified memory for receiver boundary cell ends
  cuda_check(cudaMallocManaged(&receiver_boundary_cell_end,
                               HASH_GRID_CELL_COUNT * sizeof(int)),
             "cudaMallocManaged(receiver_boundary_cell_end)");

  // Print run setup information
  std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg
            << std::endl;
  std::cout << "Simulation mode = " << mode_to_string(simulation_mode)
            << std::endl;

  // Build the initial scene
  initialize_scene(target_tilt_deg);

  // Print setup and first pass debug stats
  print_initial_density_stats();
  print_fluid_only_stats("Frame 0 fluid stats");
  print_source_cup_setup_stats();

  // Print particle counts
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

  // Export the starting frame
  export_csv(0);

  // Start timing totals
  double total_compute_ms = 0.0;
  double min_frame_ms = std::numeric_limits<double>::max();
  double max_frame_ms = 0.0;

  // Run the frame loop
  for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {

    // Reset penetration debug stats for this frame
    reset_penetration_stats();

    // Start frame timer
    auto frame_start = clock_type::now();

    // Run all substeps inside this frame
    for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {

      // Compute the fractional frame value for this substep
      float substep_frame_index = static_cast<float>(frame_index - 1) +
                                  static_cast<float>(step_index + 1) /
                                      static_cast<float>(SUBSTEPS_PER_FRAME);

      // Update cup position and angular velocity
      update_scene_for_frame(substep_frame_index, target_tilt_deg);

      // Rebuild only the moving source compute boundary
      rebuild_source_compute_boundary_particles();

      // Rebuild selected grids at the chosen substep frequency
      if (step_index % rebuild_freq == 0) {
        if (simulation_mode == SIM_MODE_GPU_SPATIAL_HASH) {

          // Rebuild the moving fluid grid
          build_spatial_grid();

          // Rebuild the moving source boundary grid
          build_source_boundary_grid();
        }
      }

      // Run density pressure force and integration
      compute_density_pressure();
      compute_forces();
      integrate_fluid_particles();
    }

    // Stop frame timer
    auto frame_end = clock_type::now();

    // Compute frame runtime in milliseconds
    double frame_ms =
        std::chrono::duration<double, std::milli>(frame_end - frame_start)
            .count();

    // Update timing totals
    total_compute_ms += frame_ms;
    min_frame_ms = std::min(min_frame_ms, frame_ms);
    max_frame_ms = std::max(max_frame_ms, frame_ms);

    // Print this frame timing
    std::cout << std::fixed << std::setprecision(3) << "FrameTiming "
              << frame_index << " " << frame_ms << std::endl;

    // Export this frame when needed
    if (frame_index % EXPORT_EVERY == 0) {
      export_csv(frame_index / EXPORT_EVERY);
    }

    // Print debug stats every 20 frames
    if (frame_index % 20 == 0) {
      print_stats(frame_index);
      print_fluid_only_stats("Frame " + std::to_string(frame_index) +
                             " fluid stats");
      print_penetration_stats(frame_index);
    }
  }

  // Compute average frame runtime
  double avg_frame_ms = total_compute_ms / static_cast<double>(FRAME_COUNT);

  // Print timing summary
  std::cout << std::fixed << std::setprecision(3)
            << "TimingSummary mode=" << mode_to_string(simulation_mode)
            << " total_ms=" << total_compute_ms << " avg_ms=" << avg_frame_ms
            << " min_ms=" << min_frame_ms << " max_ms=" << max_frame_ms
            << std::endl;

  // Print completion message
  std::cout << "Done" << std::endl;

  // Free receiver boundary grid arrays
  cuda_check(cudaFree(receiver_boundary_cell_end),
             "cudaFree(receiver_boundary_cell_end)");
  cuda_check(cudaFree(receiver_boundary_cell_start),
             "cudaFree(receiver_boundary_cell_start)");
  cuda_check(cudaFree(receiver_boundary_sorted_index),
             "cudaFree(receiver_boundary_sorted_index)");
  cuda_check(cudaFree(receiver_boundary_cell_id),
             "cudaFree(receiver_boundary_cell_id)");

  // Free source boundary grid arrays
  cuda_check(cudaFree(source_boundary_cell_end),
             "cudaFree(source_boundary_cell_end)");
  cuda_check(cudaFree(source_boundary_cell_start),
             "cudaFree(source_boundary_cell_start)");
  cuda_check(cudaFree(source_boundary_sorted_index),
             "cudaFree(source_boundary_sorted_index)");
  cuda_check(cudaFree(source_boundary_cell_id),
             "cudaFree(source_boundary_cell_id)");

  // Free fluid grid arrays
  cuda_check(cudaFree(fluid_cell_end), "cudaFree(fluid_cell_end)");
  cuda_check(cudaFree(fluid_cell_start), "cudaFree(fluid_cell_start)");
  cuda_check(cudaFree(fluid_particle_sorted_index),
             "cudaFree(fluid_particle_sorted_index)");
  cuda_check(cudaFree(fluid_particle_cell_id),
             "cudaFree(fluid_particle_cell_id)");

  // Free particle arrays
  cuda_check(cudaFree(receiver_compute_boundary_particles),
             "cudaFree(receiver_compute_boundary_particles)");
  cuda_check(cudaFree(source_compute_boundary_particles),
             "cudaFree(source_compute_boundary_particles)");
  cuda_check(cudaFree(boundary_particles), "cudaFree(boundary_particles)");
  cuda_check(cudaFree(fluid_particles), "cudaFree(fluid_particles)");

  return 0;
}