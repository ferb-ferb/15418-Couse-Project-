#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "container.h"
#include "sph_baseline.h"

Particle *fluid_particles;
int num_fluid_particles;

// Compute density and pressure on the GPU
// One CUDA thread owns one fluid particle
// Each thread scans all fluid particles and all boundary particles
__global__ void compute_density_pressure_kernel(Particle *fluid_particles,
                                                int num_fluid_particles,
                                                Particle *boundary_particles,
                                                int num_boundary_particles) {
  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles)
    return;

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

  // Sum density from all boundary particles
  for (int j = 0; j < num_boundary_particles; j++) {

    // Build the fluid to boundary offset
    float dx = boundary_particles[j].x - p_i_x;
    float dy = boundary_particles[j].y - p_i_y;
    float dz = boundary_particles[j].z - p_i_z;

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

// Run the CPU density pass
// This is kept as the original sequential version for comparison
void compute_density_pressure() {

  // Loop over every fluid particle
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];

    // Reset density before accumulation
    particle_i.rho = 0.0f;

    // Sum density from all fluid particles
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      // Build the fluid to fluid offset
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;

      // Build squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Skip particles outside the density radius
      if (r2 >= H_DENS * H_DENS) {
        continue;
      }

      // Add the density kernel weight
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      particle_i.rho += MASS * POLY6 * weight;
    }

    // Clamp density away from zero
    particle_i.rho = std::max(particle_i.rho, REST_DENS * 0.1f);

    // Convert density into pressure
    particle_i.p = std::max(GAS_CONST * (particle_i.rho - REST_DENS), 0.0f);
  }
}

// Compute forces on the GPU
// One CUDA thread owns one fluid particle
// Each thread scans all fluid particles and all boundary particles
__global__ void compute_forces_kernel(Particle *fluid_particles,
                                      int num_fluid_particles,
                                      Particle *boundary_particles,
                                      int num_boundary_particles) {
  // Map this CUDA thread to one fluid particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip threads outside the fluid particle range
  if (i >= num_fluid_particles)
    return;

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
    if (i == j)
      continue;

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
          -MASS * (p_i_p / std::fmaxf(p_i_rho * p_i_rho, EPS) +
                   fluid_particles[j].p / std::fmaxf(fluid_particles[j].rho *
                                                         fluid_particles[j].rho,
                                                     EPS));
      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      // Accumulate viscosity force
      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / std::fmaxf(fluid_particles[j].rho, EPS);
      viscosity_fx += VISCOSITY * MASS * (fluid_particles[j].vx - p_i_vx) *
                      inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (fluid_particles[j].vy - p_i_vy) *
                      inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (fluid_particles[j].vz - p_i_vz) *
                      inv_rho_j * visc_coeff;
    }
  }

  // Sum pressure force from all boundary particles
  for (int j = 0; j < num_boundary_particles; j++) {

    // Build the fluid to boundary offset
    float dx = p_i_x - boundary_particles[j].x;
    float dy = p_i_y - boundary_particles[j].y;
    float dz = p_i_z - boundary_particles[j].z;

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
      float rhob = boundary_particles[j].rho;

      // Accumulate boundary pressure force
      float p_term = -MASS * (p_i_p / std::fmaxf(p_i_rho * p_i_rho, EPS) +
                              pb / std::fmaxf(rhob * rhob, EPS));

      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      // Skip boundary viscosity so water can slide along walls
    }
  }

  // Store the final force with gravity
  fluid_particles[i].fx = pressure_fx + viscosity_fx;
  fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
  fluid_particles[i].fz = pressure_fz + viscosity_fz;
}

// Run the CPU force pass
// This is kept as the original sequential version for comparison
void compute_forces() {

  // Loop over every fluid particle
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];

    // Reset force accumulators
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
      if (&particle_i == &particle_j) {
        continue;
      }

      // Build the fluid to fluid offset
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;

      // Build squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Skip particles outside the force radius
      if (r2 >= H_FORCE * H_FORCE) {
        continue;
      }

      // Build true distance
      float r = std::sqrt(r2);

      // Skip nearly overlapping particles
      if (r < EPS) {
        continue;
      }

      // Build direction from neighbor to query particle
      float dir_x = dx / r;
      float dir_y = dy / r;
      float dir_z = dz / r;

      // Accumulate pressure force
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;
      float pressure_term =
          -MASS *
          (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
           particle_j.p / std::max(particle_j.rho * particle_j.rho, EPS));
      pressure_fx += pressure_term * grad_coeff * dir_x;
      pressure_fy += pressure_term * grad_coeff * dir_y;
      pressure_fz += pressure_term * grad_coeff * dir_z;

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
  } else if (particle.x > BOX_X_MAX) {

    // Resolve the right world wall
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
  } else if (particle.y > BOX_Y_MAX) {

    // Resolve the ceiling
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
  } else if (particle.z > BOX_Z_MAX) {

    // Resolve the back world wall
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

// Export the current scene to csv
void export_csv(int frame_index) {

  // Rebuild boundary particles for the current cup positions
  rebuild_boundary_particles_for_export();

  // Build the output file name
  std::string frame_string = std::to_string(frame_index);
  frame_string = std::string(4 - frame_string.length(), '0') + frame_string;
  std::string file_name = "/tmp/athanf/frame_" + frame_string + ".csv";
  // std::string file_name = "/tmp/athanf/frame_" + frame_string + ".csv";

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

  // Write all boundary particles
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

// Run the full CUDA baseline simulation
int main(int argc, char **argv) {

  // Seed random jitter for repeatable initialization
  srand(0);

  // Read the target tilt angle
  float target_tilt_deg = 0.0f;

  if (argc >= 2) {
    target_tilt_deg = std::stof(argv[1]);
  }

  // Clamp the target tilt angle
  target_tilt_deg = std::max(0.0f, std::min(180.0f, target_tilt_deg));

  // Allocate unified memory for fluid and boundary arrays
  cudaError_t err1 = cudaMallocManaged(&fluid_particles,
                                       MAX_FLUID_PARTICLES * sizeof(Particle));
  cudaError_t err2 = cudaMallocManaged(
      &boundary_particles, MAX_BOUNDARY_PARTICLES * sizeof(Particle));

  // Check fluid allocation
  if (err1 != cudaSuccess) {
    std::cerr << "Fluid allocation failed: " << cudaGetErrorString(err1)
              << std::endl;
    return -1;
  }

  // Check boundary allocation
  if (err2 != cudaSuccess) {
    std::cerr << "Boundary allocation failed: " << cudaGetErrorString(err2)
              << std::endl;
    return -1;
  }

  // Build the initial scene
  std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg
            << std::endl;
  initialize_scene(target_tilt_deg);

  // Build the CUDA launch shape
  int threadsPerBlock = 256;
  int blocksPerGrid =
      (num_fluid_particles + threadsPerBlock - 1) / threadsPerBlock;

  // Print setup and first pass debug stats
  print_initial_density_stats();
  print_fluid_only_stats("Frame 0 fluid stats");
  print_source_cup_setup_stats();

  // Print initial particle counts
  std::cout << "Fluid particles = " << num_fluid_particles << std::endl;
  std::cout << "Boundary particles = " << num_boundary_particles << std::endl;

  // Export the starting frame
  export_csv(0);

  // Run the frame loop
  for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {

    // Reset penetration debug stats for this frame
    reset_penetration_stats();

    // Run all substeps inside this frame
    for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {

      // Compute the fractional frame value for this substep
      float substep_frame_index = static_cast<float>(frame_index - 1) +
                                  static_cast<float>(step_index + 1) /
                                      static_cast<float>(SUBSTEPS_PER_FRAME);

      // Update cup position and angular velocity
      update_scene_for_frame(substep_frame_index, target_tilt_deg);

      // Rebuild boundary particles for the current cup pose
      rebuild_boundary_particles_for_export();

      // Run density and pressure on the GPU
      compute_density_pressure_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          fluid_particles, num_fluid_particles, boundary_particles,
          num_boundary_particles);
      cudaDeviceSynchronize();

      // Run force computation on the GPU
      compute_forces_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          fluid_particles, num_fluid_particles, boundary_particles,
          num_boundary_particles);
      cudaDeviceSynchronize();

      // Integrate particle motion on the CPU
      integrate_fluid_particles();
    }

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

  // Print completion message
  std::cout << "Done" << std::endl;
  return 0;
}