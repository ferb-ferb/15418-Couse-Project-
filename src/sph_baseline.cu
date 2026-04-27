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
/* =========================================================
 * DENSITY PRESSURE KERNEL
 * =========================================================*/
__global__ void compute_density_pressure_kernel(Particle *fluid_particles,
                                                int num_fluid_particles,
                                                Particle *boundary_particles,
                                                int num_boundary_particles) {
  // 1. Get the thread ID. This replaces the OUTER 'for' loop.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. Safety check: don't do work if the thread ID is higher than our particle
  // count
  if (i >= num_fluid_particles)
    return;

  // Cache the current particle's position to avoid repeated global memory reads
  float p_i_x = fluid_particles[i].x;
  float p_i_y = fluid_particles[i].y;
  float p_i_z = fluid_particles[i].z;

  float density = 0.0f;

  // --- FLUID-FLUID INTERACTIONS (INNER LOOP) ---
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

  // --- BOUNDARY INTERACTIONS (INNER LOOP) ---
  // for (int j = 0; j < num_boundary_particles; j++) {
  //   float dx = boundary_particles[j].x - p_i_x;
  //   float dy = boundary_particles[j].y - p_i_y;
  //   float dz = boundary_particles[j].z - p_i_z;
  //   float r2 = dx * dx + dy * dy + dz * dz;
  //
  //   if (r2 < H_DENS * H_DENS) {
  //     float h2_minus_r2 = H_DENS * H_DENS - r2;
  //     float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
  //     density +=
  //         MASS * POLY6 * weight; // Assuming boundary particles have same
  //         mass
  //   }
  // }

  // Write the final density and pressure back to global memory
  // (Check your baseline to ensure this pressure formula matches yours!)
  fluid_particles[i].rho = fmaxf(density, REST_DENS * 0.1f);
  fluid_particles[i].p = fmaxf(GAS_CONST * (density - REST_DENS), 0.0f);
}

// Run the density pass
void compute_density_pressure() {

  // Loop over the fluid particles
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];

    // Reset the density
    particle_i.rho = 0.0f;

    // Sum the nearby density values
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      // Build the particle offset
      float dx = particle_j.x - particle_i.x;
      float dy = particle_j.y - particle_i.y;
      float dz = particle_j.z - particle_i.z;

      // Build the squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Skip particles outside the density radius
      if (r2 >= H_DENS * H_DENS) {
        continue;
      }

      // Add the density weight
      float h2_minus_r2 = H_DENS * H_DENS - r2;
      float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
      particle_i.rho += MASS * POLY6 * weight;
    }
    // for(int j = 0; j < num_fluid_particles; j++) {
    //   Particle &particle = fluid_particles[j];
    //
    //   // Build the particle offset
    //   float dx = particle_j.x - particle_i.x;
    //   float dy = particle_j.y - particle_i.y;
    //   float dz = particle_j.z - particle_i.z;
    //
    //   // Build the squared distance
    //   float r2 = dx * dx + dy * dy + dz * dz;
    //
    //   // Skip particles outside the density radius
    //   if (r2 >= H_DENS * H_DENS) {
    //     continue;
    //   }
    //
    //   // Add the density weight
    //   float h2_minus_r2 = H_DENS * H_DENS - r2;
    //   float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
    //   particle_i.rho += MASS * POLY6 * weight;
    // }

    // Clamp the density
    particle_i.rho = std::max(particle_i.rho, REST_DENS * 0.1f);

    // Build the pressure
    particle_i.p = std::max(GAS_CONST * (particle_i.rho - REST_DENS), 0.0f);
  }
}
/* ==========================================================================
 * FOCES KERNEL
 * ==========================================================================*/

__global__ void compute_forces_kernel(Particle *fluid_particles,
                                      int num_fluid_particles,
                                      Particle *boundary_particles,
                                      int num_boundary_particles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_fluid_particles)
    return;

  // Cache properties
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

  // --- FLUID-FLUID FORCES ---
  for (int j = 0; j < num_fluid_particles; j++) {
    if (i == j)
      continue; // Don't interact with yourself
    float dx = fluid_particles[j].x - p_i_x;
    float dy = fluid_particles[j].y - p_i_y;
    float dz = fluid_particles[j].z - p_i_z;
    float r2 = dx * dx + dy * dy + dz * dz;

    float r = sqrtf(r2); // Use CUDA's sqrtf
    // NORMAL CHECKS R AGAINST EPS
    if (r2 < H_FORCE * H_FORCE && r >= EPS) {
      float h_minus_r = H_FORCE - r;

      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;
      // Pressure Force
      // float p_term = -MASS * (p_i_p + fluid_particles[j].p) /
      //                (2.0f * fluid_particles[j].rho) * grad_coeff;
      float p_term =
          -MASS * (p_i_p / std::fmaxf(p_i_rho * p_i_rho, EPS) +
                   fluid_particles[j].p / std::fmaxf(fluid_particles[j].rho *
                                                         fluid_particles[j].rho,
                                                     EPS));
      pressure_fx += p_term * grad_coeff * dx / r;
      pressure_fy += p_term * grad_coeff * dy / r;
      pressure_fz += p_term * grad_coeff * dz / r;

      // Viscosity Force
      // float v_term =
      //     VISCOSITY * MASS / fluid_particles[j].rho * VISC_LAP * h_minus_r;
      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / std::fmaxf(fluid_particles[j].rho, EPS);
      viscosity_fx += VISCOSITY * MASS * (fluid_particles[j].vx - p_i_vx) *
                      inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (fluid_particles[j].vy - p_i_vy) *
                      inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (fluid_particles[j].vz - p_i_vz) *
                      inv_rho_j * visc_coeff;

      // fx += p_term * dx / r + v_term * (fluid_particles[j].vx - p_i_vx);
      // fy += p_term * dy / r + v_term * (fluid_particles[j].vy - p_i_vy);
      fluid_particles[i].fx = pressure_fx + viscosity_fx;
      fluid_particles[i].fy = pressure_fy + viscosity_fy + p_i_rho * GRAVITY;
      fluid_particles[i].fz =
          pressure_fz + viscosity_fz; // fz += p_term * dz / r + v_term *
                                      // (fluid_particles[j].vz - p_i_vz);
    }
  }

  // --- BOUNDARY FORCES ---
  // (Paste your inner loop for boundary forces here, following the same
  // pattern)

  // Add Gravity
  // fy += GRAVITY * MASS;
  // fy += GRAVITY * p_i_rho;

  // Write back to global memory
}

// Run the force pass
void compute_forces() {

  // Loop over the fluid particles
  for (int i = 0; i < num_fluid_particles; i++) {
    Particle &particle_i = fluid_particles[i];

    // Reset the force sums
    float pressure_fx = 0.0f;
    float pressure_fy = 0.0f;
    float pressure_fz = 0.0f;
    float viscosity_fx = 0.0f;
    float viscosity_fy = 0.0f;
    float viscosity_fz = 0.0f;

    // Sum the nearby forces
    for (int j = 0; j < num_fluid_particles; j++) {
      Particle &particle_j = fluid_particles[j];

      // Skip the same particle
      if (&particle_i == &particle_j) {
        continue;
      }

      // Build the particle offset
      float dx = particle_i.x - particle_j.x;
      float dy = particle_i.y - particle_j.y;
      float dz = particle_i.z - particle_j.z;

      // Build the squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Skip particles outside the force radius
      if (r2 >= H_FORCE * H_FORCE) {
        continue;
      }

      // Build the true distance
      float r = std::sqrt(r2);

      // Skip very small distances
      if (r < EPS) {
        continue;
      }

      // Build the direction
      float dir_x = dx / r;
      float dir_y = dy / r;
      float dir_z = dz / r;

      // Build the pressure term
      float h_minus_r = H_FORCE - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;
      float pressure_term =
          -MASS *
          (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) +
           particle_j.p / std::max(particle_j.rho * particle_j.rho, EPS));
      pressure_fx += pressure_term * grad_coeff * dir_x;
      pressure_fy += pressure_term * grad_coeff * dir_y;
      pressure_fz += pressure_term * grad_coeff * dir_z;

      // Build the viscosity term
      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / std::max(particle_j.rho, EPS);
      viscosity_fx += VISCOSITY * MASS * (particle_j.vx - particle_i.vx) *
                      inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (particle_j.vy - particle_i.vy) *
                      inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (particle_j.vz - particle_i.vz) *
                      inv_rho_j * visc_coeff;
    }

    // Store the final force
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

    particle.vy *= WALL_TANGENTIAL_DAMPING;
    particle.vz *= WALL_TANGENTIAL_DAMPING;
  } else if (particle.x > BOX_X_MAX) {
    particle.x = BOX_X_MAX - WALL_EPS;

    if (particle.vx > 0.0f) {
      particle.vx = -particle.vx * WALL_RESTITUTION;
    }

    particle.vy *= WALL_TANGENTIAL_DAMPING;
    particle.vz *= WALL_TANGENTIAL_DAMPING;
  }

  // Resolve the floor and ceiling
  if (particle.y < BOX_Y_MIN) {
    particle.y = BOX_Y_MIN + WALL_EPS;

    if (particle.vy < 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }

    particle.vx *= WALL_TANGENTIAL_DAMPING;
    particle.vz *= WALL_TANGENTIAL_DAMPING;
  } else if (particle.y > BOX_Y_MAX) {
    particle.y = BOX_Y_MAX - WALL_EPS;

    if (particle.vy > 0.0f) {
      particle.vy = -particle.vy * WALL_RESTITUTION;
    }

    particle.vx *= WALL_TANGENTIAL_DAMPING;
    particle.vz *= WALL_TANGENTIAL_DAMPING;
  }

  // Resolve the front and back walls
  if (particle.z < BOX_Z_MIN) {
    particle.z = BOX_Z_MIN + WALL_EPS;

    if (particle.vz < 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    particle.vx *= WALL_TANGENTIAL_DAMPING;
    particle.vy *= WALL_TANGENTIAL_DAMPING;
  } else if (particle.z > BOX_Z_MAX) {
    particle.z = BOX_Z_MAX - WALL_EPS;

    if (particle.vz > 0.0f) {
      particle.vz = -particle.vz * WALL_RESTITUTION;
    }

    particle.vx *= WALL_TANGENTIAL_DAMPING;
    particle.vy *= WALL_TANGENTIAL_DAMPING;
  }
}

void integrate_fluid_particles() {
  for (int i = 0; i < num_fluid_particles; i++) {
    // THE '&' HERE IS THE MOST IMPORTANT CHARACTER IN THIS FILE!
    // Without it, you are modifying a copy, not the actual global array!
    Particle &p = fluid_particles[i];

    // 1. Calculate Acceleration (a = F / density)
    float ax = p.fx / std::fmaxf(p.rho, EPS);
    float ay = p.fy / std::fmaxf(p.rho, EPS);
    float az = p.fz / std::fmaxf(p.rho, EPS);

    // 2. IDIOT-PROOF GRAVITY: Just add it directly here so we KNOW it works

    // 3. Update Velocities
    p.vx += ax * DT;
    p.vy += ay * DT;
    p.vz += az * DT;

    // Apply Damping
    p.vx *= VELOCITY_DAMPING;
    p.vy *= VELOCITY_DAMPING;
    p.vz *= VELOCITY_DAMPING;

    // 4. Update Positions
    p.x += p.vx * DT;
    p.y += p.vy * DT;
    p.z += p.vz * DT;

    // 5. Boundary collisions (This is what was bulldozing the water!)
    resolve_cup_collision(p, source_cup);
    resolve_cup_collision(p, receiver_cup);
    apply_world_box_collision(p);
  }
}
// // Move the fluid particles
// void integrate_fluid_particles() {
//
//   // Loop over the fluid particles
//   for (int i = 0; i < num_fluid_particles; i++) {
//     Particle &particle = fluid_particles[i];
//
//     // Build the acceleration
//     float ax = particle.fx / std::max(particle.rho, EPS);
//     float ay = particle.fy / std::max(particle.rho, EPS);
//     float az = particle.fz / std::max(particle.rho, EPS);
//
//     // Update the velocity
//     particle.vx += ax * DT;
//     particle.vy += ay * DT;
//     particle.vz += az * DT;
//
//     // Dampen the velocity
//     particle.vx *= VELOCITY_DAMPING;
//     particle.vy *= VELOCITY_DAMPING;
//     particle.vz *= VELOCITY_DAMPING;
//
//     // Update the position
//     particle.x += particle.vx * DT;
//     particle.y += particle.vy * DT;
//     particle.z += particle.vz * DT;
//
//     // Resolve the cup collisions
//     resolve_cup_collision(particle, source_cup);
//     resolve_cup_collision(particle, receiver_cup);
//
//     // Resolve the world box collisions
//     apply_world_box_collision(particle);
//
//     // Clamp extreme particle speeds
//     float speed =
//         std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy +
//                   particle.vz * particle.vz);
//     const float MAX_SPEED = 2000000.0f;
//
//     if (speed > MAX_SPEED) {
//       float scale = MAX_SPEED / speed;
//       particle.vx *= scale;
//       particle.vy *= scale;
//       particle.vz *= scale;
//     }
//   }
// }

// Export the scene to csv
void export_csv(int frame_index) {

  // Build the current boundary particles
  rebuild_boundary_particles_for_export();

  // Build the output file name
  std::string frame_string = std::to_string(frame_index);
  frame_string = std::string(4 - frame_string.length(), '0') + frame_string;
  std::string file_name = "/tmp/athanf/frame_" + frame_string + ".csv";
  // std::string file_name = "/tmp/athanf/frame_" + frame_string + ".csv";

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
  // Seed the random generator
  srand(0);

  // Read the target tilt angle
  float target_tilt_deg = 0.0f;

  if (argc >= 2) {
    target_tilt_deg = std::stof(argv[1]);
  }
  // Clamp the target tilt angle
  target_tilt_deg = std::max(0.0f, std::min(180.0f, target_tilt_deg));

  cudaError_t err1 = cudaMallocManaged(&fluid_particles,
                                       MAX_FLUID_PARTICLES * sizeof(Particle));
  cudaError_t err2 = cudaMallocManaged(
      &boundary_particles, MAX_BOUNDARY_PARTICLES * sizeof(Particle));

  if (err1 != cudaSuccess) {
    std::cerr << "Fluid allocation failed: " << cudaGetErrorString(err1)
              << std::endl;
    return -1;
  }

  if (err2 != cudaSuccess) {
    std::cerr << "Boundary allocation failed: " << cudaGetErrorString(err2)
              << std::endl;
    return -1;
  }

  // Build the initial scene
  std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg
            << std::endl;
  initialize_scene(target_tilt_deg);

  int threadsPerBlock = 256;
  int blocksPerGrid =
      (num_fluid_particles + threadsPerBlock - 1) / threadsPerBlock;
  // Print the initial stats
  print_initial_density_stats();
  print_fluid_only_stats("Frame 0 fluid stats");
  print_source_cup_setup_stats();

  // Print the particle counts
  std::cout << "Fluid particles = " << num_fluid_particles << std::endl;
  std::cout << "Boundary particles = " << num_boundary_particles << std::endl;

  // Export the first frame
  export_csv(0);

  // Run the frame loop
  for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {

    // Reset the penetration stats
    reset_penetration_stats();

    // Run the substeps
    for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {

      // Update the scene for this substep
      float substep_frame_index = static_cast<float>(frame_index - 1) +
                                  static_cast<float>(step_index + 1) /
                                      static_cast<float>(SUBSTEPS_PER_FRAME);
      update_scene_for_frame(substep_frame_index, target_tilt_deg);

      // Run the fluid step
      // compute_density_pressure();
      compute_density_pressure_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          fluid_particles, num_fluid_particles, boundary_particles,
          num_boundary_particles);
      cudaDeviceSynchronize();
      // compute_forces();
      compute_forces_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          fluid_particles, num_fluid_particles, boundary_particles,
          num_boundary_particles);
      cudaDeviceSynchronize(); // MUST WAIT FOR GPU
      integrate_fluid_particles();
    }

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

  std::cout << "Done" << std::endl;
  return 0;
}
