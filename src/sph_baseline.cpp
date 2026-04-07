#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/* SIMULATION CONSTANTS  */
const int NUM_PARTICLES_1D = 10;     // 10x10x10 cube
const float H = 0.1f;                // Smoothing radius
const float GRAVITY = -9.81f;        // Gravity acceleration
const float DT = 0.001f;            // Time step
const float PI = 3.1415926535f;
const float MASS = 0.125f;           // Mass of a single particle
const float REST_DENS = 1000.0f;     // Rest density of water
const float GAS_CONST = 500.0f;     // Pressure stiffness
const float VISCOSITY = 50.0f;      // Viscosity coefficient
const float EPS = 1e-6f;             // Small epsilon for safe division

// constant for density estimation
const float POLY6 = 315.0f / (64.0f * PI * std::pow(H, 9));

//constant for pressure gradient
const float SPIKY_GRAD = -45.0f / (PI * std::pow(H, 6));

// constant for viscosity Laplacian
const float VISC_LAP = 45.0f / (PI * std::pow(H, 6));

// BOX CONSTANTS
const float BOX_X_MIN = 0.0f;
const float BOX_X_MAX = 1.0f;
const float BOX_Y_MIN = 0.0f;
const float BOX_Y_MAX = 2.0f;
const float BOX_Z_MIN = 0.0f;
const float BOX_Z_MAX = 1.0f;
const float BOUNDARY_DAMPING = -0.5f;

// Particle struct
struct Particle {
  float x, y, z;
  float vx, vy, vz;
  float fx, fy, fz;
  float rho, p;
  bool is_boundary;
};

std::vector<Particle> particles;

/* Initialize Particles */

/*
// JITTER: We add a microscopic random offset to every particle.
// If they are perfectly aligned in a flawless grid, the physics engine
// will divide by zero later when they perfectly collide.
float jitter_x = static_cast<float>(rand()) / RAND_MAX * 0.01f;
float jitter_y = static_cast<float>(rand()) / RAND_MAX * 0.01f;
float jitter_z = static_cast<float>(rand()) / RAND_MAX * 0.01f; */

// Initializes a 3D block of particles above the floor
void initParticles() {

  // Set spacing between particles
  float spacing = H * 0.65f;

  // Create a cube of particles
  for (int i = 0; i < NUM_PARTICLES_1D; i++) {
    for (int j = 0; j < NUM_PARTICLES_1D; j++) {
      for (int k = 0; k < NUM_PARTICLES_1D; k++) {
        Particle p;

        // Set particle position
        p.x = i * spacing;
        p.y = j * spacing + 0.03f;
        p.z = k * spacing;

        // Set initial velocity
        p.vx = 0.0f;
        p.vy = 0.0f;
        p.vz = 0.0f;

        // Set initial force
        p.fx = 0.0f;
        p.fy = 0.0f;
        p.fz = 0.0f;

        // Set initial SPH quantities
        p.rho = REST_DENS;
        p.p = 0.0f;

        // Mark this as a regular fluid particle
        p.is_boundary = false;

        // Add particle to the simulation
        particles.push_back(p);
      }
    }
  }
}

// Computes particle density and pressure using O(N^2) neighbor search
void computeDensityPressure() {
  
  // Loop over all particles
  for (auto &p_i : particles) {
    
    // Reset density for this particle
    p_i.rho = 0.0f;

    // Sum of density contributions from nearby particles
    for (const auto &p_j : particles) {
      
      // Compute displacement from j to i
      float dx = p_j.x - p_i.x;
      float dy = p_j.y - p_i.y;
      float dz = p_j.z - p_i.z;

      // Compute squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Accumulate density if neighbor is inside smoothing radius
      if (r2 < H * H) {
        float h2_minus_r2 = H * H - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        p_i.rho += MASS * POLY6 * weight;
      }
    }

    // Clamp density to avoid division issues
    p_i.rho = std::max(p_i.rho, REST_DENS * 0.1f);

    // Compute pressure using equation of state
    p_i.p = std::max(GAS_CONST * (p_i.rho - REST_DENS), 0.0f);
  }
}

// Computes pressure, viscosity, and gravity forces for all particles
void computeForces() {
  
  // Loop over all particles
  for (auto &p_i : particles) {
    
    // Skip boundary particles if added later
    if (p_i.is_boundary) {
      p_i.fx = 0.0f;
      p_i.fy = 0.0f;
      p_i.fz = 0.0f;
      continue;
    }

    // Reset force accumulators
    float pressure_fx = 0.0f;
    float pressure_fy = 0.0f;
    float pressure_fz = 0.0f;

    float viscosity_fx = 0.0f;
    float viscosity_fy = 0.0f;
    float viscosity_fz = 0.0f;

    // Loop over neighbors
    for (const auto &p_j : particles) {
      
      // Skip yourself
      if (&p_i == &p_j) {
        continue;
      }

      // Compute displacement from j to i
      float dx = p_j.x - p_i.x;
      float dy = p_j.y - p_i.y;
      float dz = p_j.z - p_i.z;

      // Compute squared distance
      float r2 = dx * dx + dy * dy + dz * dz;

      // Ignore particles outside smoothing radius
      if (r2 >= H * H) {
        continue;
      }

      // Compute actual distance
      float r = std::sqrt(r2);

      // Skip nearly identical positions to avoid division by zero
      if (r < EPS) {
        continue;
      }

      // Compute normalized direction
      float dir_x = dx / r;
      float dir_y = dy / r;
      float dir_z = dz / r;

      // Compute pressure force contribution
      float h_minus_r = H - r;
      float grad_coeff = SPIKY_GRAD * h_minus_r * h_minus_r;
      float pressure_term = -MASS * (p_i.p + p_j.p) / (2.0f * std::max(p_j.rho, EPS));

      pressure_fx += pressure_term * grad_coeff * dir_x;
      pressure_fy += pressure_term * grad_coeff * dir_y;
      pressure_fz += pressure_term * grad_coeff * dir_z;

      // Compute viscosity force contribution
      float visc_coeff = VISC_LAP * h_minus_r;
      float inv_rho_j = 1.0f / std::max(p_j.rho, EPS);

      viscosity_fx += VISCOSITY * MASS * (p_j.vx - p_i.vx) * inv_rho_j * visc_coeff;
      viscosity_fy += VISCOSITY * MASS * (p_j.vy - p_i.vy) * inv_rho_j * visc_coeff;
      viscosity_fz += VISCOSITY * MASS * (p_j.vz - p_i.vz) * inv_rho_j * visc_coeff;
    }

    // Combine pressure, viscosity, and gravity into total force
    p_i.fx = pressure_fx + viscosity_fx;
    p_i.fy = pressure_fy + viscosity_fy + p_i.rho * GRAVITY;
    p_i.fz = pressure_fz + viscosity_fz;
  }
}

// Integrates particle motion and keeps particles inside a 3D box
void integrate() {
  // Loop over all particles
  for (auto &p : particles) {
    // Keep boundary particles fixed
    if (p.is_boundary) {
      continue;
    }

    // Convert force to acceleration
    float ax = p.fx / std::max(p.rho, EPS);
    float ay = p.fy / std::max(p.rho, EPS);
    float az = p.fz / std::max(p.rho, EPS);

    // Update velocity using acceleration
    p.vx += ax * DT;
    p.vy += ay * DT;
    p.vz += az * DT;

    // Apply slight damping for stability
    p.vx *= 0.999f;
    p.vy *= 0.999f;
    p.vz *= 0.999f;

    // Update position using velocity
    p.x += p.vx * DT;
    p.y += p.vy * DT;
    p.z += p.vz * DT;

    // Bounce off left wall
    if (p.x < BOX_X_MIN) {
      p.x = BOX_X_MIN;
      p.vx *= BOUNDARY_DAMPING;
    }

    // Bounce off right wall
    if (p.x > BOX_X_MAX) {
      p.x = BOX_X_MAX;
      p.vx *= BOUNDARY_DAMPING;
    }

    // Bounce off floor
    if (p.y < BOX_Y_MIN) {
      p.y = BOX_Y_MIN;
      p.vy *= BOUNDARY_DAMPING;
    }

    // Bounce off ceiling
    if (p.y > BOX_Y_MAX) {
      p.y = BOX_Y_MAX;
      p.vy *= BOUNDARY_DAMPING;
    }

    // Bounce off front wall
    if (p.z < BOX_Z_MIN) {
      p.z = BOX_Z_MIN;
      p.vz *= BOUNDARY_DAMPING;
    }

    // Bounce off back wall
    if (p.z > BOX_Z_MAX) {
      p.z = BOX_Z_MAX;
      p.vz *= BOUNDARY_DAMPING;
    }
  }
}

// Writes particle data to CSV files inside the output folder for ParaView
void exportCSV(int frame) {
  
  // Convert frame number to string and pad with zeros (e.g., "0005")
  std::string frame_str = std::to_string(frame);
  frame_str = std::string(4 - frame_str.length(), '0') + frame_str;

  // Build output filename inside the output folder
  std::string filename = "output/frame_" + frame_str + ".csv";

  // Open output file
  std::ofstream file(filename);

  // Write CSV header
  file << "x,y,z,rho,p,is_boundary\n";

  // Write one row per particle
  for (const auto &p : particles) {
    file << p.x << "," << p.y << "," << p.z << "," << p.rho << "," << p.p
         << "," << p.is_boundary << "\n";
  }

  // Close file
  file.close();
}

// Prints a few useful statistics for debugging
void printStats(int frame) {
  // Track min and max density
  float min_rho = particles[0].rho;
  float max_rho = particles[0].rho;

  // Track min and max pressure
  float min_p = particles[0].p;
  float max_p = particles[0].p;

  // Scan all particles
  for (const auto &p : particles) {
    min_rho = std::min(min_rho, p.rho);
    max_rho = std::max(max_rho, p.rho);
    min_p = std::min(min_p, p.p);
    max_p = std::max(max_p, p.p);
  }

  // Print summary line
  std::cout << "Frame " << frame << " | rho: [" << min_rho << ", " << max_rho
            << "] | p: [" << min_p << ", " << max_p << "]" << std::endl;
}

// Runs the baseline SPH simulation and exports frames
int main() {
  
  std::cout << "Generating particle block..." << std::endl;

  // Create initial particle configuration
  initParticles();

  std::cout << "Successfully created " << particles.size() << " particles." << std::endl;

  // Set the number of frames to simulate
  int total_frames = 200;

  std::cout << "Starting simulation for " << total_frames << " frames..." << std::endl;

  // Export the initial frame before any updates
  exportCSV(0);

  // Run the simulation loop
  for (int frame = 1; frame <= total_frames; frame++) {

    // Compute density and pressure from neighbors
    computeDensityPressure();

    // Compute fluid and gravity forces
    computeForces();

    // Move particles forward by one timestep
    integrate();

    // Save the updated particle state
    exportCSV(frame);

    // Print progress every 20 frames
    if (frame % 20 == 0) {
      std::cout << "Calculated frame " << frame << std::endl;
      printStats(frame);
    }

  }

  // Print completion message
  std::cout << "Done! Open the CSV sequence in ParaView and hit play." << std::endl;

  return 0;
}
