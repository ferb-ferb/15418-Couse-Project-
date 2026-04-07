#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

/* SIMULATION CONSTANTS  */

// Simulation Constants
const int FRAME_COUNT = 1000;     // Total number of frames to simulate
const float STARTING_HEIGHT = 0.1f;  // Initial height of the particle block
const int NUM_PARTICLES_1D = 11;     // 10x10x10 cube
const int SUBSTEPS_PER_FRAME = 5;
const float SPACING_MULTIPLIER = 0.75f; // Multiplier to reduce initial spacing and increase density
const float POS_JITTER_CONST = 0.3f; 
const float VEL_JITTER_CONST = 0.05f;

// Physics Constants
const float H = 0.13f;                // Smoothing radius
const float GRAVITY = -9.81f;        // Gravity acceleration
const float DT = 0.0003f;            // Time step
const float PI = 3.1415926535f;
const float MASS = 0.1f;           // Mass of a single particle
const float REST_DENS = 104.0f;     // Rest density of water
const float GAS_CONST = 100.0f;     // Pressure stiffness
const float VISCOSITY = 25.0f;      // Viscosity coefficient
const float EPS = 1e-6f;             // Small epsilon for safe division
const float VELOCITY_DAMPING = 0.9995f; // Damping factor for velocities

// Calculated Constants
const float POLY6 = 315.0f / (64.0f * PI * std::pow(H, 9)); // density estimation
const float SPIKY_GRAD = -45.0f / (PI * std::pow(H, 6)); // pressure gradient
const float VISC_LAP = 45.0f / (PI * std::pow(H, 6)); // viscosity Laplacian

// Boundary box Constants 
const float BOX_X_MIN = 0.0f;
const float BOX_X_MAX = 1.0f;
const float BOX_Y_MIN = 0.0f;
const float BOX_Y_MAX = 2.0f;
const float BOX_Z_MIN = 0.0f;
const float BOX_Z_MAX = 1.0f;
const float BOUNDARY_DAMPING = -0.1f;

// Particle struct
struct Particle {
  float x, y, z;
  float vx, vy, vz;
  float fx, fy, fz;
  float rho, p;
  bool is_boundary;
};

// Global data structures 
std::vector<Particle> particles;

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
  float spacing = H * SPACING_MULTIPLIER; 

  // Create a cube of particles
  for (int i = 0; i < NUM_PARTICLES_1D; i++) {
    for (int j = 0; j < NUM_PARTICLES_1D; j++) {
      for (int k = 0; k < NUM_PARTICLES_1D; k++) {
        Particle p;

        // Add a tiny random offset to break perfect grid symmetry
        float jitter_scale = POS_JITTER_CONST * spacing;
        float jitter_x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
        float jitter_y = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
        float jitter_z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

        // Set particle position
        p.x = i * spacing + jitter_x;
        p.y = j * spacing + STARTING_HEIGHT + jitter_y;
        p.z = k * spacing + jitter_z;

        // Set initial velocity
        p.vx = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
        p.vy = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
        p.vz = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;

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
      float dx = p_i.x - p_j.x;
      float dy = p_i.y - p_j.y;
      float dz = p_i.z - p_j.z;

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
      float pressure_term = -MASS * (p_i.p / std::max(p_i.rho * p_i.rho, EPS) + p_j.p / std::max(p_j.rho * p_j.rho, EPS));

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
    p.vx *= VELOCITY_DAMPING;
    p.vy *= VELOCITY_DAMPING;
    p.vz *= VELOCITY_DAMPING;

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
/* *** DEBUG FUNCTIONS *** */

void printStats(int frame) {
  float min_rho = particles[0].rho;
  float max_rho = particles[0].rho;
  float min_p = particles[0].p;
  float max_p = particles[0].p;

  float sum_rho = 0.0f;
  float sum_speed = 0.0f;
  float max_speed = 0.0f;

  for (const auto &p : particles) {
    min_rho = std::min(min_rho, p.rho);
    max_rho = std::max(max_rho, p.rho);
    min_p = std::min(min_p, p.p);
    max_p = std::max(max_p, p.p);

    sum_rho += p.rho;

    float speed = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    sum_speed += speed;
    max_speed = std::max(max_speed, speed);
  }

  float avg_rho = sum_rho / particles.size();
  float avg_speed = sum_speed / particles.size();

  std::cout << "Frame " << frame
            << " | rho: [" << min_rho << ", " << max_rho << "] avg=" << avg_rho
            << " | p: [" << min_p << ", " << max_p << "]"
            << " | speed avg=" << avg_speed << " max=" << max_speed
            << std::endl;
}

// Prints the initial density and pressure statistics after initialization
void printInitialDensityStats() {
  // Compute density and pressure once for the initial particle layout
  computeDensityPressure();

  // Initialize min, max, and sum values
  float min_rho = particles[0].rho;
  float max_rho = particles[0].rho;
  float sum_rho = 0.0f;

  float min_p = particles[0].p;
  float max_p = particles[0].p;
  float sum_p = 0.0f;

  // Scan all particles
  for (const auto &p : particles) {
    min_rho = std::min(min_rho, p.rho);
    max_rho = std::max(max_rho, p.rho);
    sum_rho += p.rho;

    min_p = std::min(min_p, p.p);
    max_p = std::max(max_p, p.p);
    sum_p += p.p;
  }

  // Compute averages
  float avg_rho = sum_rho / particles.size();
  float avg_p = sum_p / particles.size();

  // Print summary
  std::cout << "Initial density stats | rho: [" << min_rho << ", " << max_rho
            << "] avg = " << avg_rho
            << " | p: [" << min_p << ", " << max_p
            << "] avg = " << avg_p << std::endl;
}

// Runs the baseline SPH simulation and exports frames
int main() {

  // Seed RNG for jitter
  // srand(time(nullptr)); //random each time 
  srand(0); //deterministic for debugging
  
  std::cout << "Generating particle block..." << std::endl;

  // Create initial particle configuration
  initParticles();
  printInitialDensityStats();

  std::cout << "Successfully created " << particles.size() << " particles." << std::endl;

  // Set the number of frames to simulate
  int total_frames = FRAME_COUNT;

  std::cout << "Starting simulation for " << total_frames << " frames..." << std::endl;

  // Export the initial frame before any updates
  exportCSV(0);

  // Run the simulation loop
  for (int frame = 1; frame <= total_frames; frame++) {

    // Run multiple internal physics steps before exporting this frame
    for (int step = 0; step < SUBSTEPS_PER_FRAME; step++) {
      computeDensityPressure();
      computeForces();
      integrate();
    }

    // Save the updated particle state once per exported frame
    exportCSV(frame);

    // Print progress every 20 exported frames
    if (frame % 20 == 0) {
      std::cout << "Calculated frame " << frame << std::endl;
      printStats(frame);
    }
  }

  // Print completion message
  std::cout << "Done! Open the CSV sequence in ParaView and hit play." << std::endl;

  return 0;
}