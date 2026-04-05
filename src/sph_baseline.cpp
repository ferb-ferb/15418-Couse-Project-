#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

/* Parameters */
const int NUM_PARTICLES_1D = 10; // 10x10x10 cube
const float H = 0.1f;            // Scale
const float GRAVITY = -9.81f;
const float DT = 0.005f; // Time step
const float PI = 3.1415926535f;
const float MASS = 0.125f;       // Mass of a single water droplet
const float REST_DENS = 1000.0f; // Natural density of water
const float GAS_CONST = 2000.0f; // Stiffness of the fluid
int *iter = new int(0);
// The mathematical "weight" of our smoothing radius
const float POLY6 = 315.0f / (64.0f * PI * std::pow(H, 9));

/* Define a Particle */
struct Particle {
  float x, y, z, vy, vx, vz;
  float rho, p; /* density, pressure */
  bool is_boundary;
};

std::vector<Particle> particles;

/* Initialize Particles */
void initParticles() {
  float spacing = H * 0.5f; /* Spawn particles with half a unit spacing */

  for (int i = 0; i < NUM_PARTICLES_1D; i++) {
    for (int j = 0; j < NUM_PARTICLES_1D; j++) {
      for (int k = 0; k < NUM_PARTICLES_1D; k++) {
        Particle p;

        /*
        // JITTER: We add a microscopic random offset to every particle.
        // If they are perfectly aligned in a flawless grid, the physics engine
        // will divide by zero later when they perfectly collide.
        float jitter_x = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        float jitter_y = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        float jitter_z = static_cast<float>(rand()) / RAND_MAX * 0.01f; */

        p.x = i * spacing /*+ jitter_x*/;
        p.y = j * spacing + 1.0f /*+ jitter_y*/; // Spawn 1.0 units up in
                                                 // the air
        p.z = k * spacing /*+ jitter_z*/;
        p.vx = 0;
        p.vy = 0;
        p.vz = 0;
        p.is_boundary = false;

        particles.push_back(p);
      }
    }
  }
}
void computeDensityPressure(int *iter) {
  for (auto &p_i : particles) {
    p_i.rho = 0.0f;

    /* O(N^2) neighbor search for now */
    for (auto &p_j : particles) {
      float dx = p_j.x - p_i.x;
      float dy = p_j.y - p_i.y;
      float dz = p_j.z - p_i.z;

      /* Distance */
      float r2 = dx * dx + dy * dy + dz * dz;

      /* Check if you are in my smoothing radius
       * sqrts are hard to compute I guess
       */
      if (r2 < H * H) {
        // Apply the Poly6 math
        float h2_minus_r2 = H * H - r2;
        float weight = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;
        p_i.rho += MASS * POLY6 * weight;
      }
    }

    /* Now calculate pressure using the Equation of State
     * std::max ensures we never get negative "sticky" pressure
     */
    p_i.p = std::max(GAS_CONST * (p_i.rho - REST_DENS), 0.0f);
    if (*iter == 400)
      printf("particles[455].rho: %f", p_i.rho);
    (*iter)++;
  }
}

computeForces() {}

void integrate() {
  for (auto &p : particles) {
    /* Boundary Particles Remain in Place */
    if (p.is_boundary)
      continue;

    /* vy = vy + a*t
     * a is just gravity for now
     */
    p.vy += GRAVITY * DT;

    /* Move particles forward one timestep */
    p.x += p.vx * DT;
    p.y += p.vy * DT;
    p.z += p.vz * DT;

    // 4. Temporary invisible floor just to catch them for this test
    if (p.y < 0.0f) {
      p.y = 0.0f;
      p.vy *= -0.5f; // Bounce with some energy loss
    }
  }
}
/* Writing CSV for use in paraview */
void exportCSV(int frame) {
  // Convert frame number to string and pad with zeros (e.g., "0005")
  std::string frame_str = std::to_string(frame);
  frame_str = std::string(4 - frame_str.length(), '0') + frame_str;
  std::string filename = "frame_" + frame_str + ".csv";

  std::ofstream file(filename);

  // Added is_boundary to the header so you can color-code later
  file << "x,y,z,is_boundary\n";

  for (const auto &p : particles) {
    file << p.x << "," << p.y << "," << p.z << "," << p.is_boundary << "\n";
  }
  file.close();
}

int main() {
  std::cout << "Generating particle block..." << std::endl;

  initParticles();

  std::cout << "Successfully created " << particles.size() << " particles."
            << std::endl;

  int total_frames = 200; // Total number of frames to simulate

  std::cout << "Starting simulation for " << total_frames << " frames..."
            << std::endl;

  for (int frame = 0; frame < total_frames; frame++) {
    exportCSV(frame); // Save the current positions
    computeDensityPressure(iter);
    integrate(); // Move the particles forward in time

    // Print progress every 20 frames
    if (frame % 20 == 0) {
      std::cout << "Calculated frame " << frame << std::endl;
    }
  }

  std::cout << "Done! Open the file sequence in ParaView and hit play."
            << std::endl;
  return 0;
}
