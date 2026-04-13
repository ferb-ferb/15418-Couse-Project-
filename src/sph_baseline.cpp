#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "container.h"
#include "sph_baseline.h"

std::vector<Particle> fluid_particles;

// Run the density pass
void compute_density_pressure() {

    // Loop over the fluid particles
    for (Particle &particle_i : fluid_particles) {

        // Reset the density
        particle_i.rho = 0.0f;

        // Sum the nearby density values
        for (const Particle &particle_j : fluid_particles) {

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

        // Clamp the density
        particle_i.rho = std::max(particle_i.rho, REST_DENS * 0.1f);

        // Build the pressure
        particle_i.p = std::max(GAS_CONST * (particle_i.rho - REST_DENS), 0.0f);
    }
}

// Run the force pass
void compute_forces() {

    // Loop over the fluid particles
    for (Particle &particle_i : fluid_particles) {

        // Reset the force sums
        float pressure_fx = 0.0f;
        float pressure_fy = 0.0f;
        float pressure_fz = 0.0f;
        float viscosity_fx = 0.0f;
        float viscosity_fy = 0.0f;
        float viscosity_fz = 0.0f;

        // Sum the nearby forces
        for (const Particle &particle_j : fluid_particles) {

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
            float pressure_term = -MASS * (particle_i.p / std::max(particle_i.rho * particle_i.rho, EPS) + particle_j.p / std::max(particle_j.rho * particle_j.rho, EPS));
            pressure_fx += pressure_term * grad_coeff * dir_x;
            pressure_fy += pressure_term * grad_coeff * dir_y;
            pressure_fz += pressure_term * grad_coeff * dir_z;

            // Build the viscosity term
            float visc_coeff = VISC_LAP * h_minus_r;
            float inv_rho_j = 1.0f / std::max(particle_j.rho, EPS);
            viscosity_fx += VISCOSITY * MASS * (particle_j.vx - particle_i.vx) * inv_rho_j * visc_coeff;
            viscosity_fy += VISCOSITY * MASS * (particle_j.vy - particle_i.vy) * inv_rho_j * visc_coeff;
            viscosity_fz += VISCOSITY * MASS * (particle_j.vz - particle_i.vz) * inv_rho_j * visc_coeff;
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
    }
    else if (particle.x > BOX_X_MAX) {
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
    }
    else if (particle.y > BOX_Y_MAX) {
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
    }
    else if (particle.z > BOX_Z_MAX) {
        particle.z = BOX_Z_MAX - WALL_EPS;

        if (particle.vz > 0.0f) {
            particle.vz = -particle.vz * WALL_RESTITUTION;
        }

        particle.vx *= WALL_TANGENTIAL_DAMPING;
        particle.vy *= WALL_TANGENTIAL_DAMPING;
    }
}

// Move the fluid particles
void integrate_fluid_particles() {

    // Loop over the fluid particles
    for (Particle &particle : fluid_particles) {

        // Build the acceleration
        float ax = particle.fx / std::max(particle.rho, EPS);
        float ay = particle.fy / std::max(particle.rho, EPS);
        float az = particle.fz / std::max(particle.rho, EPS);

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

        // Resolve the world box collisions
        apply_world_box_collision(particle);

        // Clamp extreme particle speeds
        float speed = std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);
        const float MAX_SPEED = 2.0f;

        if (speed > MAX_SPEED) {
            float scale = MAX_SPEED / speed;
            particle.vx *= scale;
            particle.vy *= scale;
            particle.vz *= scale;
        }
    }
}

// Export the scene to csv
void export_csv(int frame_index) {

    // Build the current boundary particles
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
    for (const Particle &particle : fluid_particles) {
        file << particle.x << "," << particle.y << "," << particle.z << "," << particle.rho << "," << particle.p << "," << particle.is_boundary << "," << particle.kind << "\n";
    }

    // Write the boundary particles
    for (const Particle &particle : boundary_particles) {
        file << particle.x << "," << particle.y << "," << particle.z << "," << particle.rho << "," << particle.p << "," << particle.is_boundary << "," << particle.kind << "\n";
    }
}

// Print the frame stats
void print_stats(int frame_index) {

    // Skip the stats when there is no fluid
    if (fluid_particles.empty()) {
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
    for (const Particle &particle : fluid_particles) {
        min_rho = std::min(min_rho, particle.rho);
        max_rho = std::max(max_rho, particle.rho);
        min_p = std::min(min_p, particle.p);
        max_p = std::max(max_p, particle.p);
        sum_rho += particle.rho;

        float speed = std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);
        sum_speed += speed;
        max_speed = std::max(max_speed, speed);
    }

    // Build the averages
    float avg_rho = sum_rho / static_cast<float>(fluid_particles.size());
    float avg_speed = sum_speed / static_cast<float>(fluid_particles.size());

    // Print the values
    std::cout << "Frame " << frame_index
              << " | rho: [" << min_rho << ", " << max_rho << "] avg=" << avg_rho
              << " | p: [" << min_p << ", " << max_p << "]"
              << " | speed avg=" << avg_speed << " max=" << max_speed
              << std::endl;
}

// Print the initial stats
void print_initial_density_stats() {

    // Skip the stats when there is no fluid
    if (fluid_particles.empty()) {
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
    for (const Particle &particle : fluid_particles) {
        min_rho = std::min(min_rho, particle.rho);
        max_rho = std::max(max_rho, particle.rho);
        min_p = std::min(min_p, particle.p);
        max_p = std::max(max_p, particle.p);
        sum_rho += particle.rho;
        sum_p += particle.p;
    }

    // Build the averages
    float avg_rho = sum_rho / static_cast<float>(fluid_particles.size());
    float avg_p = sum_p / static_cast<float>(fluid_particles.size());

    // Print the values
    std::cout << "Initial density stats | rho: [" << min_rho << ", " << max_rho << "] avg = " << avg_rho
              << " | p: [" << min_p << ", " << max_p << "] avg = " << avg_p
              << std::endl;
}

// Print fluid only stats
void print_fluid_only_stats(const std::string &label) {

    // Skip the stats when there is no fluid
    if (fluid_particles.empty()) {
        std::cout << label << " | no fluid particles" << std::endl;
        return;
    }

    // Start the stat values
    float min_rho = fluid_particles[0].rho;
    float max_rho = fluid_particles[0].rho;
    float min_p = fluid_particles[0].p;
    float max_p = fluid_particles[0].p;
    float min_speed = std::sqrt(fluid_particles[0].vx * fluid_particles[0].vx + fluid_particles[0].vy * fluid_particles[0].vy + fluid_particles[0].vz * fluid_particles[0].vz);
    float max_speed = min_speed;
    float sum_rho = 0.0f;
    float sum_p = 0.0f;
    float sum_speed = 0.0f;
    int rho_floor_count = 0;
    int zero_pressure_count = 0;

    // Accumulate the stats
    for (const Particle &particle : fluid_particles) {
        float speed = std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);

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
    float avg_rho = sum_rho / static_cast<float>(fluid_particles.size());
    float avg_p = sum_p / static_cast<float>(fluid_particles.size());
    float avg_speed = sum_speed / static_cast<float>(fluid_particles.size());

    // Print the values
    std::cout << label
              << " | fluid_count = " << fluid_particles.size()
              << " | rho = [" << min_rho << ", " << max_rho << "] avg = " << avg_rho
              << " | p = [" << min_p << ", " << max_p << "] avg = " << avg_p
              << " | speed = [" << min_speed << ", " << max_speed << "] avg = " << avg_speed
              << " | rho_floor_count = " << rho_floor_count
              << " | zero_pressure_count = " << zero_pressure_count
              << std::endl;
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

    // Build the initial scene
    std::cout << "Generate the cup scene with target tilt = " << target_tilt_deg << std::endl;
    initialize_scene(target_tilt_deg);

    // Print the initial stats
    print_initial_density_stats();
    print_fluid_only_stats("Frame 0 fluid stats");
    print_source_cup_setup_stats();

    // Print the particle counts
    std::cout << "Fluid particles = " << fluid_particles.size() << std::endl;
    std::cout << "Boundary particles = " << boundary_particles.size() << std::endl;

    // Export the first frame
    export_csv(0);

    // Run the frame loop
    for (int frame_index = 1; frame_index <= FRAME_COUNT; frame_index++) {

        // Reset the penetration stats
        reset_penetration_stats();

        // Run the substeps
        for (int step_index = 0; step_index < SUBSTEPS_PER_FRAME; step_index++) {

            // Update the scene for this substep
            float substep_frame_index = static_cast<float>(frame_index - 1) + static_cast<float>(step_index + 1) / static_cast<float>(SUBSTEPS_PER_FRAME);
            update_scene_for_frame(substep_frame_index, target_tilt_deg);

            // Run the fluid step
            compute_density_pressure();
            compute_forces();
            integrate_fluid_particles();
        }

        // Export this frame when needed
        if (frame_index % EXPORT_EVERY == 0) {
            export_csv(frame_index / EXPORT_EVERY);
        }

        // Print the stats sometimes
        if (frame_index % 20 == 0) {
            print_stats(frame_index);
            print_fluid_only_stats("Frame " + std::to_string(frame_index) + " fluid stats");
            print_penetration_stats(frame_index);
        }
    }

    std::cout << "Done" << std::endl;
    return 0;
}