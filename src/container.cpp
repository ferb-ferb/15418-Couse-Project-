#include <cmath>
#include <cstdlib>

#include "container.h"

Cup source_cup;
Cup receiver_cup;
std::vector<Particle> boundary_particles;

// Build one cup
Cup make_cup(float center_x, float center_y, float center_z, float width, float height, float depth, float wall_thickness, float tilt_deg) {

    // Fill the cup fields
    Cup cup;
    cup.center_x = center_x;
    cup.center_y = center_y;
    cup.center_z = center_z;
    cup.width = width;
    cup.height = height;
    cup.depth = depth;
    cup.wall_thickness = wall_thickness;
    cup.tilt_deg = tilt_deg;

    // Store the rotation values
    float theta = tilt_deg * PI / 180.0f;
    cup.cos_t = std::cos(theta);
    cup.sin_t = std::sin(theta);

    return cup;
}

// Update one cup rotation
void update_cup_rotation(Cup &cup, float tilt_deg) {

    // Store the new angle
    cup.tilt_deg = tilt_deg;

    // Store the new rotation values
    float theta = tilt_deg * PI / 180.0f;
    cup.cos_t = std::cos(theta);
    cup.sin_t = std::sin(theta);
}

// Convert cup local coordinates to world coordinates
static void cup_local_to_world(const Cup &cup, float local_x, float local_y, float &world_x, float &world_y) {

    // Rotate and shift the local point
    world_x = cup.center_x + local_x * cup.cos_t + local_y * cup.sin_t;
    world_y = cup.center_y - local_x * cup.sin_t + local_y * cup.cos_t;
}

// Convert world coordinates to cup local coordinates
static void world_to_cup_local(const Cup &cup, float world_x, float world_y, float &local_x, float &local_y) {

    // Shift the point into the cup frame
    float dx = world_x - cup.center_x;
    float dy = world_y - cup.center_y;

    // Rotate the point into the cup frame
    local_x = dx * cup.cos_t - dy * cup.sin_t;
    local_y = dx * cup.sin_t + dy * cup.cos_t;
}

// Convert world velocity to cup local velocity
static void world_velocity_to_cup_local(const Cup &cup, float world_vx, float world_vy, float &local_vx, float &local_vy) {

    // Rotate the velocity into the cup frame
    local_vx = world_vx * cup.cos_t - world_vy * cup.sin_t;
    local_vy = world_vx * cup.sin_t + world_vy * cup.cos_t;
}

// Convert cup local velocity to world velocity
static void cup_velocity_to_world(const Cup &cup, float local_vx, float local_vy, float &world_vx, float &world_vy) {

    // Rotate the velocity back into world space
    world_vx = local_vx * cup.cos_t + local_vy * cup.sin_t;
    world_vy = -local_vx * cup.sin_t + local_vy * cup.cos_t;
}

// Add one boundary particle
static void add_boundary_particle(float x, float y, float z, int kind) {

    // Fill the boundary particle
    Particle particle;
    particle.x = x;
    particle.y = y;
    particle.z = z;
    particle.vx = 0.0f;
    particle.vy = 0.0f;
    particle.vz = 0.0f;
    particle.fx = 0.0f;
    particle.fy = 0.0f;
    particle.fz = 0.0f;
    particle.rho = 0.0f;
    particle.p = 0.0f;
    particle.is_boundary = true;
    particle.kind = kind;

    boundary_particles.push_back(particle);
}

// Add one fluid particle
static void add_fluid_particle(float x, float y, float z) {

    // Build the position jitter
    float jitter_scale = POS_JITTER_CONST * INITIAL_PARTICLE_SPACING;
    float jitter_x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
    float jitter_y = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;
    float jitter_z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * jitter_scale;

    // Fill the fluid particle
    Particle particle;
    particle.x = x + jitter_x;
    particle.y = y + jitter_y;
    particle.z = z + jitter_z;
    particle.vx = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
    particle.vy = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
    particle.vz = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * VEL_JITTER_CONST;
    particle.fx = 0.0f;
    particle.fy = 0.0f;
    particle.fz = 0.0f;
    particle.rho = REST_DENS;
    particle.p = 0.0f;
    particle.is_boundary = false;
    particle.kind = 0;

    fluid_particles.push_back(particle);
}

// Add one cup wall surface
static void add_cup_surface(const Cup &cup, float local_x_min, float local_x_max, float local_y_min, float local_y_max, float z_min, float z_max, int kind) {

    // Loop across the cup surface
    for (float local_x = local_x_min; local_x <= local_x_max + 0.5f * CUP_RENDER_SPACING; local_x += CUP_RENDER_SPACING) {
        for (float local_y = local_y_min; local_y <= local_y_max + 0.5f * CUP_RENDER_SPACING; local_y += CUP_RENDER_SPACING) {
            for (float z = z_min; z <= z_max + 0.5f * CUP_RENDER_SPACING; z += CUP_RENDER_SPACING) {

                // Skip the second loop direction for lines
                if ( (std::abs(local_x_max - local_x_min) < EPS) && (std::abs(local_y_max - local_y_min) < EPS) ) {
                    continue;
                }

                // Convert the point to world space
                float world_x;
                float world_y;
                cup_local_to_world(cup, local_x, local_y, world_x, world_y);

                // Add the render particle
                add_boundary_particle(world_x, world_y, z, kind);
            }
        }
    }
}

// Add all render particles for one cup
static void add_cup_render_particles(const Cup &cup, int kind) {

    // Build the depth limits
    float z_min = cup.center_z - 0.5f * cup.depth;
    float z_max = cup.center_z + 0.5f * cup.depth;

    // Add the bottom surface
    add_cup_surface(cup, -0.5f * cup.width, 0.5f * cup.width, 0.0f, 0.0f, z_min, z_max, kind);

    // Add the left wall surface
    add_cup_surface(cup, -0.5f * cup.width, -0.5f * cup.width, 0.0f, cup.height, z_min, z_max, kind);

    // Add the right wall surface
    add_cup_surface(cup, 0.5f * cup.width, 0.5f * cup.width, 0.0f, cup.height, z_min, z_max, kind);

    // Add the front wall surface
    for (float local_x = -0.5f * cup.width; local_x <= 0.5f * cup.width + 0.5f * CUP_RENDER_SPACING; local_x += CUP_RENDER_SPACING) {
        for (float local_y = 0.0f; local_y <= cup.height + 0.5f * CUP_RENDER_SPACING; local_y += CUP_RENDER_SPACING) {

            // Convert the point to world space
            float world_x;
            float world_y;
            cup_local_to_world(cup, local_x, local_y, world_x, world_y);

            // Add the front render particle
            add_boundary_particle(world_x, world_y, z_min, kind);
        }
    }

    // Add the back wall surface
    for (float local_x = -0.5f * cup.width; local_x <= 0.5f * cup.width + 0.5f * CUP_RENDER_SPACING; local_x += CUP_RENDER_SPACING) {
        for (float local_y = 0.0f; local_y <= cup.height + 0.5f * CUP_RENDER_SPACING; local_y += CUP_RENDER_SPACING) {

            // Convert the point to world space
            float world_x;
            float world_y;
            cup_local_to_world(cup, local_x, local_y, world_x, world_y);

            // Add the back render particle
            add_boundary_particle(world_x, world_y, z_max, kind);
        }
    }
}

// Add fluid inside the source cup
static void add_fluid_in_source_cup() {

    // Build the inner cup limits
    float local_x_min = -0.5f * source_cup.width + source_cup.wall_thickness + INITIAL_PARTICLE_SPACING;
    float local_x_max = 0.5f * source_cup.width - source_cup.wall_thickness - INITIAL_PARTICLE_SPACING;
    float local_y_min = source_cup.wall_thickness + INITIAL_PARTICLE_SPACING;
    float local_y_max = source_cup.wall_thickness + SOURCE_FILL_RATIO * (source_cup.height - source_cup.wall_thickness - INITIAL_PARTICLE_SPACING);
    float z_min = source_cup.center_z - 0.5f * source_cup.depth + source_cup.wall_thickness + INITIAL_PARTICLE_SPACING;
    float z_max = source_cup.center_z + 0.5f * source_cup.depth - source_cup.wall_thickness - INITIAL_PARTICLE_SPACING;

    // Loop through the cup interior
    for (float local_x = local_x_min; local_x <= local_x_max + 0.5f * INITIAL_PARTICLE_SPACING; local_x += INITIAL_PARTICLE_SPACING) {
        for (float local_y = local_y_min; local_y <= local_y_max + 0.5f * INITIAL_PARTICLE_SPACING; local_y += INITIAL_PARTICLE_SPACING) {
            for (float z = z_min; z <= z_max + 0.5f * INITIAL_PARTICLE_SPACING; z += INITIAL_PARTICLE_SPACING) {

                // Convert the point to world space
                float world_x;
                float world_y;
                cup_local_to_world(source_cup, local_x, local_y, world_x, world_y);

                // Add the fluid particle
                add_fluid_particle(world_x, world_y, z);
            }
        }
    }
}

// Build the current cup render particles
static void rebuild_boundary_particles() {

    // Clear the old boundary particles
    boundary_particles.clear();

    // Add the source cup particles
    add_cup_render_particles(source_cup, 1);

    // Add the receiver cup particles
    add_cup_render_particles(receiver_cup, 2);
}

// Get the source tilt for one frame
static float get_source_tilt_for_frame(int frame_index, float target_tilt_deg) {

    // Hold the cup upright first
    if (frame_index < SETTLE_FRAMES) {
        return 0.0f;
    }
    else if (frame_index < (SETTLE_FRAMES + TILT_FRAMES)) {

        // Blend from upright to the target angle
        float alpha = static_cast<float>(frame_index - SETTLE_FRAMES) / static_cast<float>(TILT_FRAMES);
        return alpha * target_tilt_deg;
    }
    else {

        // Hold the target angle
        return target_tilt_deg;
    }
}

// Build the whole scene
void initialize_scene(float target_tilt_deg) {

    // Clear the old particles
    fluid_particles.clear();
    boundary_particles.clear();

    // Reserve some room for the particles
    fluid_particles.reserve(4000);
    boundary_particles.reserve(40000);

    // Build the cups upright
    source_cup = make_cup(SOURCE_CUP_CENTER_X, SOURCE_CUP_CENTER_Y, SOURCE_CUP_CENTER_Z, SOURCE_CUP_WIDTH, SOURCE_CUP_HEIGHT, SOURCE_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);
    receiver_cup = make_cup(RECEIVER_CUP_CENTER_X, RECEIVER_CUP_CENTER_Y, RECEIVER_CUP_CENTER_Z, RECEIVER_CUP_WIDTH, RECEIVER_CUP_HEIGHT, RECEIVER_CUP_DEPTH, CUP_WALL_THICKNESS, 0.0f);

    // Add the starting fluid to the source cup
    add_fluid_in_source_cup();

    // Build the first boundary particles
    update_scene_for_frame(0, target_tilt_deg);
}

// Update the cups for one frame
void update_scene_for_frame(int frame_index, float target_tilt_deg) {

    // Get the current source tilt
    float current_tilt_deg = get_source_tilt_for_frame(frame_index, target_tilt_deg);

    // Update the cup angles
    update_cup_rotation(source_cup, current_tilt_deg);
    update_cup_rotation(receiver_cup, 0.0f);

    // Rebuild the render particles
    rebuild_boundary_particles();
}

// Resolve the cup bottom collision
static void resolve_bottom_collision(Particle &particle, float &local_x, float &local_y, float &local_vx, float &local_vy, float inner_half_width, float inner_bottom, float inner_depth_min, float inner_depth_max) {

    // Skip particles outside the cup opening
    if ( (local_x < -inner_half_width) || (local_x > inner_half_width) ) {
        return;
    }

    // Skip particles outside the cup depth
    if ( (particle.z < inner_depth_min) || (particle.z > inner_depth_max) ) {
        return;
    }

    // Push the particle above the bottom wall
    if (local_y < inner_bottom) {
        local_y = inner_bottom + WALL_EPS;

        if (local_vy < 0.0f) {
            local_vy = -local_vy * WALL_RESTITUTION;
        }

        local_vx *= WALL_TANGENTIAL_DAMPING;
        particle.vz *= WALL_TANGENTIAL_DAMPING;
    }
}

// Resolve the cup side collision
static void resolve_side_collision(Particle &particle, float &local_x, float &local_y, float &local_vx, float &local_vy, float inner_half_width, float inner_bottom, float inner_depth_min, float inner_depth_max) {

    // Skip particles below the cup bottom or above the rim
    if ( (local_y < inner_bottom) || (local_y > source_cup.height) ) {
        return;
    }

    // Skip particles outside the cup depth
    if ( (particle.z < inner_depth_min) || (particle.z > inner_depth_max) ) {
        return;
    }

    // Push the particle off the left wall
    if (local_x < -inner_half_width) {
        local_x = -inner_half_width + WALL_EPS;

        if (local_vx < 0.0f) {
            local_vx = -local_vx * WALL_RESTITUTION;
        }

        local_vy *= WALL_TANGENTIAL_DAMPING;
        particle.vz *= WALL_TANGENTIAL_DAMPING;
    }

    // Push the particle off the right wall
    if (local_x > inner_half_width) {
        local_x = inner_half_width - WALL_EPS;

        if (local_vx > 0.0f) {
            local_vx = -local_vx * WALL_RESTITUTION;
        }

        local_vy *= WALL_TANGENTIAL_DAMPING;
        particle.vz *= WALL_TANGENTIAL_DAMPING;
    }
}

// Resolve the cup depth collision
static void resolve_depth_collision(Particle &particle, float local_x, float local_y, float &local_vx, float &local_vy, float inner_half_width, float inner_bottom, float inner_depth_min, float inner_depth_max) {

    // Skip particles outside the cup cavity
    if ( (local_x < -inner_half_width) || (local_x > inner_half_width) ) {
        return;
    }

    if ( (local_y < inner_bottom) || (local_y > source_cup.height) ) {
        return;
    }

    // Push the particle off the front wall
    if (particle.z < inner_depth_min) {
        particle.z = inner_depth_min + WALL_EPS;

        if (particle.vz < 0.0f) {
            particle.vz = -particle.vz * WALL_RESTITUTION;
        }

        local_vx *= WALL_TANGENTIAL_DAMPING;
        local_vy *= WALL_TANGENTIAL_DAMPING;
    }

    // Push the particle off the back wall
    if (particle.z > inner_depth_max) {
        particle.z = inner_depth_max - WALL_EPS;

        if (particle.vz > 0.0f) {
            particle.vz = -particle.vz * WALL_RESTITUTION;
        }

        local_vx *= WALL_TANGENTIAL_DAMPING;
        local_vy *= WALL_TANGENTIAL_DAMPING;
    }
}

// Push one fluid particle out of one cup wall
void resolve_cup_collision(Particle &particle, const Cup &cup) {

    // Skip boundary particles
    if (particle.is_boundary) {
        return;
    }

    // Move the particle into the cup frame
    float local_x;
    float local_y;
    world_to_cup_local(cup, particle.x, particle.y, local_x, local_y);

    // Move the velocity into the cup frame
    float local_vx;
    float local_vy;
    world_velocity_to_cup_local(cup, particle.vx, particle.vy, local_vx, local_vy);

    // Build the inner cup limits
    float inner_half_width = 0.5f * cup.width - cup.wall_thickness;
    float inner_bottom = cup.wall_thickness;
    float inner_depth_min = cup.center_z - 0.5f * cup.depth + cup.wall_thickness;
    float inner_depth_max = cup.center_z + 0.5f * cup.depth - cup.wall_thickness;

    // Resolve the three cup directions
    resolve_bottom_collision(particle, local_x, local_y, local_vx, local_vy, inner_half_width, inner_bottom, inner_depth_min, inner_depth_max);
    resolve_side_collision(particle, local_x, local_y, local_vx, local_vy, inner_half_width, inner_bottom, inner_depth_min, inner_depth_max);
    resolve_depth_collision(particle, local_x, local_y, local_vx, local_vy, inner_half_width, inner_bottom, inner_depth_min, inner_depth_max);

    // Move the particle back to world space
    cup_local_to_world(cup, local_x, local_y, particle.x, particle.y);

    // Move the velocity back to world space
    cup_velocity_to_world(cup, local_vx, local_vy, particle.vx, particle.vy);
}
