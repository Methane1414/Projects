#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// Constants
const double g = 9.81;            // Gravity (m/s²)
const double slope_angle = 30.0;  // Slope angle (degrees)
const double delt = 0.01;         // Time step (s)
const int total_steps = 1000;     // Simulation steps
const int num_particles = 100;    // Number of particles
const double friction_coeff = 0.1; // Friction coefficient
const double fragmentation_threshold = 5.0; // Kinetic energy threshold

// Structure to represent a particle
struct Particle {
    double x, y;      // Position
    double vx, vy;    // Velocity
    double size;      // Size
};

// Function to initialize particles
void initialize_particles(vector<Particle>& particles) {
    for (int i = 0; i < num_particles; i++) {
        particles[i].x = i * 0.1;
        particles[i].y = 10.0; // Initial height
        particles[i].vx = 0.0;
        particles[i].vy = 0.0;
        particles[i].size = 2.0;
    }
}

// Function to apply forces to a particle
void apply_forces(Particle& p, double dt) {
    // Calculate acceleration along the slope
    double slope_rad = slope_angle * M_PI / 180.0;
    double ax = g * sin(slope_rad) - friction_coeff * p.vx;
    double ay = -g * cos(slope_rad);

    // Update velocity based on forces
    p.vx += ax * dt;
    p.vy += ay * dt;
}

// Function to update particle position
void update_particle(Particle& p, double dt) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
}

// Function to handle collisions between particles
void handle_collisions(vector<Particle>& particles) {
    for (int i = 0; i < num_particles; i++) {
        for (int j = i + 1; j < num_particles; j++) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dist = sqrt(dx * dx + dy * dy);
            double min_dist = (particles[i].size + particles[j].size) / 2.0;

            if (dist < min_dist) {
                // Normalize the collision normal
                double nx = dx / dist;
                double ny = dy / dist;

                // Compute relative velocity
                double vx_relative = particles[j].vx - particles[i].vx;
                double vy_relative = particles[j].vy - particles[i].vy;

                // Compute collision impulse
                double impulse = (vx_relative * nx + vy_relative * ny) * 0.5; // Equal mass

                // Update velocities
                particles[i].vx += impulse * nx;
                particles[i].vy += impulse * ny;
                particles[j].vx -= impulse * nx;
                particles[j].vy -= impulse * ny;

                // Separate overlapping particles
                double overlap = min_dist - dist;
                particles[i].x -= 0.5 * overlap * nx;
                particles[i].y -= 0.5 * overlap * ny;
                particles[j].x += 0.5 * overlap * nx;
                particles[j].y += 0.5 * overlap * ny;
            }
        }
    }
}

// Function to apply fragmentation to particles
void apply_fragmentation(Particle& p) {
    double kinetic_energy = 0.5 * (p.vx * p.vx + p.vy * p.vy);
    if (p.size > 0.1 && kinetic_energy > fragmentation_threshold) {
        p.size *= 0.9; // Reduce by 10%
    }
}

int main() {
    // Vector to store particles
    vector<Particle> particles(num_particles);

    // Initialize particles
    initialize_particles(particles);

    // Open file for data output
    ofstream file("avalanche_data.csv");
    if (!file.is_open()) {
        cerr << "Error opening file." << endl;
        return 1;
    }
    file << "time,x,y,vx,vy,size" << endl;

    // Main simulation loop
    for (int step = 0; step < total_steps; step++) {
        double time = step * delt;

        // Update each particle
        for (auto& p : particles) {
            apply_forces(p, delt);     // Apply gravitational and frictional forces
            update_particle(p, delt);   // Update position
            apply_fragmentation(p);     // Apply fragmentation rule
        }

        // Handle collisions between particles
        handle_collisions(particles);

        // Write data to file
        for (const auto& p : particles) {
            file << time << "," << p.x << "," << p.y << ","
                 << p.vx << "," << p.vy << "," << p.size << endl;
        }
    }

    // Close the file
    file.close();
    cout << "Simulation complete. Data written to avalanche_data.csv" << endl;

    return 0;
}
