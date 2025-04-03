#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm> // For std::max

using namespace std;

// Constants
const double nu = 1.5e-5;  // Kinematic viscosity (m²/s)
const double U_inf = 10.0; // Free stream velocity (m/s)
const double rho = 1.225;  // Air density (kg/m³)
const double L = 1.0;      // Plate length (m)
const int N = 100;         // Number of grid points
const double dx = L / (N - 1); // Grid spacing
const double dt = 0.0001;  // Reduced time step for stability
const int time_steps = 10000; // More steps for better convergence

// Turbulence model parameters
const double C_mu = 0.09;
const double sigma_k = 1.0;
const double sigma_e = 1.3;
const double C1e = 1.44;
const double C2e = 1.92;
const double k_min = 1e-6;  // Prevent negative k
const double eps_min = 1e-6; // Prevent division by zero in epsilon
const double k_init = 0.1;   // Small initial k
const double eps_init = 0.01; // Small but nonzero epsilon

int main() {
    vector<double> u(N, 0.0);      // Velocity profile
    vector<double> k(N, k_init);   // Turbulence kinetic energy
    vector<double> eps(N, eps_init); // Turbulence dissipation rate

    // Smooth velocity initialization
    for (int i = 0; i < N; i++) {
        u[i] = U_inf * (double(i) / (N - 1));
    }

    // Time-marching loop
    for (int t = 0; t < time_steps; t++) {
        vector<double> u_new = u;
        vector<double> k_new = k;
        vector<double> eps_new = eps;

        for (int i = 1; i < N - 1; i++) {
            // Clamp k and eps to prevent instability
            k[i] = max(k[i], k_min);
            eps[i] = max(eps[i], eps_min);

            // Compute turbulent viscosity
            double nu_t = C_mu * pow(k[i], 2) / eps[i];
            nu_t = max(nu_t, 1e-6); // Lower bound for stability

            // Adjust time step dynamically for stability
            double dt_local = min(dt, 0.1 * dx * dx / (nu + nu_t));

            // Compute second derivatives (diffusion terms)
            double d2u_dx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx * dx);
            double dk_dx2 = (k[i + 1] - 2 * k[i] + k[i - 1]) / (dx * dx);
            double de_dx2 = (eps[i + 1] - 2 * eps[i] + eps[i - 1]) / (dx * dx);

            // Momentum equation
            u_new[i] = u[i] + dt_local * (nu_t * d2u_dx2);

            // Turbulence kinetic energy (k) equation
            k_new[i] = k[i] + dt_local * (nu_t * dk_dx2 - eps[i]);

            // Turbulence dissipation (ε) equation
            eps_new[i] = eps[i] + dt_local * (
                C1e * (eps[i] / max(k[i], k_min)) * nu_t * dk_dx2
                - C2e * eps[i] * eps[i] / max(k[i], k_min)
                + nu_t * de_dx2
            );

            // Ensure values don't go negative
            k_new[i] = max(k_new[i], k_min);
            eps_new[i] = max(eps_new[i], eps_min);
        }

        // Apply boundary conditions
        u_new[0] = 0.0;      // No-slip at the plate
        u_new[N - 1] = U_inf; // Free stream velocity
        k_new[0] = k_min;
        eps_new[0] = eps_min;

        u = u_new;
        k = k_new;
        eps = eps_new;

        // Debug print every 1000 steps
        if (t % 1000 == 0) {
            cout << "Step: " << t << " | u[50]: " << u[50] << " | k[50]: " << k[50] << " | eps[50]: " << eps[50] << endl;
        }
    }

    // ✅ Write final data to CSV only once at the end
    ofstream file("turbulent.csv");
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    file << "x,u,k,epsilon\n";
    for (int i = 0; i < N; i++) {
        file << i * dx << "," << u[i] << "," << k[i] << "," << eps[i] << "\n";
    }

    file.close();
    cout << "Simulation complete. Results saved in turbulent.csv" << endl;
    return 0;
}
