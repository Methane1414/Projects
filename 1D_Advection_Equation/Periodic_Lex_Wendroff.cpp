//LAX-WENDROFF SCHEME FOR SMOOTH INITIAL CONDITION
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;

// Linear spaced grid generator
void linspace(double start, double end, int N, double* array) {
    double step = (end - start) / (N - 1);
    for (int i = 0; i < N; i++) {
        array[i] = start + i * step;
    }
}

// Initial condition
double initialCondition(double x, double L) {
    return sin(2.0 * M_PI * x / L);
}

// Analytical solution: shift by a*t, periodic BCs
double analyticalSolution(double x, double t, double a, double L) {
    return sin(2.0 * M_PI *((x - a * t)));
}

// Lax-Wendroff scheme with periodic boundary conditions
void runLaxWendroff(double a, double L, double T, int M, int N, double** u, double* x, double* t) {
    // For periodic boundary conditions, we need M-1 intervals, so dx = L/(M-1)
    // But we treat the domain as having M unique points with periodic wrapping
    double dx = L / M; 
    double dt = T / (N - 1);
    double R = a * dt / dx;
    double R_sq = R * R;

    if (R > 1.0) {
        cout << "CFL condition violated: R = " << R << endl;
        exit(1);
    }

    // Setup periodic grid: x[i] = i * dx, i = 0, 1, ..., M-1
    for (int i = 0; i < M; i++) {
        x[i] = i * dx;
    }
    linspace(0, T, N, t);

    // Initial condition
    for (int i = 0; i < M; i++) {
        u[i][0] = initialCondition(x[i], L);
    }

    // Time stepping using Lax-Wendroff scheme
    for (int j = 0; j < N - 1; j++) {
        // Create temporary array for new time step
        double* u_new = new double[M];
        
        // Apply Lax-Wendroff scheme for all points (including boundaries with periodicity)
        for (int i = 0; i < M; i++) {
            // Handle periodic boundaries
            int i_minus = (i - 1 + M) % M;
            int i_plus = (i + 1) % M;
            
            // Lax-Wendroff scheme:
            u_new[i] = u[i][j] 
                     - 0.5 * R * (u[i_plus][j] - u[i_minus][j])
                     + 0.5 * R_sq * (u[i_plus][j] - 2.0 * u[i][j] + u[i_minus][j]);
        }
        
        // Copy new values to main array
        for (int i = 0; i < M; i++) {
            u[i][j + 1] = u_new[i];
        }
        
        delete[] u_new;
    }
}

// Compute L2 error norm at final time
double computeL2Analytical(double* x, double** u, double a, double T, double L, int M, int N) {
    double dx = x[1] - x[0];
    double error_sq = 0.0;
    for (int i = 0; i < M; i++) {
        double u_exact = analyticalSolution(x[i], T, a, L);
        double diff = u[i][N - 1] - u_exact;
        error_sq += diff * diff;
    }
    return sqrt(error_sq * dx);
}

// Linear interpolation for comparison
double linearInterpolate(double* x_old, double* u_old, int n_old, double x_new) {
    if (x_new <= x_old[0]) return u_old[0];
    if (x_new >= x_old[n_old - 1]) return u_old[n_old - 1];
    int i = 0;
    while (i < n_old - 1 && x_old[i+1] < x_new) i++;     
    double xL = x_old[i], xR = x_old[i + 1];
    double uL = u_old[i], uR = u_old[i + 1];
    return uL + (uR - uL) * (x_new - xL) / (xR - xL);
}

int main() {
    double L = 1.0, T = 1.0, a = 1;
    double R_target = 0.5; // Targetted CFL number

    // Mesh sizes
    int M1 = 10, M2 = 20, M3 = 40;

    int N1 = static_cast<int>(a * T / (R_target * (L / M1))) + 1;
    int N2 = static_cast<int>(a * T / (R_target * (L / M2))) + 1;
    int N3 = static_cast<int>(a * T / (R_target * (L / M3))) + 1;

    // Allocate memory for M1
    double** u1 = new double*[M1];
    for (int i = 0; i < M1; i++) {
        u1[i] = new double[N1];
    }
    double* x1 = new double[M1];
    double* t1 = new double[N1];

    // Allocate memory for M2
    double** u2 = new double*[M2];
    for (int i = 0; i < M2; i++){ 
        u2[i] = new double[N2];
    }
    double* x2 = new double[M2];
    double* t2 = new double[N2];

    // Allocate memory for M3
    double** u3 = new double*[M3];
    for (int i = 0; i < M3; i++){ 
        u3[i] = new double[N3];
    }
    double* x3 = new double[M3];
    double* t3 = new double[N3];

    // Run simulations
    runLaxWendroff(a, L, T, M1, N1, u1, x1, t1);
    runLaxWendroff(a, L, T, M2, N2, u2, x2, t2);
    runLaxWendroff(a, L, T, M3, N3, u3, x3, t3);

    // L2 errors
    double L2_1 = computeL2Analytical(x1, u1, a, T, L, M1, N1);
    double L2_2 = computeL2Analytical(x2, u2, a, T, L, M2, N2);
    double L2_3 = computeL2Analytical(x3, u3, a, T, L, M3, N3);

    cout<<"L2 Error Norm M1: "<< L2_1 << endl;
    cout<<"L2 Error Norm M2: "<< L2_2 << endl;
    cout<<"L2 Error Norm M3: "<< L2_3 << endl;

    // EOC estimation
    double h1 = L / M1;  
    double h2 = L / M2;
    double h3 = L / M3;

    double EOC1 = log(L2_1 / L2_2) / log(h1 / h2);
    double EOC2 = log(L2_2 / L2_3) / log(h2 / h3);

    cout <<"EOC (M1 -> M2): " << EOC1 << endl;
    cout <<"EOC (M2 -> M3): " << EOC2 << endl;

    // Prepare 1D solution arrays for output
    double* u1_last = new double[M1];
    double* u2_last = new double[M2];
    for (int i = 0; i < M1; i++){ 
        u1_last[i] = u1[i][N1 - 1];
    }
    for (int i = 0; i < M2; i++){ 
        u2_last[i] = u2[i][N2 - 1];
    }

    // Write output file for visualization
    ofstream file("lax_wendroff_vs_analytical_all_meshes.csv");
    file << "x,u_numeric_M1,u_analytical,u_numeric_M2,u_analytical,u_numeric_M3,u_analytical"<<endl;
    for (int i = 0; i < M3; i++) {
        double x_val = x3[i];
        double u_exact = analyticalSolution(x_val, T, a, L);
        double u_num_1 = linearInterpolate(x1, u1_last, M1, x_val);
        double u_num_2 = linearInterpolate(x2, u2_last, M2, x_val);
        double u_num_3 = u3[i][N3 - 1];
        file << x_val << "," << u_num_1 << "," << u_exact << "," << u_num_2 << "," << u_exact << "," << u_num_3 << "," << u_exact <<endl;
    }
    file.close();
    cout << "Data saved to lax_wendroff_vs_analytical_all_meshes.csv" << endl;

    // Cleanup
    for (int i = 0; i < M1; i++){ 
        delete[] u1[i];
    }
    for (int i = 0; i < M2; i++){ 
        delete[] u2[i];
    }
    for (int i = 0; i < M3; i++){ 
        delete[] u3[i];
    }
    delete[] u1; delete[] u2; delete[] u3;
    delete[] x1; delete[] x2; delete[] x3;
    delete[] t1; delete[] t2; delete[] t3;
    delete[] u1_last; delete[] u2_last;

    return 0;
}