//FTBS UPWIND SCHEME WITH EOC AND L2 NORM ERROR ESTIMATES
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;

// Grid generator
void linspace(double start, double end, int N, double* array) {
    double step = (end - start) / (N - 1);
    for (int i = 0; i < N; i++) {
        array[i] = start + i * step;
    }
}

// Initial rectangular pulse (Box Function)
double initialCondition(double x) {
    if (x > 0.5 && x < 0.75){
        return 1.0;
    }    
    return 0.0;
}

// Analytical solution: shift initial condition by a*t to the right
double analyticalSolution(double x, double t, double a) {
    return initialCondition(x -a*t);
}

//FTBS upwind scheme
void runUpwind(double a, double L, double T, int M, int N, double** u, double* x, double* t) {
    double dx = L / (M - 1);
    double dt = T / (N - 1);
    double R = a * dt / dx;

    if (R > 1.0) {
        cout << "CFL condition violated: R ="<< R << endl;
        exit(1);
    }

    linspace(0, L, M, x);
    linspace(0, T, N, t);

    for (int i = 0; i < M; i++){    
        u[i][0] = initialCondition(x[i]);
    }

    for (int j = 0; j < N - 1; j++) {
        u[0][j + 1] = 0.0;  // Fixed BC
        u[M -1][j + 1] = 0.0; // Fixed BC
        for (int i = 1; i < M - 1; i++) {
            u[i][j + 1] = u[i][j] - R * (u[i][j] - u[i - 1][j]);
        }
  
    }
}

// Compute L2 error norm vs analytical
double computeL2Analytical(double* x, double** u, double a, double T, int M, int N) {
    double dx = x[1] - x[0];
    double error_sq = 0.0;
    for (int i = 0; i < M; i++) {
        double u_exact = analyticalSolution(x[i], T, a);
        double diff = u[i][N - 1] - u_exact;
        error_sq = error_sq + diff * diff;
    }
    return sqrt(error_sq * dx);
}

// Linear interpolation function to get all solution at same points
double linearInterpolate(double* x_old, double* u_old, int n_old, double x_new) {
    if (x_new <= x_old[0]){
          return u_old[0];
    }
    if (x_new >= x_old[n_old - 1]){
          return u_old[n_old - 1];
    }

    int i = 0;
    while (i < n_old - 1 && x_old[i+1] < x_new){
          i++;
    }

    double xL = x_old[i];
    double xR = x_old[i + 1];
    double uL = u_old[i];
    double uR = u_old[i + 1];

    return uL + (uR - uL) * (x_new - xL) / (xR - xL);
}

int main() {
    double L = 2.0, T = 1.0, a = 0.4;
    int N = 550;
    int M1 = 800, M2 = 1400, M3 = 2600;

    // Allocate memory for M1
    double** u1 = new double*[M1];
    for (int i = 0; i < M1; i++){ 
         u1[i] = new double[N];
    }
    double* x1 = new double[M1];

    // Allocate memory for M2
    double** u2 = new double*[M2];
    for (int i = 0; i < M2; i++){ 
         u2[i] = new double[N];
    }
    double* x2 = new double[M2];

    // Allocate memory for M3
    double** u3 = new double*[M3];
    for (int i = 0; i < M3; i++){ 
         u3[i] = new double[N];
    }
    double* x3 = new double[M3];

    double* t = new double[N];

    // Run simulations
    runUpwind(a, L, T, M1, N, u1, x1, t);
    runUpwind(a, L, T, M2, N, u2, x2, t);
    runUpwind(a, L, T, M3, N, u3, x3, t);

    // Compute L2 error norms vs analytical
    double L2_1 = computeL2Analytical(x1, u1, a, T, M1, N);
    double L2_2 = computeL2Analytical(x2, u2, a, T, M2, N);
    double L2_3 = computeL2Analytical(x3, u3, a, T, M3, N);

    cout << "L2 Error Norm M1: " << L2_1 << endl;
    cout << "L2 Error Norm M2: " << L2_2 << endl;
    cout << "L2 Error Norm M3: " << L2_3 << endl;

    // Compute EOC
    double h1 = L / (M1 - 1);
    double h2 = L / (M2 - 1);
    double h3 = L / (M3 - 1);

    double EOC1 = log(L2_1 / L2_2) / log(h1 / h2);
    double EOC2 = log(L2_2 / L2_3) / log(h2 / h3);

    cout <<"EOC (M1 -> M2): " << EOC1 << endl;
    cout <<"EOC (M2 -> M3): " << EOC2 << endl;

    // Prepare 1D arrays for last timestep numeric solutions for interpolation
    double* u1_last = new double[M1];
    double* u2_last = new double[M2];
    for (int i = 0; i < M1; i++){ 
         u1_last[i] = u1[i][N - 1];
    }
    for (int i = 0; i < M2; i++){ 
         u2_last[i] = u2[i][N - 1];
    }

    // Save data with all solutions w.r.t to M3 (finest mesh)
    ofstream file("upwind_vs_analytical_all_meshes.csv");
    file << "x,u_numeric_M1,u_analytical,u_numeric_M2,u_analytical,u_numeric_M3,u_analytical"<<endl;

    for (int i = 0; i < M3; i++) {
        double x_values = x3[i];
        double u_exact = analyticalSolution(x_values, T, a);

        double u_num_1 = linearInterpolate(x1, u1_last, M1, x_values);
        double u_num_2 = linearInterpolate(x2, u2_last, M2, x_values);
        double u_num_3 = u3[i][N - 1];

        file << x_values << ","<< u_num_1 << "," << u_exact << ","<< u_num_2 << "," << u_exact << ","<< u_num_3 << "," << u_exact <<endl;
        
    }
    cout<<"Data is saved to csv file"<<endl;
    file.close();

    // Cleanup the memory
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
    delete[] x1; delete[] x2; delete[] x3; delete[] t;
    delete[] u1_last; delete[] u2_last;

    return 0;
}
