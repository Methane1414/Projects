#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>  // For measuring execution time
using namespace std;

// Global Variables
const double F0 = 40.0;   // ft3/h
const double T0 = 530;    // R
const double CA0 = 0.5;   // lbmolA/ft3
const double alpha = 7.08e10;  // h 1
const double E = 30000;   // Btu/lbmol
const double R = 1.99;    // Btu/lbmol R
const double rho = 50;    // lbm/ft3
const double CP = 0.75;   // Btu/lbm R
const double U = 150;     // Btu/h ft2 R
const double AH = 250;    // ft2
const double rhoJ = 62.3; // lbm/ft3
const double VJ = 48.0;   // ft3
const double CJ = 1.0;    // Btu/lbm R
const double TJ0 = 530;   // R
const double Kc = 4; // ft3/hr R
const double end_time = 4.0; // hr
const double delta_t = 0.0001; // Step size
const double EPSILON = 1e-6;  // Small value to prevent division by zero

// Function to compute derivatives
void compute_derivatives(double V, double Ca, double T, double Tj, double d[]) {
    double k;
    if (T > 0) {
        k = alpha * exp(-E / (R * T)); // Normal computation
    } else {
        k = 0; // Avoid exponential overflow
    }
    double FJ = 49.9 - Kc*(600 - T);
    d[0] = F0 - (10 * V - 440);
    d[1] = (F0 * CA0 - (10 * V - 440) * Ca - k * V * Ca)/V;
    d[2] = ((F0 * T0 - (10 * V - 440) * T + (30000 * V * k * Ca - U * AH * (T - Tj)) / (rho * CP + EPSILON)))/V;
    d[3] = (FJ / VJ) * (TJ0 - Tj) + (U * AH * (T - Tj)) / (rhoJ * VJ * CJ + EPSILON);
}

// Runge-Kutta 4th Order Method (RK4)
void RK4(double h, double &V, double &Ca, double &T, double &Tj) {
    double k1[4], k2[4], k3[4], k4[4], temp[4];

    // Compute k1
    compute_derivatives(V, Ca, T, Tj, k1);
    for (int i = 0; i < 4; i++) 
        temp[i] = (i == 0 ? V : (i == 1 ? Ca : (i == 2 ? T : Tj))) + k1[i] * h / 2;

    // Compute k2
    compute_derivatives(temp[0], temp[1], temp[2], temp[3], k2);
    for (int i = 0; i < 4; i++) 
        temp[i] = (i == 0 ? V : (i == 1 ? Ca : (i == 2 ? T : Tj))) + k2[i] * h / 2;

    // Compute k3
    compute_derivatives(temp[0], temp[1], temp[2], temp[3], k3);
    for (int i = 0; i < 4; i++) 
        temp[i] = (i == 0 ? V : (i == 1 ? Ca : (i == 2 ? T : Tj))) + k3[i] * h;

    // Compute k4
    compute_derivatives(temp[0], temp[1], temp[2], temp[3], k4);

    // Update values
    V = V + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
    Ca = Ca + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
    T = T + h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6;
    Tj = Tj + h * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6;
}

int main() {
    // Initial conditions
    double V = 48;
    double Ca = 0.245;
    double T = 600;
    double Tj = 594.59;
    double t = 0;
    double h = delta_t;

    ofstream file("Dynamic_CSTR.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open file!" << endl;
        return 1;
    }

    file << "t,V,Ca,T,Tj"<<endl;

    //Start measuring computation time
    clock_t start_time = clock();

    while (t < end_time) {
        RK4(h, V, Ca, T, Tj);
        file << t << "," << V << "," << Ca << "," << T << "," << Tj <<endl;
        t = t + h;
    }

    // Stop measuring computation time
    clock_t end_time = clock();
    double computation_time = double(end_time - start_time) / CLOCKS_PER_SEC;

    // Output computation time
    cout << "\nComputation Time: " << computation_time << " seconds\n";

    file.close();
    return 0;
}
