#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
using namespace std;

// Global Constants
const double k0 = 1e3;         // Pre-exponential factor
const double E = 8000;         // Activation energy in (J/mol)
const double R = 8.314;        // Universal gas constant in (J/mol.K)
const double C_A0 = 1.0;       // Initial concentration in (mol/L)
const double V = 100.0;        // Volume of CSTR in (L)

//Function to calculate rate constant using Arhenius equation
double reaction_rate(double T){
    return k0*exp(-E/(R*T));
}
// Function to calculate Conversion of Steady State CSTR
double conversion(double F_A0,double T,double tau){
    double k=reaction_rate(T);
    double C_A=C_A0/(1 + k*tau);
    double X = 1 - (C_A/C_A0);
    return X;
}

int main(){
    ofstream file("Sythetic_dataset.csv");
    if (!file.is_open()){
        cerr<<"Error opening file!"<< endl;
        return 1;
    }

    file<<"F_A0,T,tau,X"<<endl;

    //Random number Generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist_F_A0(0.1, 10.0);  // Inlet flow rate in (L/s)
    uniform_real_distribution<> dist_T(300, 1000);     // Temperature in (K)
    uniform_real_distribution<> dist_tau(0, 10.0);  // Residence time in (s)

    int samples=1000;

    for(int i=0;i<samples;i++){
        double F_A0 = dist_F_A0(gen); 
        double T = dist_T(gen);
        double tau = dist_tau(gen);

        double X =conversion(F_A0,T,tau);
        file<<F_A0<<","<<T<<","<<tau<<","<<X<<endl;
    }
    
    file.close();
    cout<<"DATA IS SAVED "<<endl;

 
return 0;
}