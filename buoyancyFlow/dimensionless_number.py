
# Given parameters
g = 9.81                  # Gravity (m/s²)
beta = 0.000214           # Thermal expansion coefficient (1/K)
T_hot = 350               # Hot wall temperature (K)
T_cold = 300              # Cold wall temperature (K)
L = 0.1                   # Characteristic length (m)
nu = 1.5e-5               # Kinematic viscosity (m²/s)
alpha = 2.0e-5            # Thermal diffusivity (m²/s)

# Input validation
if any(param <= 0 for param in [g, beta, T_hot, T_cold, L, nu, alpha]):
    raise ValueError("All parameters must be positive.")

# Grashof Number
Gr = (g * beta * (T_hot - T_cold) * L**3) / (nu**2)

# Prandtl Number
Pr = nu / alpha

# Rayleigh Number
Ra = Gr * Pr

print(f"Grashof Number (Gr): {Gr:.2e}")
print(f"Prandtl Number (Pr): {Pr:.2e}")
print(f"Rayleigh Number (Ra): {Ra:.2e}")

# Flow Type Estimation
if Ra < 1e7:
    flow_type = "Laminar"
elif 1e7 <= Ra < 1e9:
    flow_type = "Transition"
else:
    flow_type = "Turbulent"

print(f"Estimated Flow Type: {flow_type}")
