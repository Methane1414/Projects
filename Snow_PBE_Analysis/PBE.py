import pandas as pd
import numpy as np

# Load avalanche data from C++ simulation
avalanche_data = pd.read_csv("C:\\Users\\chaur\\Documents\\Projects\\Snow_PBE_Analysis\\avalanche_data.csv")

# Define size categories for Population Balance Equation (PBE) analysis
size_categories = np.linspace(0.1, 2.0, 20)  # 20 categories from 0.1m to 2.0m

# Initialize population distribution
population_distribution = np.zeros((len(avalanche_data["time"].unique()), len(size_categories) - 1))

# Process data step by step
time_steps = sorted(avalanche_data["time"].unique())
for t_idx, t in enumerate(time_steps):
    data_t = avalanche_data[avalanche_data["time"] == t]
    hist, _ = np.histogram(data_t["size"], bins=size_categories)
    population_distribution[t_idx] = hist

# Fragmentation model (simple rule: break large particles into smaller ones)
def fragmentation(population, breakage_prob=0.05):
    new_population = np.copy(population)
    for i in range(1, len(population)):
        broken = breakage_prob * population[i]
        new_population[i] -= broken
        new_population[i - 1] += broken  # Move to smaller size category
    return new_population

# Apply fragmentation over time
for t in range(1, len(time_steps)):
    population_distribution[t] = fragmentation(population_distribution[t])

# Compute statistical properties
total_population = np.sum(population_distribution, axis=1)
valid_mask = total_population > 0  # Mask to avoid division by zero

mean_size = np.zeros_like(total_population)
std_dev_size = np.zeros_like(total_population)

mean_size[valid_mask] = np.sum(population_distribution[valid_mask] * size_categories[:-1], axis=1) / total_population[valid_mask]
std_dev_size[valid_mask] = np.sqrt(np.sum(population_distribution[valid_mask] * (size_categories[:-1] - mean_size[valid_mask, None]) ** 2, axis=1) / total_population[valid_mask])

# Save results to a new CSV
pbm_results = pd.DataFrame({"time": time_steps, "mean_size": mean_size, "std_dev_size": std_dev_size})
pbm_results.to_csv("pbm_results.csv", index=False)

print("PBE analysis complete. Data saved in pbm_results.csv")