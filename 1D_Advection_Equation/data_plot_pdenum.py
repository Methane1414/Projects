import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Plot Style Settings (Advisor Style) ------------------ #
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['text.usetex'] = True  # Uncomment if LaTeX font rendering is needed

# ------------------------ Data from CSV ------------------------ #
file_path = "D:\\Internship\\Dr Sudarshan IISER\\upwind_vs_analytical_all_meshes.csv" #Write your pathway where the csv file is saved
data = pd.read_csv(file_path)

# Assign columns
x = data['x']
u1 = data['u_numeric_M1']
u2 = data['u_numeric_M2']
u3 = data['u_numeric_M3']
ua = data['u_analytical']

# ------------------------- Plotting --------------------------- #
plt.figure()

# Numerical solutions
plt.plot(x, u1, '-', label='UW M1', color='darkgreen', markersize=3)
plt.plot(x, u2, '-', label='UW M2', color='orange', markersize=3)
plt.plot(x, u3, '-', label='UW M3', color='red', markersize=3)

# Analytical solution
plt.plot(x, ua, '-', label='Exact', color='blue')

# Axis, labels, and layout
plt.xlim(0, 1)
plt.xlabel('x')
plt.ylabel('u')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.axis('tight')
plt.autoscale(enable=True, axis='x', tight=True)

# ------------------------- Save Figure -------------------------- #
plt.savefig("solution1.pdf")
print("plot is saved")
plt.show()