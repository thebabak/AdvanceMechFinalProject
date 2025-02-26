import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import differential_evolution, NonlinearConstraint

# Initialize storage for convergence plot
convergence_data = []

def objective_function(X):
    # Beam properties
    rho_b = 1190         # Density, kg/m³
    E_b = 3.1028e9       # Young's modulus, Pa
    t_b = 1.6e-3         # Thickness, m
    L_b = 0.5            # Length, m
    b = 0.01             # Width, m
    J_b = b * t_b**3 / 12  # Area moment of inertia
    A_b = b * t_b        # Cross-sectional area
    zeta = np.diag([0.01]*4)  # Damping ratios

    # Piezoelectric properties
    d31 = 2.3e-11        # Piezoelectric constant, m/V
    h31 = 4.32e8         # Piezoelectric constant, V/m
    t_p = 4e-5           # Thickness, m

    # Derived constants
    Ka = b * ((t_b + t_p)/2) * d31 * E_b
    Ks1 = -t_p * h31 * ((t_b + t_p)/2) / (X[1] - X[0])

    # Natural frequency matrix
    w_j = (np.pi/L_b)**2 * np.sqrt(E_b*J_b/(rho_b*A_b))
    W = w_j * np.diag([1, 4, 9, 16])

    # Mode shape derivatives
    sqrt_term = np.sqrt(2/(rho_b*L_b*A_b))
    U_diff = [sqrt_term * (i*np.pi/L_b) * 
              (np.cos(i*np.pi*X[1]/L_b) - np.cos(i*np.pi*X[0]/L_b)) 
              for i in range(1,5)]

    # Actuation and sensing vectors
    B_ = Ka * np.array(U_diff).reshape(-1, 1)  # Column vector (4x1)
    C_ = Ks1 * np.array(U_diff).reshape(1, -1)  # Row vector (1x4)

    # Closed-loop system matrix
    G = 0.4  # Feedback gain
    control_term = G * B_ @ C_  # (4x4 matrix)
    
    Ac = np.block([
        [np.zeros((4,4)), np.eye(4)],
        [-W@W, control_term - 2*zeta@W]
    ])

    # Lyapunov equation setup
    Q = np.block([
        [W@W, np.zeros((4,4))],
        [np.zeros((4,4)), np.eye(4)]
    ])
    
    try:
        P = solve_continuous_lyapunov(Ac.T, -Q)
    except np.linalg.LinAlgError:
        return np.inf

    # Performance metric
    n = np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8])
    return max(-n @ P @ n.T, 0)

# Optimization constraints
patch_constraint = NonlinearConstraint(
    lambda x: x[1] - x[0],
    lb=0.1, ub=0.1  # Fixed patch length constraint
)

# Callback function for tracking convergence
def optimization_callback(xk, convergence):
    convergence_data.append(objective_function(xk))
    return False

# Run optimization
result = differential_evolution(
    objective_function,
    bounds=[(0, 0.5), (0, 0.5)],
    constraints=patch_constraint,
    callback=optimization_callback,
    popsize=15,
    maxiter=200,
    tol=1e-6,
    polish=True
)

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Optimization Convergence
plt.subplot(1, 2, 1)
plt.semilogy(convergence_data, 'b-o', markersize=4, linewidth=1.5)
plt.title('Optimization Convergence History')
plt.xlabel('Iteration')
plt.ylabel('Performance Metric')
plt.grid(True, which='both', linestyle='--')

# Plot 2: Beam Configuration
plt.subplot(1, 2, 2)
plt.plot([0, 0.5], [0, 0], 'k-', linewidth=10, label='Beam')
plt.plot(result.x, [0, 0], 'r-', linewidth=12, alpha=0.6,
        label=f'Optimal Patch\n({result.x[0]:.3f}m - {result.x[1]:.3f}m)')
plt.title('Optimal Piezoelectric Patch Placement')
plt.xlabel('Position Along Beam (m)')
plt.yticks([])
plt.xlim(0, 0.5)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, axis='x')

plt.tight_layout()
plt.show()

# Print final results
print("\nOptimization Results:")
print(f"Start Position (x1): {result.x[0]:.4f} m")
print(f"End Position (x2):   {result.x[1]:.4f} m")
print(f"Minimum Objective:   {result.fun:.6f}")