import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import differential_evolution, LinearConstraint

# Initialize storage for plotting
best_fitness = []

def F1(X):
    # Beam properties
    rho_b = 2710
    E_b = 71e9
    v_b = 0.3
    t_b = 5e-4
    L_b = 0.5
    b = 0.01
    J_b = b * t_b**3 / 12  # Corrected exponentiation
    A_b = b * t_b
    zeta = np.diag([0.01] * 4)  # Damping coefficients

    # Piezoelectric patch properties
    rho = 7500
    E = 126e9
    d31 = 2.3e-11
    h31 = 4.32e8
    v = 0.3
    t = 1e-4

    # Derived constants
    Ka = b * ((t_b + t) / 2) * d31 * E_b
    Ks1 = -t * h31 * ((t_b + t) / 2) / (X[1] - X[0])

    # Natural frequencies
    wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))
    W = wj * np.diag([1, 4, 9, 16])  # Scaled for higher modes

    # Mode shape derivatives
    sqrt_term = np.sqrt(2 / (rho_b * L_b * A_b))
    U_diff = [
        sqrt_term * (i * np.pi / L_b) *
        (np.cos(i * np.pi * X[1] / L_b) - np.cos(i * np.pi * X[0] / L_b))
        for i in range(1, 5)
    ]

    # Actuation and sensing vectors
    B_ = Ka * np.array(U_diff).reshape(-1, 1)
    C_ = Ks1 * np.array(U_diff).reshape(1, -1)

    # Closed-loop system matrix
    G = 0.4
    Actuation_term = G * (B_ @ C_)
    
    Ac = np.block([
        [np.zeros((4, 4)), np.eye(4)],
        [-W @ W, Actuation_term - 2 * zeta @ W]
    ]).astype(float)

    # Lyapunov equation
    Q = np.block([
        [W @ W, np.zeros((4, 4))],
        [np.zeros((4, 4)), np.eye(4)]
    ])
    
    P = solve_continuous_lyapunov(Ac.T, -Q)
    n = np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8])

    # Ensure the performance metric is non-negative
    return max(-n @ P @ n.T, 0)

# Define linear constraint
patch_constraint = LinearConstraint(
    [1, -1], lb=0.1, ub=0.1  # X2 - X1 = 0.1
)

# Callback function
def callback(xk, convergence):
    best_fitness.append(F1(xk))
    return False

# Run optimization
result = differential_evolution(
    F1,
    bounds=[(0, 0.4), (0.1, 0.5)],  # Adjusted bounds for valid patch placement
    constraints=patch_constraint,
    callback=callback,
    maxiter=200,
    tol=1e-6,
    polish=True
)

# Visualization
plt.figure(figsize=(12, 5))

# Convergence plot
plt.subplot(1, 2, 1)
plt.plot(best_fitness, 'b-o', linewidth=1.5, markersize=4)
plt.title('Optimization Convergence')
plt.xlabel('Generation')
plt.ylabel('Performance Metric')
plt.grid(True)
plt.yscale('log')

# Beam configuration
plt.subplot(1, 2, 2)
plt.plot([0, 0.5], [0, 0], 'k-', linewidth=10, label='Beam')
plt.plot(result.x, [0, 0], 'r-', linewidth=12, alpha=0.6,
        label=f'Piezo Patch\nX1={result.x[0]:.3f}m\nX2={result.x[1]:.3f}m')
plt.title('Optimal Placement')
plt.xlabel('Position (m)')
plt.yticks([])
plt.xlim(0, 0.5)
plt.legend()
plt.grid(True, axis='x')

plt.tight_layout()
plt.show()

print(f"\nOptimal solution:")
print(f"Start position (X1): {result.x[0]:.4f} m")
print(f"End position (X2): {result.x[1]:.4f} m")
print(f"Performance metric: {result.fun:.6f}")