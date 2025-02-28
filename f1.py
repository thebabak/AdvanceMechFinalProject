import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from control import StateSpace, forced_response

# Beam properties
rho_b = 2710  # Density of the beam (kg/m^3)
E_b = 71e9  # Young's modulus (Pa)
v_b = 0.3  # Poisson's ratio
t_b = 5e-4  # Beam thickness (m)
L_b = 0.5  # Beam length (m)
b = 0.01  # Beam width (m)
J_b = b * t_b**3 / 12  # Second moment of area (m^4)
A_b = b * t_b  # Cross-sectional area (m^2)
zeta = np.diag([0.01] * 4)  # Damping coefficients (4 modes)

# Piezoelectric patch properties
rho = 7500  # Density of the piezoelectric patch (kg/m^3)
E = 126e9  # Young's modulus (Pa)
d31 = 2.3e-11  # Piezoelectric strain coefficient (m/V)
h31 = 4.32e8  # Piezoelectric stress coefficient (N/m^2)
v = 0.3  # Poisson's ratio
t = 1e-4  # Thickness of the piezoelectric patch (m)

# Input parameters
X = np.array([0.0268, 0.1268])  # Start and end positions of the piezoelectric patch (m)

# Derived constants
Ka = b * ((t_b + t) / 2) * d31 * E  # Actuation constant (N/V)
Ks1 = -t * h31 * ((t_b + t) / 2) / (X[1] - X[0])  # Sensing constant (N/m)

# Natural frequencies for the first 4 modes
wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))  # Base natural frequency
W = wj * np.diag([1, 4, 9, 16])  # Natural frequency matrix for 4 modes

# Mode shape derivative differences for all modes
def mode_shape_diff(X, n):
    return (np.sqrt(2 / (rho_b * L_b * A_b)) * (n * np.pi / L_b) *
            (np.cos(n * np.pi * X[1] / L_b) - np.cos(n * np.pi * X[0] / L_b)))

U_diffs = np.array([mode_shape_diff(X, n) for n in range(1, 5)])  # Shape differences

# Open-loop matrices
B_ = Ka * U_diffs.reshape(4, 1)  # Actuation matrix (4, 1)
C_ = Ks1 * U_diffs.reshape(1, 4)  # Sensing matrix (1, 4)

# State-space system matrices (open-loop)
A = np.block([
    [np.zeros((4, 4)), np.eye(4)],
    [-W @ W, -2 * zeta @ W]
])  # System matrix (8, 8)
B = np.vstack([np.zeros((4, 1)), B_])  # Input matrix (8, 1)
C = np.hstack([C_, np.zeros((1, 4))])  # Output matrix (1, 8)

# Closed-loop system matrix
G = 0.4  # Feedback gain
Ac = np.block([
    [np.zeros((4, 4)), np.eye(4)],
    [-W @ W, -2 * zeta @ W + G * (B_ @ C_)]
])  # Closed-loop system matrix (8, 8)

# Initial conditions
n0 = np.array([0, 0, 0, 0])  # Initial displacements (4,)
n0_d = np.array([0.2, 0.4, 0.6, 0.8])  # Initial velocities (4,)
n = np.concatenate([n0, n0_d])  # Total initial state (8,)

# Lyapunov equation solution
Q = np.block([
    [W @ W, np.zeros((4, 4))],
    [np.zeros((4, 4)), np.eye(4)]
])  # Lyapunov equation matrix (8, 8)
P = solve_continuous_lyapunov(Ac.T, -Q)  # Solution to the Lyapunov equation (8, 8)
f = np.max(-n @ P @ n)  # Scalar performance metric

# System response
U = np.zeros((1, 500))  # Zero input (1, time steps)
t = np.linspace(0, 1.5, 500)  # Time vector (500 points)

# Open-loop and closed-loop responses
t_ol, y_ol = forced_response(StateSpace(A, B, C, 0), T=t, U=U, X0=n)
t_cl, y_cl = forced_response(StateSpace(Ac, B, C, 0), T=t, U=U, X0=n)

# Check if the responses are 1D or 2D
if y_ol.ndim == 1:
    # For SISO systems
    plt.figure()
    plt.plot(t_ol, y_ol, label='Open Loop Response')
    plt.plot(t_cl, y_cl, label='Closed Loop Response')
    plt.title('System Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.legend()
    plt.grid()
    plt.show()
else:
    # For MIMO systems (e.g., multiple modes)
    for i in range(4):  # Modes 1 to 4
        plt.figure()
        plt.plot(t_ol, y_ol[i, :], label=f'Open Loop Response (Mode {i + 1})')
        plt.plot(t_cl, y_cl[i, :], label=f'Closed Loop Response (Mode {i + 1})')
        plt.title(f'System Response for Mode {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Displacement (Mode {i + 1})')
        plt.legend()
        plt.grid()
        plt.show()

# Print Lyapunov performance metric
print(f"Lyapunov performance metric: {f:.6f}")