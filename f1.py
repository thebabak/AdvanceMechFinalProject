import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from control import StateSpace, forced_response

# Beam properties
rho_b = 1190  # kg/m^3
E_b = 3.1028e9  # Pa
v_b = 0.3  # dimensionless
t_b = 1.6e-3  # m
L_b = 0.5  # m
b = 0.01  # m
J_b = b * t_b**3 / 12  # m^4
A_b = b * t_b  # m^2
zeta = np.diag([0.01, 0.01, 0.01, 0.01])  # dimensionless

# Piezoelectric patch properties
rho = 1800  # kg/m^3
E = 2e9  # Pa
d31 = 2.3e-11  # m/V
h31 = 4.32e8  # V/m
t = 4e-5  # m

# Input parameters
X = np.array([0.0268, 0.1268])  # positions of piezoelectric patches

# Derived constants
Ka = b * ((t_b + t) / 2) * d31 * E  # N/V
Ks1 = -t * h31 * ((t_b + t) / 2) / (X[1] - X[0])  # N/m

# Natural frequencies for first 4 modes
wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))
W = wj * np.diag([1, 2**2, 3**2, 4**2])  # Natural frequencies

# Mode shape derivative differences for all elements
def mode_shape_diff(X, n):
    return (np.sqrt(2 / (rho_b * L_b * A_b)) * (n * np.pi / L_b) *
            (np.cos(n * np.pi * X[1] / L_b) - np.cos(n * np.pi * X[0] / L_b)))

U_diffs = np.array([mode_shape_diff(X, n) for n in range(1, 5)])  # Shape differences

# Open loop matrices
B_ = Ka * U_diffs  # (4,)
B_ = B_.reshape(4, 1)  # (4, 1)

C_ = Ks1 * U_diffs  # (4,)
C_ = C_.reshape(1, 4)  # (1, 4)

# Construct A matrix
A = np.vstack([np.hstack([np.zeros((4, 4)), np.eye(4)]), 
                np.hstack([-W @ W, -2 * zeta @ W])])  # (8, 8)

# Construct B matrix
B = np.vstack([np.zeros((4, 1)), B_])  # (8, 1)

# Construct C matrix
C = np.hstack([C_, np.zeros((1, 4))])  # (1, 8)

# Construct Q matrix
Q = np.vstack([np.hstack([W @ W, np.zeros((4, 4))]), 
                np.hstack([np.zeros((4, 4)), np.eye(4)])])  # (8, 8)

# Close loop matrices
G = 0.4
# Ensure that we multiply correctly
B_G = B_ * G  # B_G should be (4, 1)

# Correctly multiply B_G and C to ensure correct dimensions
B_G_C = B_G @ C  # This will yield (4, 1) @ (1, 8) = (4, 8)

# Construct Ac matrix
Ac = np.zeros((8, 8))  # Initialize a zero matrix for Ac

# Fill in the first part of Ac
Ac[:4, :4] = np.zeros((4, 4))  # Zero matrix
Ac[:4, 4:] = np.eye(4)  # Identity matrix

# Fill in the second part of Ac
Ac[4:, :4] = -W @ W  # (4, 4)
Ac[4:, 4:] = B_G_C[:, :4]  # Adjust to ensure shape compatibility

# Initial conditions
n0 = np.array([0, 0, 0, 0])  # Initial state (4,)
n0_d = np.array([0.2, 0.4, 0.6, 0.8])  # Initial state derivatives (4,)
n = np.concatenate([n0, n0_d])  # Total initial state (8,)

# Lyapunov equation solution
P = solve_continuous_lyapunov(Ac.T, -Q)  # Should be (8, 8)
f = np.max(-n @ P @ n)  # Scalar result

# System response
sys = StateSpace(Ac, np.zeros((8, 1)), np.eye(8), 0)  # State-space representation
t = np.linspace(0, 1.5, 500)  # Time vector

# Compute the initial response using forced response
t_ol, y_ol = forced_response(StateSpace(A, B, C, 0), T=t, U=np.zeros_like(t), X0=n)  # Adjusted unpacking
t, y = forced_response(sys, T=t, U=np.zeros_like(t), X0=n)

# Check shapes
print("Shape of y_ol:", y_ol.shape)
print("Shape of y:", y.shape)

# Plotting results
plt.figure()
plt.plot(t_ol, y_ol, label='Open Loop Response')  # Adjusted for 1D array
plt.plot(t, y[3, :], label='Closed Loop Response')  # Assuming y is 2D
plt.title('Response of the System')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (y)')
plt.legend()
plt.grid()
plt.show()

# Additional plots for other modes
for i in range(1, 4):
    plt.figure()
    plt.plot(t, y[i, :], label='Closed Loop Response')
    plt.plot(t_ol, y_ol, label='Open Loop Response')  # Adjusted for 1D array
    plt.title(f'Response of Mode {i + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Displacement (y{i + 1})')
    plt.legend()
    plt.grid()
    plt.show()