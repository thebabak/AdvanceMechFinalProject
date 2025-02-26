import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.signal import StateSpace, lsim

def F3(X):
    # Beam properties
    rho_b = 1190  # density, kg/m^3
    E_b = 3.1028e9  # young's modulus, Pa
    v_b = 0.3  # poisson's ratio, dimensionless
    t_b = 1.6e-3  # thickness, m
    L_b = 0.5  # length, m
    b = 0.01  # width, m
    J_b = b * t_b**3 / 12  # Area Moment of Inertia
    A_b = b * t_b  # cross-sectional area of the beam
    zeta = np.diag([0.01, 0.01, 0.01, 0.01])  # damping ratio, dimensionless

    # Piezoelectric patch properties
    rho = 1800  # density, kg/m^3
    E = 2e9  # young's modulus, Pa
    d31 = 2.3e-11  # piezoelectric constant, m/V
    h31 = 4.32e8  # piezoelectric constant, V/m
    t = 4e-5  # thickness, m

    # Derived constants
    Ka = b * ((t_b + t) / 2) * d31 * E
    Ks1 = -t * h31 * ((t_b + t) / 2) / (X[1] - X[0])
    Ks2 = -t * h31 * ((t_b + t) / 2) / (X[3] - X[2])
    Ks3 = -t * h31 * ((t_b + t) / 2) / (X[5] - X[4])

    # Natural frequencies for first 4 modes
    wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))
    W = wj * np.diag([1, 2**2, 3**2, 4**2])

    # Mode shape derivative differences for all elements
    # Element 1
    U1diff21 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (1 * np.pi / L_b) * (np.cos((1 * np.pi * X[1]) / L_b) - np.cos((1 * np.pi * X[0]) / L_b))
    U2diff21 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (2 * np.pi / L_b) * (np.cos((2 * np.pi * X[1]) / L_b) - np.cos((2 * np.pi * X[0]) / L_b))
    U3diff21 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (3 * np.pi / L_b) * (np.cos((3 * np.pi * X[1]) / L_b) - np.cos((3 * np.pi * X[0]) / L_b))
    U4diff21 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (4 * np.pi / L_b) * (np.cos((4 * np.pi * X[1]) / L_b) - np.cos((4 * np.pi * X[0]) / L_b))

    # Element 2
    U1diff43 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (1 * np.pi / L_b) * (np.cos((1 * np.pi * X[3]) / L_b) - np.cos((1 * np.pi * X[2]) / L_b))
    U2diff43 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (2 * np.pi / L_b) * (np.cos((2 * np.pi * X[3]) / L_b) - np.cos((2 * np.pi * X[2]) / L_b))
    U3diff43 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (3 * np.pi / L_b) * (np.cos((3 * np.pi * X[3]) / L_b) - np.cos((3 * np.pi * X[2]) / L_b))
    U4diff43 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (4 * np.pi / L_b) * (np.cos((4 * np.pi * X[3]) / L_b) - np.cos((4 * np.pi * X[2]) / L_b))

    # Element 3
    U1diff65 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (1 * np.pi / L_b) * (np.cos((1 * np.pi * X[5]) / L_b) - np.cos((1 * np.pi * X[4]) / L_b))
    U2diff65 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (2 * np.pi / L_b) * (np.cos((2 * np.pi * X[5]) / L_b) - np.cos((2 * np.pi * X[4]) / L_b))
    U3diff65 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (3 * np.pi / L_b) * (np.cos((3 * np.pi * X[5]) / L_b) - np.cos((3 * np.pi * X[4]) / L_b))
    U4diff65 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (4 * np.pi / L_b) * (np.cos((4 * np.pi * X[5]) / L_b) - np.cos((4 * np.pi * X[4]) / L_b))

    # Open loop matrices
    B_ = Ka * np.array([[U1diff21, U1diff43, U1diff65],
                        [U2diff21, U2diff43, U2diff65],
                        [U3diff21, U3diff43, U3diff65],
                        [U4diff21, U4diff43, U4diff65]])

    C_ = np.array([[Ks1 * U1diff21, Ks1 * U2diff21, Ks1 * U3diff21, Ks1 * U4diff21],
                   [Ks2 * U1diff43, Ks2 * U2diff43, Ks2 * U3diff43, Ks2 * U4diff43],
                   [Ks3 * U1diff65, Ks3 * U2diff65, Ks3 * U3diff65, Ks3 * U4diff65]])

    A = np.block([[np.zeros((4, 4)), np.eye(4)],
                   [-W @ W, -2 * zeta @ W]])
    B = np.vstack([np.zeros((4, 3)), B_])
    C = np.hstack([C_, np.zeros((3, 4))])
    Q = np.block([[W @ W, np.zeros((4, 4))],
                   [np.zeros((4, 4)), np.eye(4)]])

    # Close loop matrices
    G = np.array([[0.4, 0, 0],
                  [0, 0.4, 0.0342],
                  [0, 0, 0.4]])
    Ac = np.block([[np.zeros((4, 4)), np.eye(4)],
                    [-W @ W, B_ @ G @ C_ - 2 * zeta @ W]])

    # Initial conditions
    n0 = np.array([0, 0, 0, 0])
    n0_d = np.array([0.2, 0.4, 0.6, 0.8])
    n = np.concatenate([n0, n0_d])

    # Solve Lyapunov equation
    P = solve_continuous_lyapunov(Ac.T, -Q)
    f = np.max(-n @ P @ n.T, 0)

    return Ac, A, B, C, n

# Example usage
X = [0.0326, 0.1326, 0.1406, 0.2406, 0.3661, 0.4661, 0.1393]
Ac, A, B, C, n = F3(X)

# Simulation of system response
t_span = np.linspace(0, 1.5, 500)

# Create state-space system
sys = StateSpace(Ac, B, C, np.zeros((3, 3)))  # D matrix as zero

# Simulate the system response
t, y, x_state = lsim(sys, U=np.zeros_like(t_span), T=t_span, X0=n)

# Plotting results
plt.figure(figsize=(12, 8))

# Plot the responses for each state
for i in range(y.shape[1]):
    plt.plot(t, y[:, i], label=f'Response of State {i + 1}')

plt.title('System Response Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.legend()
plt.grid()
plt.show()