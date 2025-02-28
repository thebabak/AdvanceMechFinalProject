import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lqr(A, B, Q, R):
    """Solve the continuous-time LQR controller."""
    # Solve the continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(R) @ B.T @ P
    return K

def simulate_beam_response(A, B, C, K, x0, t_span, t_eval):
    """Simulate the beam response using the LQR controller."""
    def state_equation(t, x):
        u = -K @ x  # Control input
        return A @ x + B @ u

    sol = solve_ivp(state_equation, t_span, x0, t_eval=t_eval)
    y = C @ sol.y  # Output (displacement)
    return sol.t, y

def user_defined_piezo_with_lqr():
    # Beam properties
    rho_b = 2710  # density, kg/m^3
    E_b = 71e9  # Young's modulus, Pa
    t_b = 5e-3  # thickness, m
    L_b = 0.5  # length, m
    b = 0.01  # width, m
    J_b = b * t_b**3 / 12  # Area Moment of Inertia
    A_b = b * t_b  # Cross-sectional area of the beam
    zeta = np.diag([0.01, 0.01, 0.01, 0.01])  # Damping ratio, dimensionless

    # Piezoelectric patch properties
    d31 = -6.5  # Piezoelectric constant, m/V
    t = 4e-5  # thickness, m

    # Derived constants
    Ka = b * ((t_b + t) / 2) * d31 * E_b

    # Natural frequencies for first 4 modes
    wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))
    W = wj * np.diag([1, 2**2, 3**2, 4**2])

    # State-space matrices
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [-W @ W, -2 * zeta @ W]])
    B = np.vstack([np.zeros((4, 1)), np.ones((4, 1)) * Ka])
    C = np.block([np.eye(4), np.zeros((4, 4))])

    # LQR weights
    Q = np.eye(8)  # State weight
    R = np.eye(1)  # Control input weight

    # Calculate LQR gain
    K = lqr(A, B, Q, R)

    # Initial conditions
    x0 = np.zeros(8)  # State vector: [displacements, velocities]
    x0[0] = 1e-4  # Initial displacement for the first mode

    # Simulation time
    t_span = (0, 1)  # Time span: 0 to 1 second
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points for evaluation

    # Simulate responses for three cases
    # Case 1: Optimal patch position
    t_optimal, y_optimal = simulate_beam_response(A, B, C, K, x0, t_span, t_eval)

    # Case 2: Type 1 suboptimal position (adjust B slightly)
    B_type1 = np.vstack([np.zeros((4, 1)), np.ones((4, 1)) * Ka * 0.8])  # Slightly weaker actuation
    K_type1 = lqr(A, B_type1, Q, R)
    t_type1, y_type1 = simulate_beam_response(A, B_type1, C, K_type1, x0, t_span, t_eval)

    # Case 3: Type 2 suboptimal position (adjust B further)
    B_type2 = np.vstack([np.zeros((4, 1)), np.ones((4, 1)) * Ka * 0.6])  # Even weaker actuation
    K_type2 = lqr(A, B_type2, Q, R)
    t_type2, y_type2 = simulate_beam_response(A, B_type2, C, K_type2, x0, t_span, t_eval)

    # Plot the responses
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot displacement for mode 1
    axes[0].plot(t_optimal, y_optimal[0, :], 'r-', label='Optimal')
    axes[0].plot(t_type1, y_type1[0, :], 'k-', label='Type 1')
    axes[0].plot(t_type2, y_type2[0, :], 'b-', label='Type 2')
    axes[0].set_title('Mode 1 Displacement')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Displacement (m)')
    axes[0].grid(True)
    axes[0].legend()

    # Plot displacement for mode 2
    axes[1].plot(t_optimal, y_optimal[1, :], 'r-', label='Optimal')
    axes[1].plot(t_type1, y_type1[1, :], 'k-', label='Type 1')
    axes[1].plot(t_type2, y_type2[1, :], 'b-', label='Type 2')
    axes[1].set_title('Mode 2 Displacement')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Displacement (m)')
    axes[1].grid(True)
    axes[1].legend()

    # Plot displacement for mode 3
    axes[2].plot(t_optimal, y_optimal[2, :], 'r-', label='Optimal')
    axes[2].plot(t_type1, y_type1[2, :], 'k-', label='Type 1')
    axes[2].plot(t_type2, y_type2[2, :], 'b-', label='Type 2')
    axes[2].set_title('Mode 3 Displacement')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Displacement (m)')
    axes[2].grid(True)
    axes[2].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Run the function
if __name__ == "__main__":
    user_defined_piezo_with_lqr()