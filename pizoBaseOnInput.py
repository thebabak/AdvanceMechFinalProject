import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt
from tkinter import Tk, simpledialog

def user_defined_piezo():
    # Initialize Tkinter
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    # Beam properties
    rho_b = 2710  # density, kg/m^3
    E_b = 71e9  # Young's modulus, Pa
    v_b = 0.3  # Poisson's ratio, dimensionless
    t_b = 5e-3  # thickness, m
    L_b = 0.5  # length, m
    b = 0.01  # width, m
    J_b = b * t_b**3 / 12  # Area Moment of Inertia
    A_b = b * t_b  # Cross-sectional area of the beam
    zeta = np.diag([0.01, 0.01, 0.01, 0.01])  # Damping ratio, dimensionless

    # Piezoelectric patch properties
    rho = 7500  # density, kg/m^3
    E = 126e9  # Young's modulus, Pa
    d31 = -6.5  # Piezoelectric constant, m/V
    h31 = 1.5e-8  # Piezoelectric constant, V/m
    t = 4e-5  # thickness, m

    # Ask user for number of patches
    num_patches = simpledialog.askinteger("Input", "Enter the number of piezoelectric patches (at least 1):")
    if num_patches is None or num_patches < 1:
        print('The number of patches must be at least 1.')
        return

    # Ask user for the positions of the patches
    positions = np.zeros(2 * num_patches)  # Each patch has start and end positions
    for i in range(num_patches):
        while True:
            start = simpledialog.askfloat("Input", f"Enter the start position of patch {i + 1} (in meters, between 0 and {L_b}):")
            end = simpledialog.askfloat("Input", f"Enter the end position of patch {i + 1} (in meters, between 0 and {L_b}):")
            if start is None or end is None:
                print("No input provided. Exiting.")
                return
            if 0 <= start < end <= L_b:
                positions[2 * i] = start
                positions[2 * i + 1] = end
                break
            else:
                simpledialog.messagebox.showerror("Error", "Invalid positions. Ensure the start is less than the end and within the beam length.")

    # Derived constants
    Ka = b * ((t_b + t) / 2) * d31 * E
    Ks = np.zeros(num_patches)
    for i in range(num_patches):
        Ks[i] = -t * h31 * ((t_b + t) / 2) / (positions[2 * i + 1] - positions[2 * i])

    # Natural frequencies for first 4 modes
    wj = (np.pi / L_b)**2 * np.sqrt(E_b * J_b / (rho_b * A_b))
    W = wj * np.diag([1, 2**2, 3**2, 4**2])

    # Mode shape derivative differences for all elements
    Udiff = np.zeros((4, num_patches))
    for i in range(num_patches):
        U1 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (1 * np.pi / L_b) * (
            np.cos((1 * np.pi * positions[2 * i + 1]) / L_b) - np.cos((1 * np.pi * positions[2 * i]) / L_b)
        )
        U2 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (2 * np.pi / L_b) * (
            np.cos((2 * np.pi * positions[2 * i + 1]) / L_b) - np.cos((2 * np.pi * positions[2 * i]) / L_b)
        )
        U3 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (3 * np.pi / L_b) * (
            np.cos((3 * np.pi * positions[2 * i + 1]) / L_b) - np.cos((3 * np.pi * positions[2 * i]) / L_b)
        )
        U4 = (np.sqrt(2 / (rho_b * L_b * A_b))) * (4 * np.pi / L_b) * (
            np.cos((4 * np.pi * positions[2 * i + 1]) / L_b) - np.cos((4 * np.pi * positions[2 * i]) / L_b)
        )
        Udiff[:, i] = [U1, U2, U3, U4]

    # Open loop matrices
    B_ = Ka * Udiff
    C_ = np.zeros((num_patches, 4))
    for i in range(num_patches):
        C_[i, :] = Ks[i] * Udiff[:, i]
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [-W @ W, -2 * zeta @ W]])
    B = np.vstack([np.zeros((4, num_patches)), B_])
    C = np.hstack([C_, np.zeros((num_patches, 4))])
    Q = np.block([[W @ W, np.zeros((4, 4))], [np.zeros((4, 4)), np.eye(4)]])

    # Close loop matrices
    G = np.eye(num_patches) * 0.4  # Control gain (diagonal for simplicity)
    Ac = np.block([[np.zeros((4, 4)), np.eye(4)], [-W @ W, B_ @ G @ C_ - 2 * zeta @ W]])

    # Initial conditions
    n0 = [1, 0, 0, 0]  # Initial displacement (set the first mode to 1 as an example)
    n0_d = [0, 0, 0, 0]  # Initial velocities
    n = np.hstack((n0, n0_d))

    # Calculate system response
    P = solve_continuous_lyapunov(Ac.T, -Q)  # Solve Lyapunov equation
    response = np.diag(P)  # Extract the diagonal terms (energy contribution of each mode)

    # Plot the response
    plt.figure(figsize=(10, 6))

    # Plot patch positions
    plt.subplot(2, 1, 1)
    patches = np.arange(1, num_patches + 1)
    bar_width = 0.4
    for i in range(num_patches):
        plt.bar(patches[i], positions[2 * i + 1] - positions[2 * i], bar_width, align='center', color='cyan')
    plt.title('Piezoelectric Patch Positions')
    plt.xlabel('Patch Index')
    plt.ylabel('Length (m)')
    plt.xticks(patches, [f'Patch {i + 1}' for i in range(num_patches)])
    plt.grid(True)

    # Plot response (mode energy contributions)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, 5), response[:4], 'o-', linewidth=2, label='Mode Response')
    plt.title('System Response (Mode Energy Contributions)')
    plt.xlabel('Mode Number')
    plt.ylabel('Response Amplitude')
    plt.xticks(np.arange(1, 5))
    plt.grid(True)
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()

    # Display response values
    print('System response (mode energy contributions):')
    print(response[:4])


# Run the function
if __name__ == "__main__":
    user_defined_piezo()