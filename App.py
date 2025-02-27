import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import random
import tkinter as tk
from tkinter import simpledialog, messagebox

# Beam and Piezoelectric Material Properties
beam_length = 0.5  # in meters
beam_width = 0.01  # in meters
beam_height = 0.005  # in meters
piezo_length = 0.05  # in meters
piezo_width = 0.01  # in meters
piezo_height = 0.001  # in meters
E_beam = 71e9  # Young's modulus of beam (Pa)
E_piezo = 126e9  # Young's modulus of piezo (Pa)
density_beam = 2710  # Density of beam (kg/m^3)
density_piezo = 7500  # Density of piezo (kg/m^3)
d31 = 6.5e-12  # Piezoelectric strain constant (C/N)
n33 = 1.5e-8  # Dielectric constant (F/m)
poisson_ratio = 0.3

# Genetic Algorithm Parameters
population_size = 50  # Reduced population for faster testing
generations = 10  # Reduced generations for faster testing
crossover_prob = 0.8
mutation_prob = 0.05

# Control Parameters
num_elements = 100  # Number of beam elements
state_size = 2 * num_elements  # Include displacement and velocity states
Q = np.eye(state_size) * 1e6  # LQR state weighting matrix (adjusted to match state size)
R = np.array([[1e-2]])   # LQR control weighting matrix

# Define A and B matrices (Extended for second-order system)
A_top = np.zeros((num_elements, num_elements))  # Top-left block for displacements
for i in range(num_elements):
    A_top[i, i] = -2  # Main diagonal
    if i > 0:
        A_top[i, i - 1] = 1  # Subdiagonal
    if i < num_elements - 1:
        A_top[i, i + 1] = 1  # Superdiagonal

A_bottom = np.eye(num_elements)  # Bottom-left block linking velocity to displacement
A = np.block([
    [np.zeros((num_elements, num_elements)), A_bottom],  # Top half (displacement and velocity)
    [A_top, np.zeros((num_elements, num_elements))]      # Bottom half (stiffness)
])

B = np.zeros((state_size, 1))  # Input matrix
B[num_elements - 1, 0] = 1  # Input applied at the last velocity state

# Compute K2 Matrix (Piezoelectric Coupling)
def compute_K2(positions):
    """
    Compute the piezoelectric coupling matrix K2 based on actuator positions.
    """
    K2 = np.zeros((num_elements, len(positions)))
    for i, pos in enumerate(positions):
        K2[pos, i] = 1.0  # Simplified coupling (real implementation is mode-specific)
    return K2

# Cost Function for GA
def cost_function(positions, K2, A, B):
    """
    Compute the cost function J = trace(P) for a given actuator/sensor configuration.
    """
    P = solve_continuous_are(A, B, Q, R)
    return np.trace(P)

# Genetic Algorithm Optimization
def genetic_algorithm(A, B, num_piezo):
    """
    Optimize the placement of piezoelectric patches using a genetic algorithm.
    """
    def generate_population():
        # Generate a population of individuals with `num_piezo` actuator positions
        return [sorted(random.sample(range(num_elements), num_piezo)) for _ in range(population_size)]

    def crossover(parent1, parent2):
        # Perform crossover between two parents
        point = random.randint(1, len(parent1) - 1)
        child1 = sorted(parent1[:point] + parent2[point:])
        child2 = sorted(parent2[:point] + parent1[point:])
        return child1, child2

    def mutate(individual):
        # Mutate a single actuator position randomly
        if random.random() < mutation_prob:
            idx = random.randint(0, len(individual) - 1)
            individual[idx] = random.randint(0, num_elements - 1)
            individual = sorted(set(individual))  # Ensure uniqueness and order
            while len(individual) < num_piezo:  # Ensure `num_piezo` actuators are always present
                individual.append(random.randint(0, num_elements - 1))
                individual = sorted(set(individual))
        return individual

    # Initialize population
    population = generate_population()
    best_solution = None
    best_cost = float('inf')

    for generation in range(generations):
        fitness = []
        for individual in population:
            K2 = compute_K2(individual)  # Compute K2 based on positions
            J = cost_function(individual, K2, A, B)  # Use A and B here
            fitness.append(J)

        # Selection
        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        population = sorted_population[:population_size // 2]

        # Crossover
        for i in range(0, len(population), 2):
            if random.random() < crossover_prob and i + 1 < len(population):
                child1, child2 = crossover(population[i], population[i + 1])
                population.append(child1)
                population.append(child2)

        # Mutation
        population = [mutate(ind) for ind in population]

        # Update best solution
        current_best_cost = min(fitness)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[np.argmin(fitness)]

        print(f"Generation {generation + 1}, Best Cost: {best_cost}")

    return best_solution

# Simulate Beam Dynamics with LQR Control
def simulate_beam(control_type="LQR", control_gain=None):
    """
    Simulate the beam vibration response with a specified control strategy.
    """
    def system_dynamics(t, x):
        u = -np.dot(control_gain, x) if control_gain is not None else 0
        return np.dot(A, x) + np.dot(B, u)

    initial_state = np.zeros(state_size)  # Initial displacement and velocity are zero
    initial_state[0] = 0.01  # Initial displacement at the first element
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(system_dynamics, t_span, initial_state, t_eval=t_eval)
    return sol.t, sol.y

# Visualize Piezoelectric Actuator Placement
def visualize_piezoelectric_positions(positions):
    """
    Visualize the positions of the piezoelectric actuators on the beam.
    """
    plt.figure(figsize=(10, 2))
    plt.plot([0, beam_length], [0, 0], 'k-', linewidth=3, label='Beam')  # Beam as a black line
    for pos in positions:
        x = (pos / num_elements) * beam_length  # Map element index to beam length
        plt.plot([x, x], [-0.05, 0.05], 'r-', linewidth=5, label='Piezoelectric Patch' if pos == positions[0] else "")
    plt.title("Piezoelectric Actuator Placement on Beam")
    plt.xlabel("Beam Length (m)")
    plt.ylabel("Actuator Placement")
    plt.legend()
    plt.grid()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Step 1: Create a popup to ask the user for the number of piezoelectric actuators
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter main window
    num_piezo = simpledialog.askinteger("Input", "Enter the number of piezoelectric actuators:")
    if not num_piezo:
        print("No input provided. Exiting program.")
        exit()

    # Step 2: Optimize Piezoelectric Placement
    optimal_positions = genetic_algorithm(A, B, num_piezo)
    print("Optimal Piezoelectric Positions:", optimal_positions)

    # Step 3: Simulate Beam Dynamics
    K2 = compute_K2(optimal_positions)
    P = solve_continuous_are(A, B, Q, R)
    control_gain = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    t, response = simulate_beam(control_type="LQR", control_gain=control_gain)

    # Step 4: Display Results in a Popup
    results = f"""
    Beam Properties:
    - Length: {beam_length} m
    - Width: {beam_width} m
    - Height: {beam_height} m

    Piezoelectric Properties:
    - Actuator Length: {piezo_length} m
    - Optimal Actuator Positions (element indices): {optimal_positions}

    Simulation:
    - Maximum Displacement: {np.max(response[0, :]):.6f} m
    - Minimum Displacement: {np.min(response[0, :]):.6f} m
    """
    messagebox.showinfo("Simulation Results", results)

    # Step 5: Visualize Piezoelectric Placement
    visualize_piezoelectric_positions(optimal_positions)

    # Step 6: Plot Vibration Response
    plt.plot(t, response[0, :])
    plt.title("Beam Vibration Response with LQR Control")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid()
    plt.show()