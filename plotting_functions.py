# plotting_functions.py

import numpy as np
import matplotlib.pyplot as plt

def plot_time_response():
    """Function to plot time response with three subplots."""
    t = np.linspace(0, 1, 1000)  # Time axis
    f1 = np.sin(2 * np.pi * 5 * t)  # Sine wave
    f2 = 0.8 * np.sin(2 * np.pi * 5 * t + np.pi / 8)  # Type1
    f3 = 0.6 * np.sin(2 * np.pi * 5 * t + np.pi / 4)  # Type2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # First subplot
    axes[0].plot(t, f1, 'r-', label='optimal', linewidth=1.5)
    axes[0].plot(t, f2, 'k-', label='type1', linewidth=1.5)
    axes[0].plot(t, f3, 'b-', label='type2', linewidth=1.5)
    axes[0].set_xlabel(r'$t$(s)')
    axes[0].set_ylabel(r'displacement (m)')
    axes[0].legend()
    axes[0].set_title('CC')

    # Second subplot
    axes[1].plot(t, f1 * 0.5, 'r-', label='optimal', linewidth=1.5)
    axes[1].plot(t, f2 * 0.5, 'k-', label='type1', linewidth=1.5)
    axes[1].plot(t, f3 * 0.5, 'b-', label='type2', linewidth=1.5)
    axes[1].set_xlabel(r'$t$(s)')
    axes[1].set_ylabel(r'displacement (m)')
    axes[1].legend()
    axes[1].set_title('CF')

    # Third subplot
    axes[2].plot(t, f1 * 2, 'r-', label='optimal', linewidth=1.5)
    axes[2].plot(t, f2 * 2, 'k-', label='type1', linewidth=1.5)
    axes[2].plot(t, f3 * 2, 'b-', label='type2', linewidth=1.5)
    axes[2].set_xlabel(r'$t$(s)')
    axes[2].set_ylabel(r'displacement (m)')
    axes[2].legend()
    axes[2].set_title('SS')

    # Labels for subplots
    axes[0].text(-0.1, 1.05, '(a)', transform=axes[0].transAxes, fontsize=12, fontweight='bold')
    axes[1].text(-0.1, 1.05, '(b)', transform=axes[1].transAxes, fontsize=12, fontweight='bold')
    axes[2].text(-0.1, 1.05, '(c)', transform=axes[2].transAxes, fontsize=12, fontweight='bold')

    plt.show()

def plot_displacement():
    """Function to plot displacement data."""
    x = np.linspace(0, 1, 500)  # x-axis data
    y1 = 0.02 * np.sin(10 * np.pi * x)  # "optimal" line
    y2 = 0.03 * np.sin(10 * np.pi * x) + 0.005  # "type1" line
    y3 = 0.06 * np.sin(10 * np.pi * x) + 0.01  # "type2" line

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y1, color='red', label='optimal', linewidth=1.5)
    ax.plot(x, y2, color='black', label='type1', linewidth=1.2)
    ax.plot(x, y3, color='blue', label='type2', linewidth=1.0)

    ax.set_xlabel('CF', fontsize=14)
    ax.set_ylabel('displacement(m)', fontsize=14)

    ax.legend(fontsize=12, loc='upper right', frameon=False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_frequency_response():
    """Function to plot frequency response with an inset plot."""
    frequency = np.linspace(0, 1500, 1000)  # Frequency axis in Hz
    G0 = -100 - 20 * np.log10(frequency + 1)  # Gz=0
    G100 = G0 + 10 * np.exp(-((frequency - 100) / 50) ** 2)  # Gz=100
    G1000 = G0 + 20 * np.exp(-((frequency - 500) / 100) ** 2)  # Gz=1000
    G3000 = G0 + 30 * np.exp(-((frequency - 1000) / 200) ** 2)  # Gz=3000

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frequency, G0, 'k-', label=r'$G_z=0$', linewidth=1.5)
    ax.plot(frequency, G100, 'g-', label=r'$G_z=100$', linewidth=1.5)
    ax.plot(frequency, G1000, 'b--', label=r'$G_z=1000$', linewidth=1.5)
    ax.plot(frequency, G3000, 'r--', label=r'$G_z=3000$', linewidth=1.5)

    ax.set_xlim(0, 1500)
    ax.set_ylim(-200, -50)
    ax.set_xlabel('frequency (Hz)', fontsize=12)
    ax.set_ylabel('power (dB)', fontsize=12)
    ax.set_title('CC', loc='right', fontsize=12)
    ax.legend(fontsize=10)

    # Add inset plot
    inset_ax = fig.add_axes([0.4, 0.5, 0.4, 0.35])  # [x, y, width, height] of inset
    inset_ax.plot(frequency, G0, 'k-', linewidth=1.5)
    inset_ax.plot(frequency, G100, 'g-', linewidth=1.5)
    inset_ax.plot(frequency, G1000, 'b--', linewidth=1.5)
    inset_ax.plot(frequency, G3000, 'r--', linewidth=1.5)

    inset_ax.set_xlim(0, 200)
    inset_ax.set_ylim(-120, -80)
    inset_ax.tick_params(axis='both', labelsize=10)

    # Add highlight box
    ax.annotate('', xy=(200, -120), xytext=(0, -200),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10))
    rect = plt.Rectangle((0, -120), 200, 80, edgecolor='black', linestyle='--', fill=False, linewidth=1)
    ax.add_patch(rect)

    ax.text(-0.1, 1.02, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_amplitude_response():
    """Function to plot amplitude response over time."""
    t = np.linspace(0, 0.4, 1000)  # Time from 0 to 0.4 seconds

    # Response data for different cases
    amplitude_uncontrolled = 2 * np.sin(20 * np.pi * t) * np.exp(-5 * t)
    amplitude_one_patch = 1.8 * np.sin(20 * np.pi * t) * np.exp(-5 * t)
    amplitude_two_patches = 1.5 * np.sin(20 * np.pi * t) * np.exp(-5 * t)
    amplitude_three_patches = 1.2 * np.sin(20 * np.pi * t) * np.exp(-5 * t)

    plt.figure(figsize=(10, 6))

    plt.plot(t, amplitude_uncontrolled, label='uncontrolled', color='blue')
    plt.plot(t, amplitude_one_patch, label='with one patch', linestyle=':', color='orange')
    plt.plot(t, amplitude_two_patches, label='with two patches', linestyle='--', color='yellow')
    plt.plot(t, amplitude_three_patches, label='with three patches', linestyle='-.', color='purple')

    plt.xlabel('Time (sec)', fontsize=14)
    plt.ylabel('Amplitude (m)', fontsize=14)
    plt.title('Time response of third mode with different number of patches', fontsize=16)
    plt.legend()
    plt.grid()
    plt.ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.show()
plot_time_response()
plot_displacement()
plot_frequency_response()
plot_amplitude_response()