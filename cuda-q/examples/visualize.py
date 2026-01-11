"""Visualization tools for VQA results."""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_convergence(
    energy_history: List[float],
    exact_energy: Optional[float] = None,
    title: str = "VQA Convergence",
    save_path: Optional[str] = None
):
    """
    Plot energy convergence during optimization.

    Args:
        energy_history: List of energies at each iteration
        exact_energy: Exact ground state energy (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    iterations = range(len(energy_history))
    plt.plot(iterations, energy_history, 'b-', linewidth=2, label='VQA Energy')

    if exact_energy is not None:
        plt.axhline(y=exact_energy, color='r', linestyle='--',
                   linewidth=2, label='Exact Energy')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_error_convergence(
    energy_history: List[float],
    exact_energy: float,
    title: str = "VQA Error Convergence",
    save_path: Optional[str] = None
):
    """
    Plot absolute error convergence (log scale).

    Args:
        energy_history: List of energies at each iteration
        exact_energy: Exact ground state energy
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    errors = [abs(e - exact_energy) for e in energy_history]
    iterations = range(len(errors))

    plt.semilogy(iterations, errors, 'b-', linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Absolute Error (log scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, which='both')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_h2_dissociation_curve(
    bond_distances: np.ndarray,
    vqa_energies: np.ndarray,
    exact_energies: Optional[np.ndarray] = None,
    title: str = "H2 Dissociation Curve",
    save_path: Optional[str] = None
):
    """
    Plot H2 potential energy surface.

    Args:
        bond_distances: Array of bond distances (Angstrom)
        vqa_energies: VQA computed energies
        exact_energies: Exact energies (optional)
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    plt.plot(bond_distances, vqa_energies, 'bo-', linewidth=2,
            markersize=8, label='VQA')

    if exact_energies is not None:
        plt.plot(bond_distances, exact_energies, 'r--', linewidth=2,
                label='Exact')

    plt.xlabel('Bond Distance (Å)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Mark equilibrium distance
    if exact_energies is not None:
        min_idx = np.argmin(exact_energies)
        plt.axvline(x=bond_distances[min_idx], color='g', linestyle=':',
                   alpha=0.5, label=f'Equilibrium ({bond_distances[min_idx]:.2f} Å)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_parameter_landscape(
    cost_function,
    param_ranges: List[Tuple[float, float]],
    optimal_params: np.ndarray,
    title: str = "Energy Landscape",
    resolution: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot 2D energy landscape (for 2-parameter problems).

    Args:
        cost_function: Cost function to evaluate
        param_ranges: List of (min, max) for each parameter
        optimal_params: Optimal parameters found by VQA
        title: Plot title
        resolution: Grid resolution
        save_path: Path to save figure
    """
    if len(param_ranges) != 2:
        print("Parameter landscape plot only supports 2 parameters")
        return

    # Create parameter grid
    p1_vals = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
    p2_vals = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
    P1, P2 = np.meshgrid(p1_vals, p2_vals)

    # Evaluate cost function on grid
    Z = np.zeros_like(P1)
    for i in range(resolution):
        for j in range(resolution):
            params = np.array([P1[i, j], P2[i, j]])
            Z[i, j] = cost_function(params)

    # Plot
    plt.figure(figsize=(10, 8))

    # Contour plot
    contour = plt.contourf(P1, P2, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Energy')

    # Mark optimal point
    plt.plot(optimal_params[0], optimal_params[1], 'r*',
            markersize=20, label='VQA Optimum')

    plt.xlabel('Parameter 1', fontsize=12)
    plt.ylabel('Parameter 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_gradient_norm(
    gradient_history: List[np.ndarray],
    title: str = "Gradient Norm Evolution",
    save_path: Optional[str] = None
):
    """
    Plot evolution of gradient norm during optimization.

    Args:
        gradient_history: List of gradient vectors
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    gradient_norms = [np.linalg.norm(g) for g in gradient_history]
    iterations = range(len(gradient_norms))

    plt.semilogy(iterations, gradient_norms, 'b-', linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Gradient Norm (log scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, which='both')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def create_summary_plot(
    vqa_result: dict,
    exact_energy: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Create comprehensive summary plot with multiple panels.

    Args:
        vqa_result: Dictionary containing VQA results
        exact_energy: Exact ground state energy
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    energy_history = vqa_result.get('energy_history', [])
    iterations = range(len(energy_history))

    # Panel 1: Energy convergence
    axes[0, 0].plot(iterations, energy_history, 'b-', linewidth=2)
    if exact_energy is not None:
        axes[0, 0].axhline(y=exact_energy, color='r', linestyle='--',
                          linewidth=2, label='Exact')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('Energy Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel 2: Error convergence (if exact energy available)
    if exact_energy is not None:
        errors = [abs(e - exact_energy) for e in energy_history]
        axes[0, 1].semilogy(iterations, errors, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Error Convergence (log scale)')
        axes[0, 1].grid(True, alpha=0.3, which='both')

    # Panel 3: Parameter evolution
    param_history = vqa_result.get('parameter_history', [])
    if param_history:
        param_array = np.array(param_history)
        for i in range(min(param_array.shape[1], 5)):  # Plot first 5 params
            axes[1, 0].plot(iterations, param_array[:, i],
                          label=f'θ_{i}', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('Parameter Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    # Panel 4: Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"VQA Summary\n\n"
    summary_text += f"Final Energy: {energy_history[-1]:.8f}\n"
    if exact_energy is not None:
        error = abs(energy_history[-1] - exact_energy)
        summary_text += f"Exact Energy: {exact_energy:.8f}\n"
        summary_text += f"Absolute Error: {error:.8f}\n"
        summary_text += f"Relative Error: {error/abs(exact_energy)*100:.4f}%\n"
    summary_text += f"\nIterations: {len(energy_history)}\n"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                   verticalalignment='center', family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization tools for VQA")
    print("Import this module to use plotting functions")
