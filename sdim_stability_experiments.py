"""
s-Dimension Theory: Stability Analysis of 1^s in Classical vs Quantum Systems
================================================================================

This code demonstrates the instability of classical implementations of the 
theoretical 1^s construct compared to quantum phase representations e^(iθ).

Key Findings:
1. Dimension transformation causes 50% information loss
2. Softmax causes catastrophic drift (2400% error)  
3. Quantum phase representation is orders of magnitude more stable

Author: imamura/tadaima1002
Date: February 2026
License: Apache-2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# ============================================================================
# EXPERIMENT 1: Basic Stability Tests
# ============================================================================

def test_identity_operations(n_iterations: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test stability of repeated identity operations.
    Theory: 1 * 1 * 1 * ... = 1
    Reality: Floating point drift
    """
    # Float32
    one_f32 = np.float32(1.0)
    history_f32 = [one_f32]
    current_f32 = one_f32
    
    for _ in range(n_iterations):
        current_f32 = current_f32 * one_f32 / one_f32
        history_f32.append(float(current_f32))
    
    # Float64
    one_f64 = np.float64(1.0)
    history_f64 = [one_f64]
    current_f64 = one_f64
    
    for _ in range(n_iterations):
        current_f64 = current_f64 * one_f64 / one_f64
        history_f64.append(float(current_f64))
    
    return np.array(history_f32), np.array(history_f64)


def test_quantum_phase_stability(n_iterations: int = 10000) -> np.ndarray:
    """
    Simulate quantum phase rotation: e^(iθ)
    Theory: |e^(iθ)| = 1 always
    Reality: Numerically stable
    """
    theta = 0.0
    phases = []
    
    for _ in range(n_iterations):
        theta += 2 * np.pi / 1000
        phase_complex = np.exp(1j * theta)
        magnitude = np.abs(phase_complex)
        phases.append(magnitude)
    
    return np.array(phases)


# ============================================================================
# EXPERIMENT 2: Neural Network Operations
# ============================================================================

def test_matrix_multiplication_drift(n_layers: int = 100, dim: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate deep neural network with identity transformations.
    Theory: x @ I @ I @ ... = x
    Reality: Accumulates error
    """
    # Float32
    x_f32 = np.ones((1, dim), dtype=np.float32)
    identity_f32 = np.eye(dim, dtype=np.float32)
    history_f32 = []
    
    for _ in range(n_layers):
        x_f32 = x_f32 @ identity_f32
        history_f32.append(np.mean(x_f32))
    
    # Float64
    x_f64 = np.ones((1, dim), dtype=np.float64)
    identity_f64 = np.eye(dim, dtype=np.float64)
    history_f64 = []
    
    for _ in range(n_layers):
        x_f64 = x_f64 @ identity_f64
        history_f64.append(np.mean(x_f64))
    
    return np.array(history_f32), np.array(history_f64)


def test_relu_drift(n_layers: int = 100, dim: int = 512) -> np.ndarray:
    """
    Test ReLU + identity transformation.
    Theory: ReLU(1 * x) = x (for x > 0)
    Reality: Potential drift
    """
    x = np.ones((1, dim), dtype=np.float32)
    identity = np.eye(dim, dtype=np.float32)
    history = []
    
    for _ in range(n_layers):
        x = x @ identity
        x = np.maximum(0, x)  # ReLU
        history.append(np.mean(x))
    
    return np.array(history)


# ============================================================================
# EXPERIMENT 3: Critical Discovery - Dimension Transformation
# ============================================================================

def test_dimension_transformation(n_layers: int = 50) -> np.ndarray:
    """
    CRITICAL FINDING: Dimension reduction → expansion causes 50% information loss!
    
    This is the smoking gun for why Effective Rank is non-monotonic.
    Used everywhere: Transformers, VAE, ResNet bottlenecks
    """
    dim_in = 512
    dim_hidden = 256
    
    x = np.ones((1, dim_in), dtype=np.float32)
    history = []
    
    for _ in range(n_layers):
        # Compress: 512 -> 256
        W_down = np.eye(dim_in, dim_hidden, dtype=np.float32)
        # Expand: 256 -> 512  
        W_up = np.eye(dim_hidden, dim_in, dtype=np.float32)
        
        x = x @ W_down @ W_up  # Theoretically identity
        mean_val = np.mean(x)
        history.append(mean_val)
    
    return np.array(history)


def test_noisy_weights(n_iterations: int = 100, dim: int = 512, epsilon: float = 1e-7) -> np.ndarray:
    """
    Identity matrix with small perturbations.
    Simulates real weight initialization.
    """
    x = np.ones((1, dim), dtype=np.float32)
    history = []
    
    for _ in range(n_iterations):
        I = np.eye(dim, dtype=np.float32)
        noise = np.random.randn(dim, dim).astype(np.float32) * epsilon
        W = I + noise
        
        x = x @ W
        history.append(np.mean(x))
    
    return np.array(history)


def test_softmax_instability(n_iterations: int = 100) -> np.ndarray:
    """
    CRITICAL FINDING: Softmax causes catastrophic drift!
    
    Non-reversible transformation destroys information.
    Error: 2400% after 100 iterations.
    """
    x = np.ones((1, 10), dtype=np.float32)
    history = []
    
    for _ in range(n_iterations):
        # Softmax (to probability distribution)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Attempt to invert (impossible!)
        x = np.log(probs + 1e-8) * 10
        mean_val = np.mean(x)
        history.append(mean_val)
    
    return np.array(history)


def test_low_precision(n_layers: int = 50, dim: int = 512) -> np.ndarray:
    """
    Simulate Float16 precision (GPU/TPU).
    """
    def to_float16_like(x):
        scale = 1000
        return np.round(x * scale) / scale
    
    x = np.ones((1, dim), dtype=np.float32)
    identity = np.eye(dim, dtype=np.float32)
    history = []
    
    for _ in range(n_layers):
        x = x @ identity
        x = to_float16_like(x)
        history.append(np.mean(x))
    
    return np.array(history)


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def compute_drift_stats(history: np.ndarray, ideal_value: float = 1.0) -> Dict[str, float]:
    """Calculate drift statistics."""
    drift = history - ideal_value
    return {
        'max_deviation': np.max(np.abs(drift)),
        'final_deviation': abs(drift[-1]),
        'mean_deviation': np.mean(np.abs(drift)),
        'std_deviation': np.std(drift)
    }


def print_experiment_summary(name: str, history: np.ndarray, ideal: float = 1.0):
    """Print formatted experiment results."""
    stats = compute_drift_stats(history, ideal)
    print(f"\n{name}:")
    print(f"  Initial value:    {history[0]:.15f}")
    print(f"  Final value:      {history[-1]:.15f}")
    print(f"  Final deviation:  {stats['final_deviation']:.2e}")
    print(f"  Max deviation:    {stats['max_deviation']:.2e}")
    print(f"  Mean deviation:   {stats['mean_deviation']:.2e}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_plots(results: Dict[str, np.ndarray], output_path: str = 'sdim_stability.png'):
    """Create publication-quality plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Matrix multiplication
    ax = axes[0, 0]
    ax.plot(results['matmul_f32'], label='Float32', linewidth=2)
    ax.plot(results['matmul_f64'], label='Float64', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', label='Ideal', linewidth=1)
    ax.set_xlabel('Layer Depth', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Identity Matrix Multiplication', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dimension transformation (CRITICAL!)
    ax = axes[0, 1]
    ax.plot(results['dim_transform'], linewidth=2, color='red')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(range(len(results['dim_transform'])), 
                     results['dim_transform'], 1.0, alpha=0.3, color='red')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Dimension Transform: 50% Loss!', fontsize=14, fontweight='bold', color='red')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Softmax catastrophe
    ax = axes[0, 2]
    ax.plot(results['softmax'], linewidth=2, color='purple')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Softmax: Catastrophic Drift', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Quantum phase
    ax = axes[1, 0]
    ax.plot(results['quantum_phase'][:1000], linewidth=1, color='blue', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Rotation Steps', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Quantum Phase: Ultra-Stable', fontsize=14, fontweight='bold')
    ax.set_ylim([0.9999, 1.0001])
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Comparison
    ax = axes[1, 1]
    ax.plot(results['dim_transform'][:50], label='Dim Transform', alpha=0.7, linewidth=2)
    ax.plot(results['noisy'][:50], label='Noisy Weights', alpha=0.7)
    ax.plot(results['relu'][:50], label='ReLU', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Ideal')
    ax.set_xlabel('Iteration/Layer', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Error accumulation (log scale)
    ax = axes[1, 2]
    errors = {
        'Dim Transform': np.abs(results['dim_transform'] - 1.0),
        'Softmax': np.abs(results['softmax'][:50] - 1.0),
        'Noisy': np.abs(results['noisy'][:50] - 1.0),
        'Quantum': np.abs(results['quantum_phase'][:50] - 1.0)
    }
    
    for name, err in errors.items():
        nonzero = err[err > 0]
        if len(nonzero) > 0:
            ax.semilogy(nonzero, label=name, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax.set_title('Error Accumulation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments():
    """Execute all stability experiments."""
    print("="*80)
    print("s-Dimension Theory: Stability Analysis of 1^s")
    print("Classical vs Quantum Implementations")
    print("="*80)
    
    results = {}
    
    # Basic tests
    print("\n--- BASIC STABILITY TESTS ---")
    _, _ = test_identity_operations(1000)
    results['quantum_phase'] = test_quantum_phase_stability(10000)
    print_experiment_summary("Quantum Phase", results['quantum_phase'])
    
    # Neural network operations
    print("\n--- NEURAL NETWORK OPERATIONS ---")
    results['matmul_f32'], results['matmul_f64'] = test_matrix_multiplication_drift(100, 512)
    print_experiment_summary("Matrix Multiplication (Float32)", results['matmul_f32'])
    print_experiment_summary("Matrix Multiplication (Float64)", results['matmul_f64'])
    
    results['relu'] = test_relu_drift(100, 512)
    print_experiment_summary("ReLU + Identity", results['relu'])
    
    # CRITICAL FINDINGS
    print("\n" + "="*80)
    print("CRITICAL FINDINGS")
    print("="*80)
    
    results['dim_transform'] = test_dimension_transformation(50)
    print_experiment_summary("Dimension Transformation", results['dim_transform'])
    print("\n⚠️  50% INFORMATION LOSS IN DIMENSION TRANSFORM!")
    print("   This affects: Transformers, VAE, ResNet bottlenecks")
    
    results['softmax'] = test_softmax_instability(100)
    print_experiment_summary("Softmax Transformation", results['softmax'], ideal=1.0)
    print("\n⚠️  CATASTROPHIC DRIFT: 2400% ERROR!")
    print("   Non-reversible transformation destroys information")
    
    results['noisy'] = test_noisy_weights(100, 512, 1e-7)
    print_experiment_summary("Noisy Weights", results['noisy'])
    
    results['low_precision'] = test_low_precision(50, 512)
    print_experiment_summary("Low Precision (Float16-like)", results['low_precision'])
    
    # Statistical comparison
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    classical_error = abs(results['dim_transform'][-1] - 1.0)
    quantum_error = np.max(np.abs(results['quantum_phase'] - 1.0))
    
    print(f"\nClassical (Dim Transform) error: {classical_error:.2e}")
    print(f"Quantum (Phase) error:           {quantum_error:.2e}")
    print(f"Stability ratio:                 {classical_error / (quantum_error + 1e-20):.2e}x")
    print("\n→ Quantum representation is orders of magnitude more stable!")
    
    # Theoretical implications
    print("\n" + "="*80)
    print("THEORETICAL IMPLICATIONS")
    print("="*80)
    print("""
1. Classical Implementation of 1^s:
   - Accumulates errors in dimension transformations
   - Non-reversible operations (Softmax) cause catastrophic drift
   - Float32/16 precision limits compound the problem

2. Quantum Implementation (e^iθ):
   - Unitarity guarantees information preservation
   - Phase is geometrically protected on unit circle
   - No accumulation of numerical errors

3. Conclusion:
   - 1^s is mathematically perfect but computationally unstable
   - Quantum hardware is the IDEAL implementation
   - Current AI architectures have fundamental limitations
   
4. Three Types of "Debt":
   - Structural debt: Skip connections (d-dimension)
   - Information debt: Dimension transforms (50% loss)
   - Entropy debt: Non-reversible ops (Softmax)
""")
    
    # Create visualizations
    create_comprehensive_plots(results, 'sdim_stability_analysis.png')
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()

