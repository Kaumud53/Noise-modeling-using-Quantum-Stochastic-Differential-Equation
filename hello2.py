import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class QuantumParameterEstimator:
    
    def __init__(self, hamiltonian: np.ndarray, V_operators: List[np.ndarray], 
                 epsilon: float = 0.1, dt: float = 0.01):
        assert np.allclose(hamiltonian, hamiltonian.conj().T), "Hamiltonian must be Hermitian"
        self.H = hamiltonian
        self.V_ops = V_operators
        self.epsilon = epsilon
        self.dt = dt
        self.dim = hamiltonian.shape[0]
        self.num_params = len(V_operators)
        for i, V in enumerate(V_operators):
            assert np.allclose(V, V.conj().T), f"V_operator {i} must be Hermitian"
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        self.eigenprojections = self._compute_eigenprojections()
        ground_idx = np.argmin(self.eigenvalues)
        self.rho_0 = np.outer(self.eigenvectors[:, ground_idx], self.eigenvectors[:, ground_idx].conj())
        self.convergence_history = []
        
    def set_initial_state(self, rho_0: np.ndarray):
        assert np.allclose(rho_0, rho_0.conj().T), "Initial state must be Hermitian"
        assert np.abs(np.trace(rho_0) - 1.0) < 1e-10, "Initial state must be normalized"
        self.rho_0 = rho_0.copy()
        
    def _compute_eigenprojections(self) -> List[np.ndarray]:
        projections = []
        for i in range(self.dim):
            v = self.eigenvectors[:, i:i+1]
            P = v @ v.conj().T
            projections.append(P)
        return projections
    
    def _U0_evolution(self, t: float) -> np.ndarray:
        if abs(t) < 1e-12:
            return np.eye(self.dim, dtype=complex)
        U0 = np.zeros((self.dim, self.dim), dtype=complex)
        for alpha in range(self.dim):
            phase = np.exp(-1j * self.eigenvalues[alpha] * t)
            U0 += phase * self.eigenprojections[alpha]
        return U0
    
    def _construct_noise_potential(self, theta: np.ndarray) -> np.ndarray:
        V = np.zeros((self.dim, self.dim), dtype=complex)
        for k in range(self.num_params):
            V += theta[k] * self.V_ops[k]
        return V
    
    def _first_order_term_single_realization(self, t: float, V: np.ndarray) -> np.ndarray:
        num_steps = max(int(t / self.dt), 1)
        actual_dt = t / num_steps
        dB = np.random.normal(0, np.sqrt(actual_dt), num_steps)
        U1 = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(num_steps):
            tau = (i + 0.5) * actual_dt  
            U0_t_tau = self._U0_evolution(t - tau)
            U0_tau = self._U0_evolution(tau)
            integrand = U0_t_tau @ V @ U0_tau
            U1 += -1j * integrand * dB[i]
        return U1
    
    def _second_order_deterministic_term(self, t: float, V: np.ndarray) -> np.ndarray:
        V2 = V @ V
        num_steps = max(int(t / self.dt), 1)
        actual_dt = t / num_steps
        U2_det = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(num_steps):
            tau = (i + 0.5) * actual_dt
            U0_t_tau = self._U0_evolution(t - tau)
            U0_tau = self._U0_evolution(tau)
            integrand = U0_t_tau @ V2 @ U0_tau
            U2_det += -0.5 * integrand * actual_dt
        return U2_det
    
    def compute_expected_observable(self, observable: np.ndarray, theta: np.ndarray,
                                    t: float, num_realizations: int = 100) -> float:
        V = self._construct_noise_potential(theta)
        V2 = V @ V
        U0_t = self._U0_evolution(t)
        evolved_obs_0 = U0_t.conj().T @ observable @ U0_t
        expectation_0 = np.real(np.trace(self.rho_0 @ evolved_obs_0))
        term1 = np.zeros((self.dim, self.dim), dtype=complex)
        num_steps = max(int(t / self.dt), 1)
        actual_dt = t / num_steps
        for i in range(num_steps):
            tau = (i + 0.5) * actual_dt
            U0_tau = self._U0_evolution(tau)
            U0_t_tau = self._U0_evolution(t - tau)
            integrand = U0_t.conj().T @ observable @ U0_t_tau @ V2 @ U0_tau
            term1 += -0.5 * integrand * actual_dt
        term2 = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(num_steps):
            tau = (i + 0.5) * actual_dt
            U0_tau = self._U0_evolution(tau)
            U0_t_tau = self._U0_evolution(t - tau)
            integrand = U0_tau.conj().T @ V2 @ U0_t_tau.conj().T @ observable @ U0_t
            term2 += -0.5 * integrand * actual_dt
        term3 = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(num_steps):
            tau = (i + 0.5) * actual_dt
            U0_tau = self._U0_evolution(tau)
            U0_t_tau = self._U0_evolution(t - tau)
            integrand = (U0_tau.conj().T @ V @ U0_t_tau.conj().T @ 
                         observable @ U0_t_tau @ V @ U0_tau)
            term3 += integrand * actual_dt
        evolved_obs_2 = term1 + term2 + term3
        expectation_2 = np.real(np.trace(self.rho_0 @ evolved_obs_2))
        total_expectation = expectation_0 + self.epsilon**2 * expectation_2
        return total_expectation
    
    def compute_Q_matrix(self, observable: np.ndarray) -> np.ndarray:
        X = observable
        Q = np.zeros((self.num_params, self.num_params))
        for k in range(self.num_params):
            for m in range(self.num_params):
                Vk = self.V_ops[k]
                Vm = self.V_ops[m]
                Q_km = 0.0
                for alpha in range(self.dim):
                    P_alpha = self.eigenprojections[alpha]
                    term1 = np.trace(self.rho_0 @ P_alpha @ X @ P_alpha @ Vk @ Vm @ P_alpha)
                    term2 = np.trace(self.rho_0 @ P_alpha @ Vk @ Vm @ P_alpha @ X @ P_alpha)
                    term3 = np.trace(self.rho_0 @ P_alpha @ Vk @ P_alpha @ X @ P_alpha @ Vm @ P_alpha)
                    term4 = 0.0
                    for beta in range(self.dim):
                        P_beta = self.eigenprojections[beta]
                        term4 += np.trace(self.rho_0 @ P_alpha @ Vk @ P_beta @ X @ P_beta @ Vm @ P_alpha)
                    Q_km += np.real(term1 + term2 - term3 - term4)
                Q[k, m] = Q_km
        return Q
    
    def time_averaged_measurement(self, observable: np.ndarray, theta: np.ndarray,
                                  t_min: float = 5.0, t_max: float = 20.0, 
                                  num_times: int = 20) -> float:
        times = np.linspace(t_min, t_max, num_times)
        measurements = []
        for t in times:
            E_evolved = self.compute_expected_observable(observable, theta, t)
            U0_t = self._U0_evolution(t)
            evolved_0 = U0_t.conj().T @ observable @ U0_t
            E_0 = np.real(np.trace(self.rho_0 @ evolved_0))
            scaled_diff = -2.0 / (t * self.epsilon**2) * (E_evolved - E_0)
            measurements.append(scaled_diff)
        return np.mean(measurements)
    
    def estimate_parameters(self, observables: List[np.ndarray], 
                            theta_true: np.ndarray = None,
                            initial_guess: np.ndarray = None,
                            tol: float = 1e-5,
                            verbose: bool = True) -> Tuple[np.ndarray, Dict]:
        if initial_guess is None:
            initial_guess = np.random.uniform(0.1, 0.3, self.num_params)
        Q_list = [self.compute_Q_matrix(obs) for obs in observables]
        if theta_true is not None:
            measurements = [self.time_averaged_measurement(obs, theta_true, 
                                                           t_min=8.0, t_max=15.0, 
                                                           num_times=10) for obs in observables]
        else:
            raise ValueError("theta_true required for simulation")
        if verbose:
            print("="*70)
            print("ALGORITHM 1: Quantum Parameter Estimation (CORRECTED)")
            print("="*70)
            print(f"Parameters to estimate: {self.num_params}")
            print(f"Number of observables: {len(observables)}")
            print(f"Perturbation ε: {self.epsilon}")
            print(f"Initial guess: {initial_guess}")
            print(f"True values: {theta_true}")
            print(f"Measurements: {measurements}")
            print()

        def loss(theta):
            l = 0.0
            for i, Q in enumerate(Q_list):
                pred = theta @ Q @ theta
                l += (pred - measurements[i]) ** 2
            return l

        def callback(theta):
            change = np.linalg.norm(theta - initial_guess)  # Dummy, or compute something
            self.convergence_history.append(loss(theta))
        
        result = minimize(loss, initial_guess, method='BFGS', tol=tol, callback=callback)
        theta = result.x
        info = {
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'Q_list': Q_list,
            'measurements': measurements,
            'final_predictions': [theta @ Q @ theta for Q in Q_list],
            'convergence_history': self.convergence_history
        }
        
        if verbose:
            print("\n" + "="*70)
            print("ESTIMATION COMPLETE")
            print("="*70)
            print(f"Final estimate: {theta}")
            print(f"True values:    {theta_true}")
            print(f"Absolute error: {np.linalg.norm(theta - theta_true):.6f}")
            print(f"Relative error: {np.linalg.norm(theta - theta_true) / np.linalg.norm(theta_true) * 100:.2f}%")
            print(f"Measurements: {measurements}")
            print(f"Predictions:  {info['final_predictions']}")
            print("="*70)
        return theta, info

def run_corrected_simulation():
    print("\n" + "="*70)
    print("CORRECTED SIMULATION: Two-Level Quantum System")
    print("="*70 + "\n")
    omega = 1.0
    H = omega / 2 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    V_operators = [sigma_x, sigma_y]
    theta_true = np.array([0.25, 0.15])
    print("System Configuration:")
    print(f"  Hamiltonian: H = (ω/2) σ_z, ω = {omega}")
    print(f"  Noise operators: V_1 = σ_x, V_2 = σ_y")
    print(f"  True parameters: θ = {theta_true}")
    print(f"  Noise potential: V = {theta_true[0]:.2f} σ_x + {theta_true[1]:.2f} σ_y")
    print()
    estimator = QuantumParameterEstimator(
        hamiltonian=H,
        V_operators=V_operators,
        epsilon=0.15, 
        dt=0.02
    )
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    observables = [sigma_z, sigma_x]
    print("Measurement Setup:")
    print(f"  Initial state: ground state")
    print(f"  Observables: X = σ_z, σ_x")
    print()
    theta_estimated, info = estimator.estimate_parameters(
        observables=observables,
        theta_true=theta_true,
        initial_guess=np.array([0.4, 0.4]),
        tol=5e-3,
        verbose=True
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(info['convergence_history'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss (log scale)', fontsize=12)
    axes[0].set_title('Convergence History', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    params = ['θ₁', 'θ₂']
    x = np.arange(len(params))
    width = 0.35

    axes[1].bar(x - width/2, theta_true, width, label='True', alpha=0.8, color='green')
    axes[1].bar(x + width/2, theta_estimated, width, label='Estimated', alpha=0.8, color='blue')
    axes[1].set_ylabel('Parameter Value', fontsize=12)
    axes[1].set_title('Parameter Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(params)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('algorithm1_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()
    return theta_estimated, theta_true, info

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    theta_est, theta_true, info = run_corrected_simulation()