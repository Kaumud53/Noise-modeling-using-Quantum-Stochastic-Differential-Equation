# Practical Guide: Parameter Selection and Applications
## Quantum Harmonic Oscillator Noise Modeling

---

## 1. Physical Systems and Typical Parameters

### 1.1 Cavity QED / Circuit QED

**System**: Superconducting qubit in a microwave cavity

**Typical Parameters:**
- Frequency: ω/2π = 4-8 GHz
- Damping rate: γ/2π = 0.1-1 MHz
- Quality factor: Q = 10³ - 10⁶
- Temperature: T = 10-50 mK → n_th ≈ 0.01-0.1
- Dephasing rate: γ_ph/2π = 1-10 MHz

**Lindblad Operators:**
```python
# Energy relaxation (T1 process)
L_relax = sqrt(gamma) * a

# Pure dephasing (T2* process)
L_dephase = sqrt(gamma_ph) * n

# Thermal excitation (small at low T)
L_thermal = sqrt(gamma * n_th) * a_dag
```

**Time Scales:**
- Coherence time T2: 10-100 μs
- Energy relaxation T1: 20-200 μs
- Gate time: 10-50 ns

### 1.2 Trapped Ion

**System**: Single trapped ion in harmonic potential

**Typical Parameters:**
- Trap frequency: ω/2π = 1-10 MHz
- Heating rate: γ/2π = 1-100 Hz
- Quality factor: Q = 10⁴ - 10⁷
- Temperature: T ≈ 0 (Doppler cooled)
- n_th ≈ 0.01 - 0.1 (residual heating)

**Lindblad Operators:**
```python
# Motional heating
L_heating = sqrt(gamma_heat * (n_th + 1)) * a
L_cooling = sqrt(gamma_heat * n_th) * a_dag

# Ambient field noise (dephasing)
L_dephase = sqrt(gamma_ambient) * n
```

### 1.3 Optomechanics

**System**: Mechanical oscillator coupled to optical cavity

**Typical Parameters:**
- Mechanical frequency: ω_m/2π = 100 kHz - 10 MHz
- Mechanical damping: γ_m/2π = 10 Hz - 1 kHz
- Quality factor: Q = 10⁴ - 10⁸
- Bath temperature: T = 4 K - 300 K
- n_th = k_B T / (ħω_m) = 10² - 10⁶

**Lindblad Operators:**
```python
# Mechanical damping into thermal bath
L_mech_loss = sqrt(gamma_m * (n_th + 1)) * a
L_mech_gain = sqrt(gamma_m * n_th) * a_dag

# Measurement-induced dephasing
L_measure = sqrt(gamma_measure) * n
```

---

## 2. How to Choose Parameters for Your Model

### 2.1 Determining Damping Rate γ

**Method 1: From Quality Factor**
```
γ = ω / Q
```
where Q is experimentally measured quality factor.

**Method 2: From Energy Relaxation Time T1**
```
γ = 1 / T1
```

**Method 3: From Cavity Decay Rate κ**
For cavity systems:
```
γ = κ / 2
```

### 2.2 Determining Thermal Photon Number n_th

**From Temperature:**
```python
import numpy as np

def thermal_photon_number(omega, T, hbar=1.0, k_B=1.0):
    """
    Calculate thermal photon number
    
    Parameters:
    -----------
    omega : float
        Oscillator frequency
    T : float
        Temperature (same units as ħω/k_B)
    """
    if T == 0:
        return 0
    x = hbar * omega / (k_B * T)
    if x < 1e-6:  # High temperature limit
        return k_B * T / (hbar * omega)
    else:
        return 1.0 / (np.exp(x) - 1)

# Example: Superconducting qubit
omega = 2 * np.pi * 5e9  # 5 GHz
T = 20e-3  # 20 mK
hbar = 1.055e-34
k_B = 1.381e-23
n_th = thermal_photon_number(omega, T, hbar, k_B)
print(f"n_th = {n_th:.6f}")  # Very small!
```

**From Heating Rate:**
```
n_th = (γ_heat / γ_cool) * (steady state photon number)
```

### 2.3 Determining Dephasing Rate γ_ph

**Method 1: From Coherence Times**
```
1/T2 = 1/(2*T1) + 1/T2*
γ_ph = 1/T2* = 1/T2 - 1/(2*T1)
```
where T2* is pure dephasing time.

**Method 2: From Ramsey Experiments**
Measure decay of coherence in Ramsey sequence:
```
Coherence(t) ∝ exp(-γ_ph * t)
```

**Method 3: From Noise Spectral Density**
If you have 1/f or white noise:
```
γ_ph = ∫ S(ω) dω
```
where S(ω) is the noise power spectral density.

---

## 3. Simulation Best Practices

### 3.1 Choosing Hilbert Space Dimension N

**Rule of thumb:**
```python
# For initial state with mean photon number n_mean
N = max(4 * n_mean + 10, 20)

# For thermal state with n_th
N = max(10 * n_th + 15, 20)

# For coherent state |α⟩
N = max(4 * abs(alpha)**2 + 10, 20)
```

**Verification:**
Check that highest populated Fock state has negligible population:
```python
if rho[-1, -1] > 1e-6:
    print("WARNING: Increase Hilbert space dimension!")
```

### 3.2 Time Step Selection

**For GKSL Master Equation:**
- Use adaptive integrator (solve_ivp with 'RK45' or 'DOP853')
- Check convergence by halving time step

**For Quantum Trajectories:**
```python
# Time step should resolve fastest process
dt_max = 1 / (10 * max(gamma, gamma_ph, omega))

# Number of trajectories for convergence
n_traj = 100  # Minimum
n_traj = 500  # Good statistics
n_traj = 1000  # Publication quality
```

### 3.3 Numerical Stability

**Check these properties at each time step:**
```python
def check_validity(rho):
    """Check if density matrix is physical"""
    # 1. Hermiticity
    if not np.allclose(rho, rho.T.conj()):
        print("WARNING: Non-Hermitian!")
    
    # 2. Trace = 1
    tr = np.trace(rho)
    if abs(tr - 1.0) > 1e-6:
        print(f"WARNING: Trace = {tr:.6f} ≠ 1")
    
    # 3. Positivity (all eigenvalues ≥ 0)
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -1e-10):
        print(f"WARNING: Negative eigenvalue: {eigvals.min():.2e}")
    
    return np.allclose(rho, rho.T.conj()) and abs(tr-1) < 1e-6 and np.all(eigvals > -1e-10)
```

---

## 4. Common Experimental Scenarios

### 4.1 Cavity Ringdown Measurement

**Scenario:** Measure photon decay from cavity

**Model:**
```python
# Initial: Coherent state
psi0 = coherent_state(alpha=3.0, N=30)
rho0 = state_to_density_matrix(psi0)

# Pure damping (vacuum environment)
L_ops = [sqrt(kappa) * a]
rates = [1.0]

# Measure photon number vs time
t, rho_t = solve_gksl(rho0, H, L_ops, rates, t_span)
n_avg = [expectation_value(rho, n_op) for rho in rho_t]

# Fit: n(t) = n0 * exp(-kappa * t)
```

### 4.2 Thermalization Experiment

**Scenario:** Watch system approach thermal equilibrium

**Model:**
```python
# Initial: Fock state |n⟩
psi0 = fock_state(n=5, N=30)
rho0 = state_to_density_matrix(psi0)

# Thermal bath at temperature T
n_th = thermal_photon_number(omega, T)
L_ops = [sqrt(gamma*(n_th+1)) * a, sqrt(gamma*n_th) * a_dag]
rates = [1.0, 1.0]

# Watch approach to n_th
t, rho_t = solve_gksl(rho0, H, L_ops, rates, t_span)
```

### 4.3 Decoherence of Cat State

**Scenario:** Study decoherence of superposition state

**Model:**
```python
# Initial: Cat-like state (superposition)
alpha = 2.0
psi_plus = coherent_state(alpha, N)
psi_minus = coherent_state(-alpha, N)
psi0 = (psi_plus + psi_minus) / np.sqrt(2 + 2*np.real(psi_plus.conj() @ psi_minus))
rho0 = state_to_density_matrix(psi0)

# Combined damping + dephasing
L_ops = [sqrt(gamma) * a, sqrt(gamma_ph) * n_op]
rates = [1.0, 1.0]

# Watch coherence decay
t, rho_t = solve_gksl(rho0, H, L_ops, rates, t_span)
coherence = [abs(rho[0, 1]) for rho in rho_t]  # Off-diagonal element
```

### 4.4 Quantum State Tomography Validation

**Scenario:** Simulate realistic measurements including noise

**Model:**
```python
# Prepare target state
psi_target = coherent_state(alpha=1.5, N=20)
rho_ideal = state_to_density_matrix(psi_target)

# Add realistic noise during measurement
gamma_meas = 0.01  # Measurement-induced dephasing
t_meas = 1.0  # Measurement time

L_ops = [sqrt(gamma_meas) * a, sqrt(gamma_ph) * n_op]
rates = [1.0, 1.0]

# Evolve during measurement
t_span = np.linspace(0, t_meas, 10)
_, rho_measured = solve_gksl(rho_ideal, H, L_ops, rates, t_span)
rho_final = rho_measured[-1]

# Compare ideal vs noisy
F = fidelity(rho_ideal, rho_final)
print(f"Fidelity: {F:.4f}")
```

---

## 5. Advanced Topics

### 5.1 Non-Markovian Effects

When memory effects matter (system-bath correlations):

**Indicators:**
- Bath correlation time τ_B comparable to system dynamics: τ_B ≈ 1/ω
- Non-exponential decay
- Negative decay rates at short times

**Beyond GKSL:**
- Use time-dependent Lindblad rates: γ(t)
- Use non-Markovian master equations (Nakajima-Zwanzig)
- Use full system+bath dynamics

### 5.2 Time-Dependent Noise

**Example: Pulsed noise**
```python
def gksl_rhs_time_dependent(t, rho_vec, H, lindblad_ops, rate_functions):
    """
    rate_functions: list of functions γ_k(t)
    """
    N = H.shape[0]
    rho = rho_vec.reshape((N, N))
    
    drho_dt = -1j * (H @ rho - rho @ H)
    
    for L, gamma_func in zip(lindblad_ops, rate_functions):
        gamma_t = gamma_func(t)
        drho_dt += gamma_t * lindblad_dissipator(L, rho)
    
    return drho_dt.flatten()

# Example: Sinusoidal noise modulation
def gamma_t(t):
    return gamma_0 * (1 + 0.3 * np.sin(omega_noise * t))
```

### 5.3 Multiple Oscillators (Coupled System)

**Two coupled oscillators:**
```python
# Extend Hilbert space: N1 × N2
N1, N2 = 15, 15
N_total = N1 * N2

# Tensor product operators
a1 = np.kron(a, np.eye(N2))  # Acts on oscillator 1
a2 = np.kron(np.eye(N1), a)  # Acts on oscillator 2

# Coupled Hamiltonian
H = omega1 * a1.T.conj() @ a1 + omega2 * a2.T.conj() @ a2 + g * (a1 @ a2.T.conj() + a1.T.conj() @ a2)

# Independent baths
L_ops = [sqrt(gamma1) * a1, sqrt(gamma2) * a2]
```

### 5.4 Continuous Measurement (Quantum Filtering)

Hudson-Parthasarathy with measurement output:
```python
# Homodyne detection
dU_t = {-i*H*dt - (1/2)*L†L*dt + L*dA† - L†*dA - L*dY}U_t

# Measurement record
dY_t = (⟨L + L†⟩_t dt + dW_t) / √η

where:
- dW_t: classical Wiener process (measurement noise)
- η: detection efficiency
```

---

## 6. Troubleshooting Guide

### Problem: Negative populations

**Cause:** Numerical error accumulation
**Solution:**
- Decrease time step
- Use higher-order integrator
- Project onto physical subspace: rho = (rho + rho†)/2; rho = rho / Tr(rho)

### Problem: Simulation too slow

**Causes & Solutions:**
1. Hilbert space too large
   - Use sparse matrices for large N
   - Consider Wigner function approach

2. Too many time steps
   - Use adaptive time stepping
   - Only save outputs at desired times

3. Many quantum trajectories needed
   - Parallelize trajectory loop
   - Use GPU acceleration (CuPy)

### Problem: HP and GKSL don't match

**Causes:**
1. Not enough trajectories
   - Increase n_traj (aim for >500)

2. Time step too large for HP
   - Decrease dt (make dt << 1/γ)

3. Jump probabilities > 1
   - Time step too large: dp_total should be << 1

### Problem: Steady state not reached

**Solutions:**
- Simulate longer (t_max >> 1/γ)
- Check that all decay channels are included
- Verify thermal photon number n_th is correct

---

## 7. Code Optimization Tips

### Use Sparse Matrices for Large Systems

```python
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm as sparse_expm

# Sparse operators
a_sparse = csr_matrix(a)
n_op_sparse = a_sparse.T.conj() @ a_sparse
```

### Vectorization for Multiple Trajectories

```python
# Run trajectories in parallel
from multiprocessing import Pool

def run_single_trajectory(seed):
    np.random.seed(seed)
    return quantum_trajectory(...)

with Pool(processes=8) as pool:
    results = pool.map(run_single_trajectory, range(n_traj))
```

### Memory Management

```python
# Don't store full density matrices if only need observables
n_avg = np.zeros(len(t))
for i, ti in enumerate(t):
    rho = evolve_to_time(ti)
    n_avg[i] = expectation_value(rho, n_op)
    # rho is garbage collected here
```

---

## 8. Validation Checklist

Before trusting your results, verify:

- [ ] Trace preservation: |Tr(ρ) - 1| < 10⁻⁶
- [ ] Hermiticity: ||ρ - ρ†|| < 10⁻⁶
- [ ] Positivity: all eigenvalues ≥ -10⁻¹⁰
- [ ] Known limits match:
  - [ ] T=0: reaches ground state
  - [ ] T→∞: reaches thermal state
  - [ ] γ=0: unitary evolution
- [ ] Convergence tests:
  - [ ] Halve time step → same result
  - [ ] Double N → same result (if physical)
  - [ ] Double n_traj → same result (HP)
- [ ] Physical sanity:
  - [ ] Energy decreases (for T=0)
  - [ ] Entropy increases
  - [ ] Coherences decay

---

## 9. Further Reading

### Books:
1. Breuer & Petruccione - "The Theory of Open Quantum Systems"
2. Gardiner & Zoller - "Quantum Noise"
3. Wiseman & Milburn - "Quantum Measurement and Control"

### Key Papers:
1. Lindblad (1976) - Original GKSL formulation
2. Hudson & Parthasarathy (1984) - Quantum stochastic calculus
3. Dalibard et al. (1992) - Quantum trajectories method

### Software:
- QuTiP (Python) - Quantum optics toolbox
- QuantumOptics.jl (Julia) - High performance
- Qiskit Dynamics - For superconducting qubits

---

## 10. Summary

**Key Points:**
1. GKSL = effective dynamics after tracing out environment
2. HP = microscopic unitary evolution with explicit bath
3. They give identical results for system observables
4. Choose model based on:
   - GKSL: faster, when only care about system
   - HP: when need individual trajectories, measurement records

**When to use each:**
- Use GKSL for: steady states, average dynamics, analytical calculations
- Use HP for: measurement simulation, conditional evolution, real-time feedback
