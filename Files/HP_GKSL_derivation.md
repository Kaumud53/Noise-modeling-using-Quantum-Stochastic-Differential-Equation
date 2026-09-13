# Connection Between Hudson-Parthasarathy and GKSL Master Equations
## For Quantum Harmonic Oscillator with Thermal Noise

---

## 1. Mathematical Preliminaries

### 1.1 Quantum Harmonic Oscillator

The Hamiltonian for a quantum harmonic oscillator:

$$H = \hbar\omega a^\dagger a$$

where:
- $\omega$ is the oscillator frequency
- $a, a^\dagger$ satisfy $[a, a^\dagger] = 1$
- Number operator: $n = a^\dagger a$
- Energy eigenvalues: $E_n = \hbar\omega n$

### 1.2 Quantum Itô Calculus Rules

The quantum noise processes $A_t$ (annihilation) and $A_t^\dagger$ (creation) satisfy:

$$dA_t \, dA_t^\dagger = dt$$
$$dA_t \, dA_t = 0$$
$$dA_t^\dagger \, dA_t^\dagger = 0$$
$$dA_t^\dagger \, dA_t = 0$$

These are the **quantum Itô rules** that replace classical Itô calculus.

---

## 2. Hudson-Parthasarathy Equation

### 2.1 General Form

The Hudson-Parthasarathy quantum stochastic differential equation describes the unitary evolution of a system coupled to a quantum field:

$$dU_t = \left\{-\left(\frac{i}{\hbar}H + \frac{1}{2}L^\dagger L\right)dt + L\,dA_t^\dagger - L^\dagger\,dA_t\right\}U_t$$

with initial condition $U_0 = I$ (identity).

**Key components:**
- $U_t$: Unitary evolution operator
- $H$: System Hamiltonian
- $L$: System-environment coupling operator (Lindblad operator)
- $A_t, A_t^\dagger$: Quantum Wiener processes representing the bath

### 2.2 Physical Interpretation

The HP equation describes:
1. **Hamiltonian evolution**: $-\frac{i}{\hbar}H\,dt$ term
2. **Dissipation**: $-\frac{1}{2}L^\dagger L\,dt$ term (Lamb shift)
3. **Quantum noise input**: $L\,dA_t^\dagger$ (environment acts on system)
4. **Quantum noise output**: $-L^\dagger\,dA_t$ (system acts on environment)

### 2.3 For Quantum Harmonic Oscillator

For a QHO with amplitude damping at rate $\gamma$:

$$H = \hbar\omega a^\dagger a$$
$$L = \sqrt{\gamma} a$$

The HP equation becomes:

$$dU_t = \left\{-i\omega a^\dagger a\,dt - \frac{\gamma}{2}a^\dagger a\,dt + \sqrt{\gamma}\,a\,dA_t^\dagger - \sqrt{\gamma}\,a^\dagger\,dA_t\right\}U_t$$

---

## 3. Deriving GKSL from Hudson-Parthasarathy

### 3.1 Setup

The total system is:
- **System**: Quantum harmonic oscillator with state $\rho_S$
- **Environment**: Boson field initially in vacuum state $|0\rangle_E$

Total initial state:
$$\rho_{tot}(0) = \rho_S(0) \otimes |0\rangle\langle0|_E$$

### 3.2 Evolution of Total State

The total state evolves unitarily:

$$\rho_{tot}(t) = U_t \rho_{tot}(0) U_t^\dagger$$

The reduced system state is:

$$\rho_S(t) = \text{Tr}_E[\rho_{tot}(t)] = \text{Tr}_E[U_t (\rho_S(0) \otimes |0\rangle\langle0|_E) U_t^\dagger]$$

### 3.3 Differential Evolution

Taking the differential:

$$d\rho_{tot} = dU_t \cdot \rho_{tot} \cdot U_t^\dagger + U_t \cdot \rho_{tot} \cdot dU_t^\dagger + dU_t \cdot \rho_{tot} \cdot dU_t^\dagger$$

where the last term comes from quantum Itô calculus.

### 3.4 Calculating $dU_t$

$$dU_t = \left\{-\left(\frac{i}{\hbar}H + \frac{1}{2}L^\dagger L\right)dt + L\,dA_t^\dagger - L^\dagger\,dA_t\right\}U_t$$

$$dU_t^\dagger = U_t^\dagger\left\{\left(\frac{i}{\hbar}H - \frac{1}{2}L^\dagger L\right)dt + L^\dagger\,dA_t - L\,dA_t^\dagger\right\}$$

### 3.5 Computing the Product $dU_t \cdot dU_t^\dagger$

Using quantum Itô rules ($dA_t \, dA_t^\dagger = dt$):

$$dU_t \cdot dU_t^\dagger = U_t \cdot L \, L^\dagger \, dt \cdot U_t^\dagger$$

### 3.6 Full Differential

$$d\rho_{tot} = \left\{-\left(\frac{i}{\hbar}H + \frac{1}{2}L^\dagger L\right)dt + L\,dA_t^\dagger - L^\dagger\,dA_t\right\}U_t \rho_{tot} U_t^\dagger$$
$$+ U_t \rho_{tot} U_t^\dagger\left\{\left(\frac{i}{\hbar}H - \frac{1}{2}L^\dagger L\right)dt + L^\dagger\,dA_t - L\,dA_t^\dagger\right\}$$
$$+ U_t L \rho_{tot} L^\dagger U_t^\dagger \, dt$$

### 3.7 Tracing Over Environment

For the system in vacuum environment: $\langle 0|dA_t|0\rangle = 0$ and $\langle 0|dA_t^\dagger|0\rangle = 0$

After tracing over environment:

$$d\rho_S = -\frac{i}{\hbar}[H, \rho_S]dt + \left(L\rho_S L^\dagger - \frac{1}{2}\{L^\dagger L, \rho_S\}\right)dt$$

where $\{A, B\} = AB + BA$ is the anticommutator.

### 3.8 Final Result: GKSL Master Equation

$$\boxed{\frac{d\rho_S}{dt} = -\frac{i}{\hbar}[H, \rho_S] + \mathcal{L}[\rho_S]}$$

where the Lindbladian superoperator is:

$$\boxed{\mathcal{L}[\rho] = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}}$$

---

## 4. Including Thermal Noise (Finite Temperature)

### 4.1 Thermal Bath

At finite temperature $T$, the environment is in a thermal state with mean photon number:

$$n_{th} = \frac{1}{e^{\hbar\omega/k_B T} - 1}$$

### 4.2 Modified HP Equation

For thermal bath, we need TWO Lindblad operators:

$$L_1 = \sqrt{\gamma(n_{th}+1)} \, a \quad \text{(loss/cooling)}$$
$$L_2 = \sqrt{\gamma n_{th}} \, a^\dagger \quad \text{(gain/heating)}$$

The HP equations become coupled:

$$dU_t^{(1)} = \left\{-\frac{i}{\hbar}H\,dt - \frac{\gamma(n_{th}+1)}{2}a^\dagger a\,dt + \sqrt{\gamma(n_{th}+1)}\,a\,dA_t^{(1)\dagger} - \sqrt{\gamma(n_{th}+1)}\,a^\dagger\,dA_t^{(1)}\right\}U_t^{(1)}$$

$$dU_t^{(2)} = \left\{-\frac{\gamma n_{th}}{2}a a^\dagger\,dt + \sqrt{\gamma n_{th}}\,a^\dagger\,dA_t^{(2)\dagger} - \sqrt{\gamma n_{th}}\,a\,dA_t^{(2)}\right\}U_t^{(2)}$$

### 4.3 Thermal GKSL Equation

Following the same derivation procedure, we get:

$$\boxed{\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \gamma(n_{th}+1)\mathcal{D}[a](\rho) + \gamma n_{th}\mathcal{D}[a^\dagger](\rho)}$$

where $\mathcal{D}[L](\rho) = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$

**Physical interpretation:**
- First term: Photon loss at enhanced rate $(n_{th}+1)$ (spontaneous + stimulated emission)
- Second term: Photon gain at rate $n_{th}$ (thermal absorption)
- At $T=0$: $n_{th} = 0$, only loss term survives
- At $T \to \infty$: $n_{th} \to \infty$, heating dominates

### 4.4 Equilibrium State

The steady-state solution of the thermal GKSL equation is:

$$\rho_{ss} = \frac{1}{1+n_{th}}\sum_{n=0}^{\infty}\left(\frac{n_{th}}{1+n_{th}}\right)^n |n\rangle\langle n|$$

This is a **thermal state** with mean photon number $n_{th}$.

---

## 5. Different Noise Models

### 5.1 Pure Amplitude Damping (Zero Temperature)

$$L = \sqrt{\gamma}a$$
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \gamma\left(a\rho a^\dagger - \frac{1}{2}\{a^\dagger a, \rho\}\right)$$

**Effect**: System decays to ground state $|0\rangle$

### 5.2 Thermal Damping (Finite Temperature)

$$L_1 = \sqrt{\gamma(n_{th}+1)}a, \quad L_2 = \sqrt{\gamma n_{th}}a^\dagger$$
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \gamma(n_{th}+1)\mathcal{D}[a](\rho) + \gamma n_{th}\mathcal{D}[a^\dagger](\rho)$$

**Effect**: System approaches thermal equilibrium

### 5.3 Pure Dephasing

$$L = \sqrt{\gamma_{ph}}a^\dagger a$$
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \gamma_{ph}\left(a^\dagger a\,\rho\, a^\dagger a - \frac{1}{2}\{(a^\dagger a)^2, \rho\}\right)$$

**Effect**: Destroys coherences without changing populations

### 5.4 Combined Damping and Dephasing

$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \gamma(n_{th}+1)\mathcal{D}[a](\rho) + \gamma n_{th}\mathcal{D}[a^\dagger](\rho) + \gamma_{ph}\mathcal{D}[a^\dagger a](\rho)$$

**Effect**: Realistic model including both amplitude and phase noise

---

## 6. Properties and Conservation Laws

### 6.1 Trace Preservation

The GKSL equation preserves trace:
$$\frac{d}{dt}\text{Tr}[\rho] = 0$$

**Proof**: Using cyclic property of trace:
$$\text{Tr}[L\rho L^\dagger] = \text{Tr}[L^\dagger L \rho]$$

### 6.2 Positivity Preservation

If $\rho(0) \geq 0$, then $\rho(t) \geq 0$ for all $t > 0$.

This is guaranteed by the Lindblad form.

### 6.3 Complete Positivity

The evolution map $\mathcal{E}_t: \rho(0) \mapsto \rho(t)$ is **completely positive**, meaning it remains positive even when extended to larger systems.

---

## 7. Time Scales and Physical Parameters

### 7.1 Characteristic Time Scales

- **Oscillation period**: $T = 2\pi/\omega$
- **Damping time**: $\tau_{damp} = 1/\gamma$
- **Dephasing time**: $\tau_{ph} = 1/\gamma_{ph}$
- **Thermal time**: $\tau_{th} = 1/(\gamma n_{th})$

### 7.2 Quality Factor

$$Q = \omega\tau_{damp} = \frac{\omega}{\gamma}$$

High-Q oscillators: $Q \gg 1$ (weak damping)
Low-Q oscillators: $Q \lesssim 1$ (overdamped)

### 7.3 Temperature Regimes

- **Quantum regime**: $k_B T \ll \hbar\omega$ → $n_{th} \approx 0$
- **Thermal regime**: $k_B T \gg \hbar\omega$ → $n_{th} \approx k_B T/\hbar\omega$

---

## 8. Observable Dynamics

### 8.1 Mean Photon Number

For pure damping:
$$\langle n(t) \rangle = \langle a^\dagger a \rangle = e^{-\gamma t}\langle n(0) \rangle$$

For thermal damping:
$$\langle n(t) \rangle = e^{-\gamma t}\langle n(0) \rangle + n_{th}(1 - e^{-\gamma t})$$

### 8.2 Coherences

Off-diagonal elements decay as:
$$\rho_{mn}(t) = e^{-i\omega(m-n)t} e^{-\gamma(m+n)t/2} e^{-\gamma_{ph}(m-n)^2 t} \rho_{mn}(0)$$

---

## 9. Summary of Key Results

| Equation | Domain | Description |
|----------|--------|-------------|
| Hudson-Parthasarathy | Unitary evolution | System + environment (microscopic) |
| GKSL Master Equation | Reduced dynamics | System only (effective) |
| Connection | Partial trace | GKSL = Tr_env[HP] |
| Physical content | Equivalent | Same predictions for system observables |

**Key insight**: The HP equation provides the microscopic quantum description, while the GKSL equation gives the reduced effective dynamics. They are mathematically equivalent for system observables.

