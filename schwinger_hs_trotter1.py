
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, SdgGate, RZGate

def dyn_modes_links_exact(N: int):
    S = np.tril(np.ones((N-1, N-1), dtype=float))
    K = S @ S.T
    w, v = np.linalg.eigh(K)               # ascending
    idx = np.argsort(w)[::-1]              # descending
    eDyn = w[idx]
    V = v[:, idx]                          # columns=modes
    U = V[::-1, :].T                       # reverse link index (columns), rows=modes
    for k in range(U.shape[0]):            # alternate phases
        if k % 2 == 1:
            U[k, :] *= -1.0
    return eDyn, U

def hs_field_h(N: int, dt: float, a: float, g: float, eDyn: np.ndarray, uDyn: np.ndarray, y: np.ndarray):
    assert eDyn.shape[0] == N-1 and uDyn.shape == (N-1, N-1) and y.shape[0] == N-1
    pref = np.sqrt(a * g * g / dt)        # sqrt(a g^2 / dt)
    h_links = pref * (np.sqrt(eDyn)[:, None] * uDyn)    # (N-1, N-1)
    h = (h_links.T @ y).astype(float)                   # (N-1,)
    h_full = np.zeros(N, dtype=float)
    h_full[:N-1] = h
    return h_full

def z_half_angles(N: int, dt: float, m: float, h_full: np.ndarray):
    signs = np.array([(-1)**(j+1) for j in range(N)], dtype=float)
    thetaZ = 0.5 * dt * ( m * signs + h_full )
    return thetaZ

def apply_xx_plus_yy_block(circ: QuantumCircuit, qL: int, qR: int, theta_xy: float):
    H, S, Sdg = HGate(), SGate(), SdgGate()
    circ.append(Sdg, [qL]); circ.append(H, [qL])
    circ.append(Sdg, [qR]); circ.append(H, [qR])
    circ.cx(qL, qR)
    circ.append(RZGate(4.0*theta_xy), [qR])
    circ.cx(qL, qR)
    circ.append(H, [qL]); circ.append(S, [qL])
    circ.append(H, [qR]); circ.append(S, [qR])

def trotter_step_symmetric(circ: QuantumCircuit, qubits: list[int], *, dt: float, a: float, g: float, m: float,
                           eDyn: np.ndarray, uDyn: np.ndarray, y: np.ndarray):
    N = len(qubits)
    h_full = hs_field_h(N, dt, a, g, eDyn, uDyn, y)
    thetaZ = z_half_angles(N, dt, m, h_full)
    for j, q in enumerate(qubits):
        circ.rz(thetaZ[j], q)
    theta_xy = dt / (4.0 * a)
    for j in range(N-1):
        apply_xx_plus_yy_block(circ, qubits[j], qubits[j+1], theta_xy)
    for j, q in enumerate(qubits):
        circ.rz(thetaZ[j], q)

def make_trotter_block(N: int, *, dt: float, a: float, g: float, m: float,
                       eDyn: np.ndarray, uDyn: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    circ = QuantumCircuit(N, name=f"HS_step(dt={dt})")
    trotter_step_symmetric(circ, list(range(N)), dt=dt, a=a, g=g, m=m, eDyn=eDyn, uDyn=uDyn, y=y)
    return circ

# ------------------------------
# Convenience: API mirroring your Ldyn/Ham helpers
# ------------------------------
def make_trotter_block_from_y(N: int, y, *, dt: float, a: float, g: float, m: float,
                              modes: str = "links_exact") -> QuantumCircuit:
    """
    Convenience wrapper that mirrors your Ldyn/Ham signature.
    Inputs:
      - N: number of sites/qubits
      - y: length-(N-1) HS vector (iterable), e.g. standard normal draw
      - dt, a, g, m: model parameters (dt > 0 for real-time)
      - modes: "links_exact" (default) uses dyn_modes_links_exact(N)
    Returns:
      QuantumCircuit implementing one symmetric HS Trotter step.
    """
    y = np.asarray(y, dtype=float).reshape(N-1)
    if modes == "links_exact":
        eDyn, uDyn = dyn_modes_links_exact(N)
    else:
        raise ValueError(f"Unknown modes spec '{modes}'")
    return make_trotter_block(N, dt=dt, a=a, g=g, m=m, eDyn=eDyn, uDyn=uDyn, y=y)
