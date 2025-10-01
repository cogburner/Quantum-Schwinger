
from qiskit import QuantumCircuit
import numpy as np
from schwinger_hs_trotter import dyn_modes_links_exact, make_trotter_block

N   = 6
a   = 1.0
g   = 1.0
m   = 0.5
dt  = 0.02

eDyn, uDyn = dyn_modes_links_exact(N)
rng = np.random.default_rng(7)
y   = rng.standard_normal(N-1)

step = make_trotter_block(N, dt=dt, a=a, g=g, m=m, eDyn=eDyn, uDyn=uDyn, y=y)
circ = QuantumCircuit(N)
for _ in range(4):
    circ = circ.compose(step, list(range(N)), inplace=False)

print(circ.decompose(reps=2))
