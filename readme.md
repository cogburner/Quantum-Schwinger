# Schwinger Model Simulation

The purpose of this project is to prove a quantum advantage in the simulation of the Schwinger Model.

## Milestones

#### Custom Noise Model

Qiskit has a custom noise model function, which means it is capable of creating a noise model based on the backend hardware. All that's left to do is to fine tune is based on the circuit structure we're running.

#### Custom Transpilation Scheme

The current transpilation of Qiskit is quite inefficient, so a custom transpilation scheme could increase the number of sites that can be simulated. As it turns out, since the Schwinger model doesn't really utilize that many qubits when running its operations so the default transpilation scheme would increase 2 qubit gates when doing swaps.

#### Adaptive Trotterization

Trotterization for the Schwinger model tends to fail at higher site numbers and with coupling on. At the maxima and minima of the trotterization cycle, calculations can quickly deteoriate due to the step size, so adjusting step size based on the error rates would help improve fidelity. 