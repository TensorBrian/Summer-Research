import pennylane as qml
import numpy as np
from scipy.sparse.linalg import eigsh

# Constants
N = 3  # Number of sites
a = 1  # Lattice spacing
ma = 0.25  # Mass lattice spacing
ga = 0.5  # Coupling lattice spacing
g = ga / a  # Coupling
m = ma / a - g**2 * a / 8  # Mass

# Discretized Schwinger Hamiltonian
def Schwinger_Hamiltonian(N, m=1, g=1, a=1):
    coeffs = []
    ops = []

    # Hopping terms
    for n in range(1, N):
        coeffs.append(1 / (4 * a))
        ops.append(qml.PauliX(n - 1) @ qml.PauliX(n))
        coeffs.append(1 / (4 * a))
        ops.append(qml.PauliY(n - 1) @ qml.PauliY(n))

    # Mass term
    for n in range(1, N + 1):
        coeffs.append((-1) ** n * m / 2)
        ops.append(qml.PauliZ(n - 1))

    # Interaction term
    c_ga = (a * g**2 / 2) * (1 / 4)
    for n in range(1, N):
        for j in range(1, n + 1):
            coeffs.append(c_ga * 2)
            ops.append(qml.Identity(j - 1))
            coeffs.append(c_ga * 2 * (-1) ** j)
            ops.append(qml.PauliZ(j - 1))

    H = qml.Hamiltonian(coeffs, ops)
    return H

# Construct the Hamiltonian
H_test = Schwinger_Hamiltonian(N, m, g, a)


# Convert Hamiltonian to sparse matrix
try:
    H_sparse = H_test.sparse_matrix()
except Exception as e:
    print(f"Error converting Hamiltonian to sparse matrix: {e}")
    exit()

# Numerical ground state energy
try:
    eigenvalues, eigenvectors = eigsh(H_sparse.real, k=1, which='SA')
    ground_state_energy = eigenvalues[0]
    print(f"Ground state energy: {ground_state_energy}")  # Expected ~3.09
except Exception as e:
    print(f"Error calculating eigenvalues: {e}")
