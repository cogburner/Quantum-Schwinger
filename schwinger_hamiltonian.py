import numpy as np
import functools as ft
from typing import List
from collections import deque
from qiskit.quantum_info import SparsePauliOp 

# define operators 
I_plus_op = SparsePauliOp(['I', 'Z'], coeffs=[0.5, 0.5])
I_minus_op = SparsePauliOp(['I', 'Z'], coeffs=[0.5, -0.5])
sigma_minus_op = SparsePauliOp(['X', 'Y'], coeffs=[0.5, -0.5 * 1j])
sigma_plus_op = SparsePauliOp(['X', 'Y'], coeffs=[0.5, 0.5 * 1j])

boson_1 = sigma_minus_op # annihilation operator for 1 qubit, p = 2

def sum_pauli_ops(pauli_ops: List) -> SparsePauliOp:
    """sum of pauli operators from the list

    Args:
        pauli_ops (List): _description_

    Returns:
        SparsePauliOp: _description_
    """
    num_qubits = pauli_ops[0].num_qubits
    # initialize zero operator
    op_out = SparsePauliOp(['I' * num_qubits], coeffs=[0])
    for op in pauli_ops:
        op_out = op_out + op.simplify()
        op_out = op_out.simplify()
    return op_out.simplify()

def tensor_pauli_ops(pauli_ops: List) -> SparsePauliOp:
    """tensor produce of operators from list
    [k_0, k_1, ..., k_n]

    Args:
        pauli_ops (List): _description_

    Returns:
        SparsePauliOp: k_0 ^ k_1 ... ^ k_n
    """
    # initialize the operator
    op_out = pauli_ops[0]
    for op in pauli_ops[1:]:
        op_out = op_out ^ op
        op_out = op_out.simplify()
    num_qubits = [op.num_qubits for op in pauli_ops]
    assert op_out.num_qubits == sum(num_qubits)
    return op_out

# binary encoding
def call_recursion_relation(boson_lower):
    t = boson_lower.num_qubits
    ops_to_sum = []
    for i in range(1, 2**t-1+1):
        ops_to_sum.append(np.sqrt(i) * I_plus_op ^ boson_lower)
    ops_to_tensor = []
    for i in range(1, t+1+1):
        if i != 1:
            ops_to_tensor.append(sigma_plus_op)
        else:
            ops_to_tensor.append(sigma_minus_op)
    ops_to_sum.append(np.sqrt(2**t) * tensor_pauli_ops(ops_to_tensor))
    for i in range(2 ** t +1, 2**(t+1)):
        ops_to_sum.append(np.sqrt(i) * I_minus_op ^ boson_lower)
    # print(ops_to_sum)
    boson_creation_op = sum_pauli_ops(ops_to_sum)
    return boson_creation_op.simplify()

def make_creation_operator(num_occupation):
    num_qubits = int(np.ceil(np.log2(num_occupation))) # number of qubits needed to represent p states
    if num_qubits == 1:
        return boson_1
    else:
        creation_ops = [boson_1]
        for _ in range(1,num_qubits):
            creation_ops.append(call_recursion_relation(creation_ops[-1]))
        return creation_ops[-1]

# unary encoding
# expand sigma_plus or simga_minus to all sites
def expand_operator_to_all_sites(operator: SparsePauliOp, n: int, n_tol: int) -> SparsePauliOp:
    """expand operator acts on sigle site n to all sites

    Args:
        operator (SparsePauliOp): single-qubit operator
        n (int): site that the operator acts on
        n_tol (int): total number of physical qubits
    """
    iden_op = SparsePauliOp('I')
    op_list = [iden_op if i != n else operator for i in range(n_tol) ]
    return tensor_pauli_ops(op_list)

def make_creation_operator_unary(num_occupation: int):
    """implement Eq. 2.59 of https://arxiv.org/pdf/quant-ph/0512209

    Args:
        num_occupation (int): number of occupation
    """
    # total number of qubits needed
    n_tol = num_occupation + 1
    op_list = []
    for n in range(num_occupation):
        sigma_minus_n = expand_operator_to_all_sites(sigma_minus_op, n, n_tol)
        sigma_plus_n_1 = expand_operator_to_all_sites(sigma_plus_op, n+1, n_tol)
        op = np.sqrt(n+1) * sigma_minus_n @ sigma_plus_n_1
        op_list.append(op)
    return sum_pauli_ops(op_list)

def make_number_operator_unary(num_occupation: int):
    n_tol = num_occupation + 1
    count_op = SparsePauliOp(['I', 'Z'], coeffs=[0.5, -0.5])
    op_list = []
    for n in range(num_occupation+1):
        op = n * expand_operator_to_all_sites(count_op, n, n_tol)
        op_list.append(op)
    return sum_pauli_ops(op_list)

def generate_weight_one_pauli(pauli_label, num_qubits):
    pauli_labels = []
    wt1 = deque(["I"] * (num_qubits - 1) + [pauli_label])
    for i in range(num_qubits):
        wt1_term = "".join(literal for literal in wt1)
        pauli_labels.append(wt1_term)
        wt1.rotate(-1)
    return pauli_labels


def create_spin_hamiltonian(num_spins: int, num_bosons: int, number_occupation_states:int, deltas: List[float], method='binary') -> SparsePauliOp:
    if method == 'binary':
        num_boson_qubits = int(np.ceil(np.log2(num_bosons * number_occupation_states)))
    if method == 'unary':
        num_boson_qubits = num_bosons * (number_occupation_states+1)
    devide_int = int(num_spins/2)
    paulis = generate_weight_one_pauli("Z", num_spins)
    paulis_left = [pauli[:devide_int] for pauli in paulis]
    paulis_right = [pauli[devide_int:] for pauli in paulis]
    extended_paulis = [paulis_left[i] + 'I' * num_boson_qubits + paulis_right[i] for i in range(num_spins)]
    return SparsePauliOp(extended_paulis, coeffs=[delta/2 for delta in deltas])

def create_boson_hamiltonian(number_occupation_states, omega, method='binary', simply=True):
    if method == 'binary':
        a_dagger_op = make_creation_operator(number_occupation_states)
        a_op = a_dagger_op.adjoint()
        boson_ham = omega * a_dagger_op @ a_op
    elif method == 'unary':
        # a_dagger_op = make_creation_operator_unary(number_occupation_states)
        boson_ham = omega * make_number_operator_unary(number_occupation_states)
    else:
        raise ValueError('Only binary and unary encoding are supported')
    if simply:
        return boson_ham.simplify()
    else:
        return boson_ham


def create_coupling_hamiltonian(num_spins, number_occupation_states, coupling_gs, method='binary', rwa=False):
    if method == 'binary':
        a_dagger_op = make_creation_operator(number_occupation_states)
    elif method == 'unary':
        a_dagger_op = make_creation_operator_unary(number_occupation_states)
    else:
        raise ValueError('Only binary and unary encoding are supported')
    a_op = a_dagger_op.adjoint()
    devide_int = int(num_spins/2)
    ops_to_sum = []
    if not rwa:
        # sigma_x \otimes N
        spin_coupling_terms = generate_weight_one_pauli("X", num_spins)
        for i, pauli_label in enumerate(spin_coupling_terms):
            # does the ordering matter?
            devide_int = int(num_spins/2)
            pauli_label_left = pauli_label[:devide_int]
            pauli_label_right = pauli_label[devide_int:]
            spin_op_left = SparsePauliOp(pauli_label_left, coeffs=[coupling_gs[i]/(2*np.sqrt(num_spins))])
            spin_op_right = SparsePauliOp(pauli_label_right, coeffs=[1])
            boson_op = a_op.simplify() + a_dagger_op.simplify()
            coupling_op =  spin_op_left ^ boson_op ^ spin_op_right
            ops_to_sum.append(coupling_op.simplify())
    else:
        divide_int = int(num_spins/2)
        # sigma_-*a^dagger
        for i in range(num_spins):
            if i < devide_int:
                # spin on the left of boson
                op = coupling_gs[i]/(2*np.sqrt(num_spins)) * expand_operator_to_all_sites(sigma_minus_op, i, divide_int) ^ a_dagger_op ^ SparsePauliOp(['I'*(num_spins-divide_int)])
            else:
                op = coupling_gs[i]/(2*np.sqrt(num_spins)) * SparsePauliOp(['I'*divide_int]) ^ a_dagger_op ^ expand_operator_to_all_sites(sigma_minus_op, i-divide_int, num_spins-divide_int) 
            ops_to_sum.append(op.simplify())
            ops_to_sum.append(op.adjoint())
    return sum_pauli_ops(ops_to_sum)

def create_total_hamiltonian(num_spins, num_bosons, number_occupation_states, deltas, omega, coupling_gs, method='binary', rwa=False):
    divide_int = int(num_spins/2)
    H_spin = create_spin_hamiltonian(num_spins,num_bosons, number_occupation_states, deltas, method=method)
    H_boson = create_boson_hamiltonian(number_occupation_states, omega, method=method)
    H_coupling = create_coupling_hamiltonian(num_spins, number_occupation_states, coupling_gs, method=method, rwa=rwa)
    H_total = sum_pauli_ops([H_spin, SparsePauliOp(['I'* divide_int]) ^ H_boson ^ SparsePauliOp(['I'* (num_spins-divide_int)]), H_coupling])
    return H_total.simplify()



#def create_lindblad_dissipator(number_occupation_states, method='binary'):
#    if method == 'binary':
#        a_dagger_op = make_creation_operator(number_occupation_states)
#    elif method == 'unary':
#        a_dagger_op = make_creation_operator_unary(number_occupation_states)
#    else:
#        raise ValueError('Only binary and unary encoding are supported')
#    a_op = a_dagger_op.adjoint()
#    L_ops = []
#   for i in number_occupation_states:
        



#L_ops = []
#L_sig = []
#for i in qubits:
#    X = x_ops[i]
#    Y = y_ops[i]
#    L_ops.append(np.sqrt(Gamma) * 0.5 * (X + 1j * Y))


# define operators 
#I_plus_op = SparsePauliOp(['I', 'Z'], coeffs=[0.5, 0.5])
#I_minus_op = SparsePauliOp(['I', 'Z'], coeffs=[0.5, -0.5])
#sigma_minus_op = SparsePauliOp(['X', 'Y'], coeffs=[0.5, -0.5 * 1j])
#sigma_plus_op = SparsePauliOp(['X', 'Y'], coeffs=[0.5, 0.5 * 1j])

def pauli_list(num, type='I'):
    op_list = []
    for i in range(num):
        op_list.append(SparsePauliOp([type], coeffs=[1.0]))
    return op_list

def tensor_pauli_ops(pauli_ops: List) -> SparsePauliOp:
    """tensor produce of operators from list
    [k_0, k_1, ..., k_n]

    Args:
        pauli_ops (List): _description_

    Returns:
        SparsePauliOp: k_0 ^ k_1 ... ^ k_n
    """
    if len(pauli_ops) == 0:
        return 1
    else:
        # initialize the operator
        op_out = pauli_ops[0]
        for op in pauli_ops[1:]:
            op_out = op_out ^ op
            op_out = op_out.simplify()
        num_qubits = [op.num_qubits for op in pauli_ops]
        assert op_out.num_qubits == sum(num_qubits)
        return op_out

# Jordan-Wigner operators
def JWCreateOp(num_sites, pos):
    if pos==1:
        return  np.kron(sigma_minus_op, np.identity(2**(num_sites-1)))
    else:
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        #Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        return ft.reduce(np.kron, [Tensored_Paulis_Left, sigma_minus_op, np.identity(2**(num_sites-pos))])
            
def JWAnnihOp(num_sites, pos):
    if pos==1:
        return  np.kron(sigma_plus_op, np.identity(2**(num_sites-1)))
    else:
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        #Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        return ft.reduce(np.kron, [Tensored_Paulis_Left, sigma_plus_op, np.identity(2**(num_sites-pos))])
    
def JWCreateOp_new(num_sites, pos):
    if pos==1:
        return np.kron(sigma_plus_op, np.identity(2**(num_sites-1)))
    else:
        sigma_plus_op_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(pos-1, 'I')), sigma_plus_op, tensor_pauli_ops(pauli_list(num_sites-pos, 'I'))]) 
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        #return ((-1j)**(pos-1))*ft.reduce(np.kron, [Tensored_Paulis_Left, sigma_plus_op, np.identity(2**(num_sites-pos))])
        return ((-1j)**(pos-1))*ft.reduce(np.kron, [sigma_plus_op_n, Tensored_Paulis_Right])
        #Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        #coeff = ((-1j)**(pos-1))
        #matrix_part = ft.reduce(np.kron, [np.identity(2**(pos-1)), sigma_plus_op, Tensored_Paulis_Right])
        #return coeff * matrix_part
        #return ft.reduce(np.kron, [np.identity(2**(pos-1)), sigma_plus_op, np.identity(2**(num_sites-pos))])
    
def JWAnnihOp_new(num_sites, pos):
    if pos==1:
        return np.kron(sigma_minus_op, np.identity(2**(num_sites-1)))
    else:
        sigma_minus_op_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(pos-1, 'I')), sigma_minus_op, tensor_pauli_ops(pauli_list(num_sites-pos, 'I'))]) 
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        #return ((1j)**(pos-1))*ft.reduce(np.kron, [Tensored_Paulis_Left, sigma_minus_op, np.identity(2**(num_sites-pos))])
        return ((-1j)**(pos-1))*ft.reduce(np.kron, [sigma_minus_op_n, Tensored_Paulis_Right])
        #Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        #coeff = ((-1j)**(pos-1))
        #matrix_part = ft.reduce(np.kron, [np.identity(2**(pos-1)), sigma_minus_op, Tensored_Paulis_Right])
        #return coeff * matrix_part
        #return ft.reduce(np.kron, [np.identity(2**(pos-1)), sigma_minus_op, np.identity(2**(num_sites-pos))])

def JWCreateOp_final(num_sites, pos):
    if pos==1:
        return np.kron(sigma_plus_op, np.identity(2**(num_sites-1)))
    else:
        sigma_plus_op_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(pos-1, 'I')), sigma_plus_op, tensor_pauli_ops(pauli_list(num_sites-pos, 'I'))]) 
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        Z_product = np.identity((2**(num_sites)), dtype='complex128')
        for j in range(1, pos):
            Z_n = (1j)*ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(j-1, 'I')), SparsePauliOp(['Z'], coeffs=[1.0]), tensor_pauli_ops(pauli_list(num_sites-j, 'I'))])
            Z_product = Z_product @ Z_n
        return sigma_plus_op_n @ Z_product
        #return ((-1j)**(pos-1))*ft.reduce(np.kron, [sigma_plus_op_n, Tensored_Paulis_Right])


def JWAnnihOp_final(num_sites, pos):
    if pos==1:
        return np.kron(sigma_minus_op, np.identity(2**(num_sites-1)))
    else:
        sigma_minus_op_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(pos-1, 'I')), sigma_minus_op, tensor_pauli_ops(pauli_list(num_sites-pos, 'I'))]) 
        Tensored_Paulis_Left = tensor_pauli_ops(pauli_list(pos-1, 'Z'))
        Tensored_Paulis_Right = tensor_pauli_ops(pauli_list(num_sites-pos, 'Z'))
        Z_product = np.identity((2**(num_sites)), dtype='complex128')
        for j in range(1, pos):
            Z_n = (-1j)*ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(j-1, 'I')), SparsePauliOp(['Z'], coeffs=[1.0]), tensor_pauli_ops(pauli_list(num_sites-j, 'I'))])
            Z_product = Z_product @ Z_n
        return sigma_minus_op_n @ Z_product
        #return ((-1j)**(pos-1))*ft.reduce(np.kron, [sigma_plus_op_n, Tensored_Paulis_Right])

# Construct terms of Hamiltonian
def Ham_Schwinger_Kinetic(num_sites):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for n in range(1, num_sites):
        result += JWCreateOp_final(num_sites, n) @ JWAnnihOp_final(num_sites, n+1) - JWCreateOp_final(num_sites, n+1) @ JWAnnihOp_final(num_sites, n)
    return result

def Ham_Schwinger_Mass_old(num_sites):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for n in range(1, num_sites+1):
        result += (-(-1)**n) * 0.5*(np.identity(2**num_sites)+ ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n-1, 'I')), SparsePauliOp(['Z'], coeffs=[1.0]), tensor_pauli_ops(pauli_list(num_sites-n, 'I'))]))
    result1 = -0.5*(1+(-1)**(num_sites+1))*np.identity(2**num_sites)
    total_result = result + result1
    return total_result

def Ham_Schwinger_Mass(num_sites):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for n in range(1, num_sites+1):
        result += ((-1)**n) * JWCreateOp_final(num_sites, n) @ JWAnnihOp_final(num_sites, n) 
    return result

def Local_Charge(num_sites, i):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    IZ_op = SparsePauliOp(['I', 'Z'], coeffs=[ 1.+0.j, 1.+0.j])
    temp_op = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), IZ_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
    result = 0.5*temp_op + 0.5*((-1)**i - 1)*np.identity(2**num_sites)
    return result 

def Local_Charge_Alternate(num_sites, i):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    Z_op = SparsePauliOp(['Z'], coeffs=[1.+0.j])
    temp_op = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), Z_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
    result = 0.5*temp_op + 0.5*((-1)**(i))*np.identity(2**num_sites)
    return result 

def Local_Charge_Left(num_sites, i):
    num_sites_half = int(num_sites/2)
    result = np.zeros((2**(num_sites_half), 2**(num_sites_half)), dtype='complex128')
    IZ_op = SparsePauliOp(['I', 'Z'], coeffs=[ 1.+0.j, 1.+0.j])
    temp_op = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), IZ_op, tensor_pauli_ops(pauli_list(num_sites_half-i, 'I'))])
    result = 0.5*temp_op + 0.5*((-1)**i - 1)*np.identity(2**num_sites_half)
    return result

def Local_Charge_Right(num_sites, i):
    num_sites_half = int(num_sites/2)
    k = i
    if num_sites_half%2 == 1:
        k = i+1
    result = np.zeros((2**(num_sites_half), 2**(num_sites_half)), dtype='complex128')
    IZ_op = SparsePauliOp(['I', 'Z'], coeffs=[ 1.+0.j, 1.+0.j])
    temp_op = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), IZ_op, tensor_pauli_ops(pauli_list(num_sites_half-i, 'I'))])
    result = 0.5*temp_op + 0.5*((-1)**k - 1)*np.identity(2**num_sites_half)
    return result 

def Ldyn(num_sites, n):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for i in range(1, n+1):
        result += Local_Charge(num_sites, i)
    return result 

def Ham_Schwinger_Elec(num_sites, t, ti, a):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    result_Lext = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    result_Ldynbos = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for n in range(1, num_sites):
        result_Lext = -np.heaviside(((t-ti)/a) - np.abs(n - num_sites/2), 0)*np.identity(2**num_sites)
        result += (Ldyn(num_sites, n) + result_Lext  ) @ (Ldyn(num_sites, n) + result_Lext  )
    return result


def Ham_Schwinger_Bos(N, t, ti, m, a, g):
    return -(1j/(2*a))*Ham_Schwinger_Kinetic(N) + m*Ham_Schwinger_Mass(N) + (a*g*g/2)*Ham_Schwinger_Elec(N, t, ti, a)



def Ham_Schwinger_Kinetic_Spin(num_sites):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    X_op = SparsePauliOp(['X'], coeffs=[ 1.+0.j])
    Y_op = SparsePauliOp(['Y'], coeffs=[ 1.+0.j])
    for n in range(1, num_sites):
        X_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n-1, 'I')), X_op, tensor_pauli_ops(pauli_list(num_sites-n, 'I'))])
        Y_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n-1, 'I')), Y_op, tensor_pauli_ops(pauli_list(num_sites-n, 'I'))])
        X_n1 = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n, 'I')), X_op, tensor_pauli_ops(pauli_list(num_sites-n-1, 'I'))])
        Y_n1 = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n, 'I')), Y_op, tensor_pauli_ops(pauli_list(num_sites-n-1, 'I'))])
        result += X_n @ X_n1 + Y_n @ Y_n1
    return result

def Ham_Schwinger_Mass_Spin(num_sites):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    Z_op = SparsePauliOp(['Z'], coeffs=[ 1.+0.j])
    for n in range(1, num_sites+1):
        Z_n = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(n-1, 'I')), Z_op, tensor_pauli_ops(pauli_list(num_sites-n, 'I'))])
        result += ((-1)**n) * Z_n
    #result1 = -0.5*(1+(-1)**(num_sites+1))*np.identity(2**num_sites)
    total_result = result # + result1
    return total_result

def Ldyn(num_sites, n):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for i in range(1, n+1):
        result += Local_Charge_Alternate(num_sites, i)
    return result 

def Ham_Schwinger_Elec_Spin(num_sites, t, ti, a):
    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    result_Lext = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    result_Ldynbos = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
    for n in range(1, num_sites):
        result_Lext = -np.heaviside(((t-ti)/a) - np.abs(n - num_sites/2), 0)*np.identity(2**num_sites)
        result += (Ldyn(num_sites, n) + result_Lext  ) @ (Ldyn(num_sites, n) + result_Lext  )
    return result

def Ham_Schwinger_Spin(N, t, ti, m, a, g):
    return (1/(4*a))*Ham_Schwinger_Kinetic_Spin(N) + (m/2)*Ham_Schwinger_Mass_Spin(N) + (a*g*g/2)*Ham_Schwinger_Elec_Spin(N, t, ti, a)



#def Local_Charge_alternate(num_sites, n):
#    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
#    for i in range(1, n+1):
#        temp_plus=ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), sigma_plus_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
#        temp_minus=ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), sigma_minus_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
#        result += temp_plus @ temp_minus + 0.5*((-1)**i - 1)*np.identity(2**num_sites)
#        #IZ_op = SparsePauliOp(['I', 'Z'], coeffs=[ 1.+0.j, 1.+0.j])
#        #temp_op = ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), IZ_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
#        #temp_plus=ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), sigma_plus_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
#        #temp_minus=ft.reduce(np.kron, [tensor_pauli_ops(pauli_list(i-1, 'I')), sigma_minus_op, tensor_pauli_ops(pauli_list(num_sites-i, 'I'))])
#        #result += 0.5*temp_op + 0.5*((-1)**i - 1)*np.identity(2**num_sites)
#    return result 

#def Ldynbos(num_sites, n):
#    result = np.zeros((2**(num_sites), 2**(num_sites)), dtype='complex128')
#    for i in range(1, num_sites):
#        result += Localcharge(num_sites, i)
#    return result 

### Pauli operator form of the Hamiltonian

def Ham_Schwinger_Kinetic_Pauli(num_sites:int):
    op_list = []
    n_tol = num_sites 
    X_op = SparsePauliOp(['X'], coeffs=[ 1.+0.j])
    Y_op = SparsePauliOp(['Y'], coeffs=[ 1.+0.j])
    for n in range(num_sites-1):
        X_n = expand_operator_to_all_sites(X_op, n, n_tol)
        Y_n = expand_operator_to_all_sites(Y_op, n, n_tol)
        X_n1 = expand_operator_to_all_sites(X_op, n+1, n_tol)
        Y_n1 = expand_operator_to_all_sites(Y_op, n+1, n_tol)
        op = X_n @ X_n1 + Y_n @ Y_n1
        op_list.append(op)
    return sum_pauli_ops(op_list)

def Ham_Schwinger_Mass_Pauli(num_sites:int):
    op_list = []
    n_tol = num_sites 
    Z_op = SparsePauliOp(['Z'], coeffs=[ 1.+0.j])
    for n in range(num_sites):
        Z_n = expand_operator_to_all_sites(Z_op, n, n_tol)
        op = ((-1)**(n+1)) * Z_n
        op_list.append(op)
    return sum_pauli_ops(op_list)

def Local_Charge_Pauli(num_sites:int, i:int):
    n_tol = num_sites
    Z_op = SparsePauliOp(['Z'], coeffs=[1.+0.j])
    I_op = SparsePauliOp(['I'], coeffs=[1.+0.j])
    Z_n = expand_operator_to_all_sites(Z_op, i-1, n_tol)
    I_n = expand_operator_to_all_sites(I_op, i-1, n_tol)
    op = (0.5*Z_n + 0.5*((-1)**(i))*I_n)
    return op 

def Ldyn_Pauli(num_sites:int, n:int):
    op_list = []
    for i in range(1, n+1):
        op = Local_Charge_Pauli(num_sites, i)
        op_list.append(op)
    return sum_pauli_ops(op_list)

def Ham_Schwinger_Elec_Pauli(num_sites:int, t, ti, a):
    op_list = []
    n_tol = num_sites
    I_op = SparsePauliOp(['I'], coeffs=[1.+0.j])
    I_n = expand_operator_to_all_sites(I_op, num_sites, n_tol)
    for n in range(1, num_sites):
        #heavi_ans = ((t-ti)/a) - np.abs(n - num_sites/2)
        #if ((t-ti)/a) - np.abs(n - num_sites/2) < 0:
        #    heavi_ans = 0
        #Lext = -heavi_ans*I_n
        op = (Ldyn_Pauli(num_sites, n)  ) @ (Ldyn_Pauli(num_sites, n) )
        op_list.append(op)
    return sum_pauli_ops(op_list)

def Ham_Schwinger_Elec_Pauli_Unsquared(num_sites:int, t, ti, a):
    op_list = []
    n_tol = num_sites
    I_op = SparsePauliOp(['I'], coeffs=[1.+0.j])
    I_n = expand_operator_to_all_sites(I_op, num_sites, n_tol)
    for n in range(1, num_sites):
        #heavi_ans = ((t-ti)/a) - np.abs(n - num_sites/2)
        #if ((t-ti)/a) - np.abs(n - num_sites/2) < 0:
        #    heavi_ans = 0
        #Lext = -heavi_ans*I_n
        op = (Ldyn_Pauli(num_sites, n) ) 
        op_list.append(op)
    return sum_pauli_ops(op_list)

def Ham_Schwinger_Pauli(N, t, ti, m, a, g):
    return (1/(4*a))*Ham_Schwinger_Kinetic_Pauli(N) + (m/2)*Ham_Schwinger_Mass_Pauli(N) + (a*g*g/2)*Ham_Schwinger_Elec_Pauli(N, t, ti, a)