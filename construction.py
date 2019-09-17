import numpy as np
np.set_printoptions(linewidth=200)
import cirq
import random
import itertools as it
from functools import reduce

def helstrom_block(qubits, aq):
    assert len(qubits) > 0
    setup = cirq.Circuit()
    setup.append(cirq.Y(qubits[0])**(-1/4))
    for q,r in zip(qubits[:-1],qubits[1:]):
        setup.append(cirq.H(r).controlled_by(q))
    circuit = cirq.Circuit()
    circuit.append(setup)
    circuit.append(cirq.CNOT(qubits[-1],aq))
    circuit.append(cirq.inverse(setup))
    return circuit

def decode_block(qubits):
    circuit = cirq.Circuit()
    for q,r in zip(qubits[:-1],qubits[1:]):
        circuit.append(cirq.H(r).controlled_by(q))
    return circuit

def decode(qubits, aqs, bs):
    n = len(qubits)
    assert n % bs == 0
    m = n // bs
    circuit = cirq.Circuit()
    for i in range(m-1):
        circuit.append(helstrom_block(qubits[i*bs:(i+1)*bs], aqs[i]))
    for i in range(m-1):
        circuit.append(cirq.H(qubits[(i+1)*bs]).controlled_by(aqs[i]), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    for i in range(m):
        circuit.append(decode_block(qubits[i*bs:(i+1)*bs]), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    for i in range(m-1):
        circuit.append(cirq.CNOT(qubits[(i+1)*bs-1],aqs[i]), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit

if __name__ == "__main__":
    def tensor(*args):
        return reduce(lambda x,y : np.kron(x,y), args, np.array([1])).astype(np.complex64)

    # Parameters
    import sys
    if len(sys.argv) < 3:
        print("Please supply the number of qubits and the block size as command line arguments.")
    n = int(sys.argv[1])
    bs = int(sys.argv[2])

    print("n =", n)
    print("bs =", bs)

    m = n // bs
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    aqs = [cirq.GridQubit(1,i) for i in range(m-1)]
    circuit = decode(qubits, aqs, bs)
    print("Circuit:")
    print(circuit)

    zero = np.array([1,0])
    one = np.array([0,1])
    plus = np.array([1,1]) / np.sqrt(2)
    minus = np.array([1,-1]) / np.sqrt(2)

    num_cols = min(8, 2**((n+1)//2))
    num_rows = 2**n // num_cols
    row_format = '{:<30}' * num_cols
    table = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    for i,bitstring in enumerate(it.product(range(2), repeat = n)):
        print("Simulation progress: {:.3f}%".format(i/2**n*100), end = '\r')
        qubit_states = []
        for x,y in zip([0] + list(bitstring)[:-1], bitstring):
            if x == 0:
                qubit_states.append(zero if y == 0 else one)
            else:
                qubit_states.append(plus if y == 0 else minus)
        init_state = tensor(*(qubit_states + [zero]*len(aqs)))
        fs = cirq.Simulator().simulate(circuit, initial_state = init_state, qubit_order = qubits + aqs).final_state
        expected_state = tensor(*(zero if x == 0 else one for x in bitstring))
        inp = np.sum(np.conj(expected_state) * fs[::2**len(aqs)])

        cell = '{} {:.3f} {:.3f}'.format(''.join(map(str,bitstring)), np.abs(inp), 0.)
        table[i%num_rows][i//num_rows] = cell
    print("Bitstring Innerproduct Lowerbound")
    for x in table:
        print(row_format.format(*x))
