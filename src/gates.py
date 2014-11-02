#!/usr/bin/python
# coding=utf-8

"""
Geometric utility methods used by cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
import numpy as np
import geom
import string
from rotation import *


class QuantumOperation(object):
    """
    A quantum operation that can be applied to one of several qubits, and controlled by other qubits.
    """

    def __init__(self, single_qubit_operation, wire_controls, wire_index):
        """
        :param single_qubit_operation: A 2x2 numpy unitary complex matrix.
        :param wire_index: The index of the qubit the operation should be applied to.
        :param wire_controls: A list of booleans, one for each wire, determining if they control the operation. The
        boolean for the wire the operation actually applies to is ignored. Make sure the number of entries matches the
        expected number of wires.
        """
        if not (0 <= wire_index < len(wire_controls)):
            raise ValueError("need 0 <= wire_index < len(wire_controls)")
        self.single_qubit_operation = single_qubit_operation
        self.wire_index = wire_index
        self.wire_controls = wire_controls

    def interpolated_operation(self, t):
        """
        Returns a gradually-applied version of this operation, where t=0 is not applied at all and t=1 is fully applied.
        :param t: The interpolation factor.
        """
        op = unitary_lerp(Rotation().as_pauli_operation(), self.single_qubit_operation, t)
        return QuantumOperation(op, self.wire_controls, self.wire_index).full_operation()

    def full_operation(self):
        """
        Returns this quantum operation's full matrix that, when multiplied by the full state vector for the circuit,
        applies the gate to the state.

        >>> (QuantumOperation(np.mat([[2, 3], [4, 5]]), [True, None, False], 1).full_operation() ==\
            np.mat([[1, 0, 0, 0, 0, 0, 0, 0],\
                    [0, 2, 0, 3, 0, 0, 0, 0],\
                    [0, 0, 1, 0, 0, 0, 0, 0],\
                    [0, 4, 0, 5, 0, 0, 0, 0],\
                    [0, 0, 0, 0, 1, 0, 0, 0],\
                    [0, 0, 0, 0, 0, 2, 0, 3],\
                    [0, 0, 0, 0, 0, 0, 1, 0],\
                    [0, 0, 0, 0, 0, 4, 0, 5]])).all()
        True
        >>> (QuantumOperation(np.mat([[2, 3], [4, 5]]), [False, None, True], 1).full_operation() ==\
            np.mat([[1, 0, 0, 0, 0, 0, 0, 0],\
                    [0, 1, 0, 0, 0, 0, 0, 0],\
                    [0, 0, 1, 0, 0, 0, 0, 0],\
                    [0, 0, 0, 1, 0, 0, 0, 0],\
                    [0, 0, 0, 0, 2, 0, 3, 0],\
                    [0, 0, 0, 0, 0, 2, 0, 3],\
                    [0, 0, 0, 0, 4, 0, 5, 0],\
                    [0, 0, 0, 0, 0, 4, 0, 5]])).all()
        True
        """

        acc = self.single_qubit_operation
        id2 = np.identity(2)

        for is_control in self.wire_controls[self.wire_index+1:]:
            acc = QuantumOperation.controlled_by_next_qbit(acc)\
                if is_control\
                else geom.tensor_product(id2, acc)

        for is_control in self.wire_controls[:self.wire_index].__reversed__():
            acc = QuantumOperation.controlled_by_prev_qbit(acc)\
                if is_control\
                else geom.tensor_product(acc, id2)

        return acc

    @staticmethod
    def quantum_complex_str(c):
        """
        Returns a text representation of the given complex number. Abbreviates values made up of halves, like 1/2 and
        (1+i)/2, into single unicode arrow characters with the corresponding direction. Shrinks zero into a dot. Avoids
        some cases of unnecessary ".0" and "1i".
        :param c: A complex number.
        """
        if c == 0:
            return "·"
        if c == 1j:
            return "i"
        if c == -1j:
            return "-i"
        if c == -1/2:
            return "←"
        if c == 1j/2:
            return "↑"
        if c == 1/2:
            return "→"
        if c == -1j/2:
            return "↓"
        if c == (-1 + 1j)/2:
            return "↖"
        if c == (1 + 1j)/2:
            return "↗"
        if c == (1 - 1j)/2:
            return "↘"
        if c == (-1 - 1j)/2:
            return "↙"
        if c == int(c.real):
            return str(int(c.real))
        if c == int(c.imag)*1j:
            return str(int(c.imag)) + "i"
        if c.imag == 0:
            return str(c.real)
        if c.real == 0:
            return str(c.imag) + "i"
        return str(c)

    @staticmethod
    def quantum_operation_str(op):
        """
        Returns a text representation of the matrix corresponding to a quantum operation. Abbreviates commonly occurring
        complex numbers into single characters to keep the matrices well spaced.
        :param op: A numpy complex matrix.

        >>> expected = "┌         ┐\\n" +\
                       "│ 1 i -1 -i │\\n" +\
                       "│ → ↑ ← ↓ │\\n" +\
                       "│ ↗ ↖ ↙ ↘ │\\n" +\
                       "│ · · · · │\\n" +\
                       "└         ┘"
        >>> QuantumOperation.quantum_operation_str(np.mat([\
            [1, 1j, -1, -1j],\
            [0.5, 0.5j, -0.5, -0.5j],\
            [0.5+0.5j, -0.5+0.5j, -0.5-0.5j, 0.5-0.5j],\
            [0, 0, 0, 0]])) == expected
        True
        """
        op = op.tolist()
        row_reps = ["│ " + string.join(map(QuantumOperation.quantum_complex_str, col)) + " │" for col in op]
        n = len(op)*2 + 3
        top_row = "┌" + " " * (n-2) + "┐"
        bottom_row = "└" + " " * (n-2) + "┘"
        return string.join(
            [top_row] + row_reps + [bottom_row],
            "\n")

    @staticmethod
    def controlled_by_next_qbit(m):
        """
        Expands a quantum operation to apply to one more qubit, that comes after the wires it currently applies to,
        with the caveat that it is controlled by the new qubit. The operation only applies when the new qubit is true.

        :param m: A numpy complex matrix.
        >>> (QuantumOperation.controlled_by_next_qbit(\
                np.mat([[2]]))\
            == np.mat([[1, 0], [0, 2]])).all()
        True
        >>> (QuantumOperation.controlled_by_next_qbit(np.mat(\
                [[2, 3, 5, 7],\
                 [11, 13, 17, 19],\
                 [23, 29, 31, 37],\
                 [41, 43, 47, 51]]))\
            == np.mat(\
                [[1, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 1, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 1, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 1, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 2, 3, 5, 7],\
                 [0, 0, 0, 0, 11, 13, 17, 19],\
                 [0, 0, 0, 0, 23, 29, 31, 37],\
                 [0, 0, 0, 0, 41, 43, 47, 51]])).all()
        True
        """
        d = m.shape[0]
        return np.mat([[m[i - d, j - d] if i >= d and j >= d
                        else 1 if i == j
                        else 0
                        for j in range(2*d)]
                       for i in range(2*d)])


    @staticmethod
    def controlled_by_prev_qbit(m):
        """
        Expands a quantum operation to apply to one more qubit, that comes before the wires it currently applies to,
        with the caveat that it is controlled by the new qubit. The operation only applies when the new qubit is true.

        :param m: A numpy complex matrix.
        >>> (QuantumOperation.controlled_by_prev_qbit(\
                np.mat([[2]]))\
            == np.mat([[1, 0], [0, 2]])).all()
        True
        >>> (QuantumOperation.controlled_by_prev_qbit(np.mat(\
                [[2, 3, 5, 7],\
                 [11, 13, 17, 19],\
                 [23, 29, 31, 37],\
                 [41, 43, 47, 51]]))\
            == np.mat(\
                [[1, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 2, 0, 3, 0, 5, 0, 7],\
                 [0, 0, 1, 0, 0, 0, 0, 0],\
                 [0,11, 0,13, 0,17, 0,19],\
                 [0, 0, 0, 0, 1, 0, 0, 0],\
                 [0,23, 0,29, 0,31, 0,37],\
                 [0, 0, 0, 0, 0, 0, 1, 0],\
                 [0,41, 0,43, 0,47, 0,51]])).all()
        True
        """
        d = m.shape[0]
        return np.mat([[m[i // 2, j // 2] if i % 2 == 1 and j % 2 == 1
                        else 1 if i == j
                        else 0
                        for j in range(2*d)]
                       for i in range(2*d)])

    def gate_char(self):
        """
        Returns a single-character text representation of the single-qubit operation being applied, falling back to a
        question mark for unexpected operations.
        """
        if (self.single_qubit_operation == Rotation().as_pauli_operation()).all():
            return "I"

        if (self.single_qubit_operation == Rotation(x=0.25).as_pauli_operation()).all():
            return "↓"
        if (self.single_qubit_operation == Rotation(x=0.5).as_pauli_operation()).all():
            return "X"
        if (self.single_qubit_operation == Rotation(x=-0.25).as_pauli_operation()).all():
            return "↑"

        if (self.single_qubit_operation == Rotation(y=0.25).as_pauli_operation()).all():
            return "→"
        if (self.single_qubit_operation == Rotation(y=0.5).as_pauli_operation()).all():
            return "Y"
        if (self.single_qubit_operation == Rotation(y=-0.25).as_pauli_operation()).all():
            return "←"

        if (self.single_qubit_operation == Rotation(z=0.25).as_pauli_operation()).all():
            return "↺"
        if (self.single_qubit_operation == Rotation(z=0.5).as_pauli_operation()).all():
            return "Z"
        if (self.single_qubit_operation == Rotation(z=-0.25).as_pauli_operation()).all():
            return "↻"

        return "?"

    def operator_column_str(self):
        """
        Returns a circuit-looking text representation of this gate.

        >>> expected = "─┬─\\n" +\
                       "─↓⃞─\\n" +\
                       "───\\n"
        >>> QuantumOperation(Rotation(x=0.25).as_pauli_operation(), [True, None, False], 1).operator_column_str()\
            == expected
        True
        >>> expected = "───\\n" +\
                       "─┬─\\n" +\
                       "─┼─\\n" +\
                       "─↺⃞─\\n" +\
                       "─│─\\n" +\
                       "─┴─\\n" +\
                       "───\\n"
        >>> QuantumOperation(Rotation(z=0.25).as_pauli_operation(), [False, True, True, None, False, True, False], 3 \
            ).operator_column_str() == expected
        True
        """
        controls = [self.wire_index] + [i for i in range(len(self.wire_controls)) if self.wire_controls[i]]
        start = min(controls)
        end = max(controls)
        col = [self.gate_char() + "⃞" if i == self.wire_index
               else "┬" if i == start
               else "┴" if i == end
               else "┼" if self.wire_controls[i]
               else "│" if start < i < end
               else "─"
               for i in range(len(self.wire_controls))]
        return string.join(["─" + c + "─" for c in col], "\n") + "\n"

    def __str__(self):
        return self.operator_column_str() + "\n" + QuantumOperation.quantum_operation_str(self.full_operation())

