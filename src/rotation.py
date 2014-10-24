#!/usr/bin/python
# coding=utf-8

"""
Utility methods and classes related to rotations.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
from quaternion import *
import cmath
import math
import numpy as np


tau = math.pi * 2


def cos_tau(p):
    """
    Returns the cosine of the angle corresponding to a p'th of a turn.
    :param p: The proportion of a full turn to rotate <1, 0> by before returning the resulting vector's x coordinate.

    >>> cos_tau(0)
    1
    >>> cos_tau(0.25)
    0
    >>> cos_tau(0.5)
    -1
    >>> cos_tau(0.75)
    0
    >>> cos_tau(1)
    1
    >>> abs(cos_tau(0.125) - math.sqrt(2)/2) < 0.000001
    True
    """
    p %= 1
    if p == 0:
        return 1
    if p == 0.25:
        return 0
    if p == 0.5:
        return -1
    if p == 0.75:
        return 0
    return math.cos(p*tau)


def sin_tau(p):
    """
    Returns the sine of the angle corresponding to a p'th of a turn.
    :param p: The proportion of a full turn to rotate <1, 0> by before returning the resulting vector's y coordinate.

    >>> sin_tau(0)
    0
    >>> sin_tau(0.25)
    1
    >>> sin_tau(0.5)
    0
    >>> sin_tau(0.75)
    -1
    >>> sin_tau(1)
    0
    >>> abs(sin_tau(0.125) - math.sqrt(2)/2) < 0.000001
    True
    """
    return cos_tau(p-0.25)


def atan2_tau(y, x):
    """
    Determines the angle, measured in fractions of a complete turns instead of in radians, to the given coordinates.
    Angles start from the right and rotate counter-clockwise, so +x-ward is 0, and +y-ward is 1/4.

    :param y: The y coordinate.
    :param x: The x coordinate.

    >>> atan2_tau(y=0, x=1) == 0.125*0
    True
    >>> atan2_tau(y=1, x=1) == 0.125*1
    True
    >>> atan2_tau(y=1, x=0) == 0.125*2
    True
    >>> atan2_tau(y=1, x=-1) == 0.125*3
    True
    >>> atan2_tau(y=0, x=-1) == 0.125*-4
    True
    >>> atan2_tau(y=-1, x=-1) == 0.125*-3
    True
    >>> atan2_tau(y=-1, x=0) == 0.125*-2
    True
    >>> atan2_tau(y=-1, x=1) == 0.125*-1
    True
    """
    if y == 0 and x > 0:
        return 0
    if x == y and y > 0:
        return 0.125
    if x == 0 and y > 0:
        return 0.25
    if x == -y and y > 0:
        return 0.375
    if y == 0 and x < 0:
        return -0.5
    if x == y and y < 0:
        return -0.375
    if x == 0 and y < 0:
        return -0.25
    if x == -y and y < 0:
        return -0.125

    return math.atan2(y, x) / tau


def exp_i_tau(p):
    """
    Returns Euler's constant e to the power of i times tau times p.
    :param p: The proportion of a full turn to rotate 1+0i.

    >>> exp_i_tau(0) == 1
    True
    >>> exp_i_tau(0.25) == 1j
    True
    >>> exp_i_tau(0.5) == -1
    True
    >>> exp_i_tau(0.75) == -1j
    True
    >>> exp_i_tau(1) == 1
    True
    >>> abs(exp_i_tau(0.125) - math.sqrt(2)/2 * (1 + 1j)) < 0.000001
    True
    """
    return cos_tau(p) + 1j * sin_tau(p)


def str_fraction(v):
    """
    Returns a text representation of a number, using unicode vulgar fraction characters for special values.
    :param v: The value to represent.
    """
    if v < 0:
        return "-" + str_fraction(-v)
    if v == 0:
        return "0"
    if v == 0.25:
        return "¼"
    if v == 0.5:
        return "½"
    if v == 0.75:
        return "¾"
    return str(v)


def sinc(theta):
    """
    The cardinal sine function, equal to sin(x)/x.
    :param theta: The angle, in radians.

    # known values
    >>> sinc(0) == 1
    True
    >>> abs(sinc(math.pi)) < 0.000000001
    True
    >>> abs(sinc(-math.pi)) < 0.000000001
    True
    >>> abs(sinc(math.pi/2) - 2/math.pi) < 0.000000001
    True
    >>> abs(sinc(3*math.pi/2) - -2/3/math.pi) < 0.000000001
    True

    # even-ness
    >>> sinc(0.0011) == sinc(-0.0011)
    True
    >>> sinc(0.0009) == sinc(-0.0009)
    True
    >>> sinc(math.pi/2) == sinc(-math.pi/2)
    True
    >>> sinc(3*math.pi/2) == sinc(-3*math.pi/2)
    True

    # continuity around switch to approximation
    >>> abs(sinc(0.0011) - 0.999999798333345) < 0.0000000000001
    True
    >>> abs(sinc(0.0009) - 0.999999865000005) < 0.0000000000001
    True
    """

    # Near zero, avoid the singularity by using an approximation based on the first two Taylor series terms
    if abs(theta) < 0.001:
        return 1 - theta**2/6
    return math.sin(theta) / theta


def sinc_tau(turns):
    """
    The cardinal sine function, in fractions of a turn, equal to sin_tau(x)/x.
    Note that sinc_tau(x) is NOT equal to sinc(τ x), it's equal to τ sinc(τ x).
    :param turns: The angle, in fractions of a turn.

    # known values
    >>> abs(sinc_tau(-0.75) - -4/3) < 0.0000001
    True
    >>> sinc_tau(-0.5) == 0
    True
    >>> sinc_tau(-0.25) == 4
    True
    >>> abs(sinc_tau(-0.125) - 4*math.sqrt(2)) < 0.0000001
    True
    >>> sinc_tau(0) == tau
    True
    >>> abs(sinc_tau(0.125) - 4*math.sqrt(2)) < 0.0000001
    True
    >>> sinc_tau(0.25) == 4
    True
    >>> sinc_tau(0.5) == 0
    True
    >>> abs(sinc_tau(0.75) - -4/3) < 0.0000001
    True

    # continuity around switch to approximation
    >>> abs(sinc(0.0011) - 0.999999798333345) < 0.0000000000001
    True
    >>> abs(sinc(0.0009) - 0.999999865000005) < 0.0000000000001
    True
    """

    # Near zero, avoid the singularity by using an approximation based on the first two Taylor series terms
    if abs(turns) < 0.001:
        return tau - turns**2 * tau**3 / 6
    return sin_tau(turns) / turns


def smooth_near_quarter_turn(turns):
    """
    Corrects floating point errors, based on the assumption that the given turn is often close to a multiple of 0.25.
    :param turns: An angle, in fractions of a turn.

    >>> smooth_near_quarter_turn(0) == 0
    True
    >>> smooth_near_quarter_turn(0.25) == 0.25
    True
    >>> smooth_near_quarter_turn(0.223434325454) == 0.223434325454
    True
    >>> smooth_near_quarter_turn(0.250000000001) == 0.25
    True
    >>> smooth_near_quarter_turn(0.249999999999) == 0.25
    True
    >>> smooth_near_quarter_turn(1.962615573354719e-8) == 0
    True
    >>> smooth_near_quarter_turn(1.962615573354719e-17) == 0
    True
    """
    err = ((turns - 0.125) % 0.25) - 0.125
    if abs(err) < 0.000001:
        return round(turns*4)/4
    return turns


def unitary_breakdown(m):
    """
    Breaks a 2x2 unitary matrix into quaternion-esqe Ii, X, Y, and Z components as well as a phase component.
    :param m: The 2x2 unitary matrix to break down.
    :return: (t, x, y, z, complex_unit) such that m = complex_unit * (<t, x, y, z> . <Ii, X, Y, Z>)

    >>> unitary_breakdown(Rotation().as_pauli_operation()) == (1, 0, 0, 0, -1j)
    True
    >>> unitary_breakdown(Rotation(x=0.5).as_pauli_operation()) == (0, 1, 0, 0, 1)
    True
    >>> unitary_breakdown(Rotation(y=0.5).as_pauli_operation()) == (0, 0, 1, 0, 1)
    True
    >>> unitary_breakdown(Rotation(z=0.5).as_pauli_operation()) == (0, 0, 0, 1, 1)
    True
    >>> np.sum(np.abs(np.array(unitary_breakdown(Rotation(x=0.25).as_pauli_operation())) \
        - np.array((cos_tau(0.125), sin_tau(0.125), 0, 0, exp_i_tau(-0.125))))) < 0.00001
    True
    >>> np.sum(np.abs(np.array(unitary_breakdown(Rotation(y=0.25).as_pauli_operation())) \
        - np.array((cos_tau(0.125), 0, sin_tau(0.125), 0, exp_i_tau(-0.125))))) < 0.00001
    True
    >>> np.sum(np.abs(np.array(unitary_breakdown(Rotation(z=0.25).as_pauli_operation())) \
        - np.array((cos_tau(0.125), 0, 0, sin_tau(0.125), exp_i_tau(-0.125))))) < 0.00001
    True
    >>> np.sum(np.abs(np.array(unitary_breakdown(Rotation(x=0.25).then(Rotation(z=0.25)).as_pauli_operation())) \
        - np.array((0.5, 0.5, 0.5, 0.5, exp_i_tau(-1/12))))) < 0.00001
    True
    """

    # Extract components
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    t = (a + d)/2j
    x = (b + c)/2
    y = (b - c)/-2j
    z = (a - d)/2

    # Extract phase factor
    p = max([t, x, y, z], key=lambda e: abs(e))
    p /= abs(p)
    pt, px, py, pz = t/p, x/p, y/p, z/p

    # Assuming the input was unitary, cancelling the phase factor should make the other components all real
    # (not counting floating point errors, of course)

    return pt.real, px.real, py.real, pz.real, p


def unitary_lerp(u1, u2, t):
    """
    Continuously interpolates between 2x2 unitary matrices, with unitary intermediates.
    :param u1: The initial unitary operation, used at t=0.
    :param u2: The final unitary operation, used at t=1.
    :param t: The interpolation factor, ranging from 0 to 1.

    >>> np.sum(np.abs((unitary_lerp(np.mat([[1, 0], [0, 1]]), np.mat([[1, 0], [0, 1]]) * 1j, 0.5) \
            - np.mat([[1, 0], [0, 1]]) * exp_i_tau(1/8)))) < 0.000001
    True
    >>> (unitary_lerp(Rotation(x=0.5).as_pauli_operation(), Rotation(z=0.5).as_pauli_operation(), 0.5) \
            == np.mat([[1,1],[1,-1]])/math.sqrt(2)).all()
    True
    >>> (unitary_lerp(Rotation(x=0.5).as_pauli_operation(), Rotation(x=0.5).as_pauli_operation(), 0.5) \
            == Rotation(x=0.5).as_pauli_operation()).all()
    True
    >>> np.sum(np.abs(unitary_lerp(Rotation().as_pauli_operation(), Rotation(x=-0.25).as_pauli_operation(), 0.5) \
            - Rotation(x=-0.125).as_pauli_operation())) < 0.000001
    True
    >>> np.sum(np.abs(unitary_lerp(Rotation().as_pauli_operation(), Rotation(x=0.75).as_pauli_operation(), 0.5) \
            - Rotation(x=-0.125).as_pauli_operation())) < 0.000001
    True
    >>> np.sum(np.abs(unitary_lerp(Rotation().as_pauli_operation(), Rotation(x=0.25).as_pauli_operation(), 0.5) \
            - Rotation(x=0.125).as_pauli_operation())) < 0.000001
    True
    >>> np.sum(np.abs(unitary_lerp(Rotation().as_pauli_operation(), Rotation(x=0.5).as_pauli_operation(), 0.5) \
            - Rotation(x=0.25).as_pauli_operation())) < 0.000001
    True
    >>> np.sum(np.abs(unitary_lerp(Rotation(x=0.5).as_pauli_operation(), Rotation(x=0.75).as_pauli_operation(), 0.5) \
            - Rotation(x=0.625).as_pauli_operation())) < 0.000001
    True
    """
    t1, x1, y1, z1, p1 = unitary_breakdown(u1)
    t2, x2, y2, z2, p2 = unitary_breakdown(u2)
    n1 = u1/p1
    n2 = u2/p2

    # Spherical interpolation of rotation part
    dot = Quaternion(t1, x1, y1, z1).dot(Quaternion(t2, x2, y2, z2))
    if dot < 0:
        p2 *= -1
        n2 *= -1
        dot *= -1
    theta = math.acos(max(min(dot, 1), -1))
    c1 = scaled_sin_ratio(theta, 1-t)
    c2 = scaled_sin_ratio(theta, t)
    n3 = (n1*c1 + n2*c2)

    # Angular interpolation of phase part
    phase_angle_1 = cmath.log(p1).imag
    phase_angle_2 = cmath.log(p2).imag
    phase_drift = (phase_angle_2 - phase_angle_1 + math.pi) % tau - math.pi
    phase_angle_3 = phase_angle_1 + phase_drift * t
    p3 = cmath.exp(phase_angle_3 * 1j)

    return n3 * p3


class Rotation(object):
    """
    Represents a rotation about some axis.
    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.v = (x, y, z)
        self.x = x
        self.y = y
        self.z = z

    def turns(self):
        """
        Returns the amount of turning this rotation performs, measured in full turns so 0.25 is a quarter turn.

        >>> Rotation(z=0.1).turns()
        0.1
        >>> Rotation(x=100.5).turns()
        100.5
        >>> Rotation(x=1, y=1).turns() == math.sqrt(2)
        True
        >>> Rotation(x=0.5, y=-0.5).turns() == math.sqrt(0.5)
        True
        >>> Rotation(x=3, z=4).turns()
        5.0
        >>> Rotation().turns()
        0.0
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def axis(self):
        """
        Returns a unit vector along the axis that this rotation rotates around.

        >>> Rotation(z=0.1).axis() == (0, 0, 1)
        True
        >>> Rotation(x=100.5).axis() == (1, 0, 0)
        True
        >>> Rotation(x=1, y=1).axis() == (1/math.sqrt(2), 1/math.sqrt(2), 0)
        True
        >>> Rotation(x=0.5, y=-0.5).axis() == (1/math.sqrt(2), -1/math.sqrt(2), 0)
        True
        >>> Rotation(y=-0.25).axis() == (0, -1, 0)
        True
        >>> Rotation().axis() == (0, 0, 0)
        True
        """
        length = self.turns()
        if length < 0.0000001:
            return 0, 0, 0
        return self.x / length, self.y / length, self.z / length

    def then(self, following_rotation):
        """
        Returns the net rotation from applying this rotation and then the given rotation.
        :param following_rotation: The rotation applied after this rotation.

        >>> Rotation(x=0.25).then(Rotation(x=0.25))
        X:½
        >>> Rotation(x=0.5).then(Rotation(y=0.5))
        Z:½
        >>> Rotation(x=0.5).then(Rotation(y=0.5)).then(Rotation(z=0.5))
        (no rotation)
        >>> Rotation(x=0.25).then(Rotation(x=-0.25))
        (no rotation)
        >>> Rotation(x=0.25).then(Rotation())
        X:¼
        >>> Rotation(x=0.25).then(Rotation(y=0.25)).then(Rotation(z=0.25))
        Y:¼
        """
        return Rotation.from_quaternion(following_rotation.as_quaternion() * self.as_quaternion())

    def as_pauli_operation(self):
        """
        Returns a unitary matrix corresponding to this rotation.
        The mapping is continuous, and maps half-turns around each X, Y, and Z axis to the corresponding Pauli matrix.

        # Known X rotations
        >>> np.all(Rotation(x=0.25).as_pauli_operation() == np.mat([[1j, 1], [1, 1j]]) * (1 - 1j) / 2)
        True
        >>> np.all(Rotation(x=0.5).as_pauli_operation() == np.mat([[0, 1], [1, 0]]))
        True
        >>> np.all(Rotation(x=0.75).as_pauli_operation() == np.mat([[-1j, 1], [1, -1j]]) * (1 + 1j) / 2)
        True

        # Known Y rotations
        >>> np.all(Rotation(y=0.25).as_pauli_operation() == np.mat([[1, -1], [1, 1]]) * (1 + 1j) / 2)
        True
        >>> np.all(Rotation(y=0.5).as_pauli_operation() == np.mat([[0, -1j], [1j, 0]]))
        True
        >>> np.all(Rotation(y=0.75).as_pauli_operation() == np.mat([[1, 1], [-1, 1]]) * (1 - 1j) / 2)
        True

        # Known Z rotations
        >>> np.all(Rotation(z=0.25).as_pauli_operation() == np.mat([[1, 0], [0, 1j]]))
        True
        >>> np.all(Rotation(z=0.5).as_pauli_operation() == np.mat([[1, 0], [0, -1]]))
        True
        >>> np.all(Rotation(z=0.75).as_pauli_operation() == np.mat([[1, 0], [0, -1j]]))
        True

        # Two quarter rotations are the same as a half rotation
        >>> np.all(Rotation(x=0.25).as_pauli_operation()**2 == Rotation(x=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(y=0.25).as_pauli_operation()**2 == Rotation(y=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(z=0.25).as_pauli_operation()**2 == Rotation(z=0.5).as_pauli_operation())
        True

        # Backtracking undoes effects
        >>> np.all(Rotation(x=0.25).as_pauli_operation()*Rotation(x=-0.25).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(y=0.25).as_pauli_operation()*Rotation(y=-0.25).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(z=0.25).as_pauli_operation()*Rotation(z=-0.25).as_pauli_operation() == np.identity(2))
        True

        # Full rotations around a single axis have no effect
        >>> np.all(Rotation().as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(x=0.5).as_pauli_operation()**2 == np.identity(2))
        True
        >>> np.all(Rotation(y=0.5).as_pauli_operation()**2 == np.identity(2))
        True
        >>> np.all(Rotation(z=0.5).as_pauli_operation()**2 == np.identity(2))
        True
        >>> np.all(Rotation(x=0.25).as_pauli_operation()*Rotation(x=0.75).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(y=0.25).as_pauli_operation()*Rotation(y=0.75).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(z=0.25).as_pauli_operation()*Rotation(z=0.75).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(x=0.75).as_pauli_operation()*Rotation(x=0.25).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(y=0.75).as_pauli_operation()*Rotation(y=0.25).as_pauli_operation() == np.identity(2))
        True
        >>> np.all(Rotation(z=0.75).as_pauli_operation()*Rotation(z=0.25).as_pauli_operation() == np.identity(2))
        True

        # Quarter turns double into half turns
        >>> np.all(Rotation(x=0.25).as_pauli_operation()**2 == Rotation(x=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(y=0.25).as_pauli_operation()**2 == Rotation(y=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(z=0.25).as_pauli_operation()**2 == Rotation(z=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(x=0.75).as_pauli_operation()**2 == Rotation(x=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(y=-0.25).as_pauli_operation()**2 == Rotation(y=0.5).as_pauli_operation())
        True
        >>> np.all(Rotation(z=-0.25).as_pauli_operation()**2 == Rotation(z=0.5).as_pauli_operation())
        True

        # Unlike rotations, X then Y then Z doesn't *quite* get you back to the starting point
        >>> np.all(Rotation(x=0.5).as_pauli_operation() \
                 * Rotation(y=0.5).as_pauli_operation() \
                 * Rotation(z=0.5).as_pauli_operation() \
                 == np.identity(2) * 1j)
        True
        >>> np.all(Rotation(x=0.5).as_pauli_operation() \
                 * Rotation(z=0.5).as_pauli_operation() \
                 * Rotation(y=0.5).as_pauli_operation() \
                 == np.identity(2) * -1j)
        True
        """
        x, y, z = self.v

        # flip axis vector if its first non-zero coordinate is negative, so phase correction plays out correctly
        s = x if x != 0 \
            else y if y != 0 \
            else z
        if s < 0:
            ux, uy, uz = self.axis()
            x -= ux
            y -= uy
            z -= uz
        theta = math.sqrt(x**2 + y**2 + z**2)

        if theta < 0.00001:
            return np.identity(2)

        # Pauli matrices
        p1 = np.identity(2)
        px = np.mat([[0, 1],
                     [1, 0]])
        py = np.mat([[0, -1j],
                     [1j, 0]])
        pz = np.mat([[1, 0],
                     [0, -1]])

        # Pauli vector for rotation
        pv = (x*px + y*py + z*pz)/theta

        # Magic!
        return (p1+pv + exp_i_tau(theta) * (p1-pv)) / 2

    def as_quaternion(self):
        """
        Returns a quaternion corresponding to this rotation.

        >>> Rotation().as_quaternion()
        1
        >>> Rotation(x=0.5).as_quaternion().rotate(Quaternion(x=1))
        i
        >>> Rotation(x=0.5).as_quaternion().rotate(Quaternion(y=1))
        -j
        >>> Rotation(x=0.5).as_quaternion().rotate(Quaternion(z=1))
        -k
        >>> abs(Rotation(x=0.25).as_quaternion().rotate(Quaternion(y=1)) - Quaternion(z=1)) < 0.00001
        True
        >>> Rotation(x=1/3**1.5, y=1/3**1.5, z=1/3**1.5).as_quaternion()
        0.5 + 0.5i + 0.5j + 0.5k
        """
        t = self.turns()
        if t < 0.000001:
            return Quaternion(1)
        c = cos_tau(t/2)
        s = sin_tau(t/2) / t
        return Quaternion(c, s * self.v[0], s * self.v[1], s * self.v[2])

    @staticmethod
    def from_quaternion(q):
        """
        Returns a rotation that rotates in the same way as the given quaternion.
        :param q: The quaternion to translate.

        >>> Rotation.from_quaternion(Quaternion(1, 1, 1, 1)/2).as_quaternion()
        0.5 + 0.5i + 0.5j + 0.5k
        >>> Rotation.from_quaternion(Quaternion(1, 1, -1, 1)/2).as_quaternion()
        0.5 + 0.5i - 0.5j + 0.5k

        # Preserves simple rotations
        >>> Rotation.from_quaternion(Quaternion(1)) == Rotation()
        True
        >>> Rotation.from_quaternion(Rotation(x=0.25).as_quaternion())
        X:¼
        >>> Rotation.from_quaternion(Rotation(x=-0.25).as_quaternion())
        X:¾
        >>> Rotation.from_quaternion(Rotation(y=0.25).as_quaternion())
        Y:¼
        >>> Rotation.from_quaternion(Rotation(y=-0.25).as_quaternion())
        Y:¾
        >>> Rotation.from_quaternion(Rotation(z=0.25).as_quaternion())
        Z:¼
        >>> Rotation.from_quaternion(Rotation(z=-0.25).as_quaternion())
        Z:¾
        >>> Rotation.from_quaternion(Rotation(x=0.5).as_quaternion())
        X:½
        >>> Rotation.from_quaternion(Rotation(y=0.5).as_quaternion())
        Y:½
        >>> Rotation.from_quaternion(Rotation(z=0.5).as_quaternion())
        Z:½
        """
        turns = 2*atan2_tau(math.sqrt(q.x**2 + q.y**2 + q.z**2), q.w)
        smoothed_turns = smooth_near_quarter_turn(turns)
        d = sinc_tau(smoothed_turns/2)/2
        x, y, z = q.x/d, q.y/d, q.z/d
        sx, sy, sz = smooth_near_quarter_turn(x), smooth_near_quarter_turn(y), smooth_near_quarter_turn(z)
        return Rotation(sx, sy, sz)

    def __canonical(self):
        (x, y, z), t = self.axis(), self.turns()

        # flip axis vector if its first non-zero coordinate is negative
        s = x
        if s == 0:
            s = y
        if s == 0:
            s = z
        if s < 0:
            t, x, y, z = -t, -x, -y, -z

        # remove extra turns, keeping t as close to zero as possible
        t %= 1
        if t == 0:
            return 0, 0, 0, 0
        if t >= 0.5:
            t -= 1

        return t, x, y, z

    def __eq__(self, other):
        """
        >>> Rotation() == Rotation(x=1)
        True
        >>> Rotation(z=-2) == Rotation(y=1)
        True
        >>> Rotation(x=0.25) == Rotation(x=0.25)
        True
        >>> Rotation(x=0.5) == Rotation(x=-0.5)
        True
        >>> Rotation(y=0.25) == Rotation(y=-0.75)
        True

        >>> Rotation() == Rotation(x=0.25)
        False
        >>> Rotation(x=0.25) == Rotation(y=0.25)
        False
        >>> Rotation(y=0.5) == Rotation(z=0.5)
        False
        """
        return self.__canonical() == other.__canonical()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """
        >>> Rotation(x=0.25).__hash__() == Rotation(x=-0.75).__hash__()
        True
        """
        return self.__canonical().__hash__()

    def __neg__(self):
        return Rotation(-self.x, -self.y, -self.z)

    def __repr__(self):
        """
        >>> Rotation()
        (no rotation)
        >>> Rotation(x=1)
        (no rotation)
        >>> Rotation(x=0.0001)
        (negligible rotation)
        >>> Rotation(x=0.5)
        X:½
        >>> Rotation(y=-0.25)
        Y:¾
        >>> Rotation(y=0.75)
        Y:¾
        >>> Rotation(z=0.25)
        Z:¼
        """
        t = self.turns() % 1
        if t == 0:
            return "(no rotation)"
        if t < 0.001 or t > 0.999:
            return "(negligible rotation)"

        if self.y == 0 and self.z == 0:
            return "X:" + str_fraction(self.x % 1)
        if self.x == 0 and self.z == 0:
            return "Y:" + str_fraction(self.y % 1)
        if self.x == 0 and self.y == 0:
            return "Z:" + str_fraction(self.z % 1)

        return (self.x, self.y, self.z).__repr__()

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def plus_rotation_simplified(prev_rotations, next_rotation):
        """
        :param prev_rotations: The sequence of rotations already performed.
        :param next_rotation: The next rotation to perform.
        :return A possibly simplified sequence of rotations leading to the same final orientation.

        >>> Rotation.plus_rotation_simplified([], Rotation(x=0.25))
        [X:¼]
        >>> Rotation.plus_rotation_simplified([Rotation(x=0.25)], Rotation(y=-0.25))
        [X:¼, Y:¾]
        >>> Rotation.plus_rotation_simplified([Rotation(x=0.25)], Rotation(x=0.25))
        [X:¼, X:¼]

        >>> Rotation.plus_rotation_simplified([Rotation(x=0.25)], Rotation(x=-0.25))
        []
        >>> Rotation.plus_rotation_simplified([Rotation(y=0.25)], Rotation(y=-0.25))
        []
        >>> Rotation.plus_rotation_simplified([Rotation(z=-0.25)], Rotation(z=0.25))
        []

        >>> Rotation.plus_rotation_simplified([Rotation(x=0.25), Rotation(x=0.25)], Rotation(x=0.25))
        [X:¾]
        >>> Rotation.plus_rotation_simplified([Rotation(y=-0.25), Rotation(y=-0.25)], Rotation(y=-0.25))
        [Y:¼]
        >>> Rotation.plus_rotation_simplified([Rotation(z=0.25), Rotation(z=0.25)], Rotation(z=0.25))
        [Z:¾]
        """
        n = len(prev_rotations)
        if n >= 1 and next_rotation == -prev_rotations[-1]:
            return prev_rotations[:-1]
        if n >= 2 and next_rotation.turns() == 0.25 \
                and next_rotation == prev_rotations[-1] \
                and next_rotation == prev_rotations[-2]:
            return prev_rotations[:-2] + [-next_rotation]
        return prev_rotations + [next_rotation]
