# coding=utf-8

"""
Contains the Quaternion type.
"""

from __future__ import division
import math
import trig_tau
import string


class Quaternion(object):
    """
    A type of number that extends the complex numbers, useful for representing rotations.
    """
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        """
        >>> Quaternion() == 0
        True
        >>> 0 == Quaternion()
        True
        >>> Quaternion() == Quaternion()
        True
        >>> Quaternion(w=5) == 5
        True
        >>> 5 == Quaternion(w=5)
        True
        >>> Quaternion(w=5, x=4) == 5 + 4j
        True
        >>> 3 + 2j == Quaternion(w=3, x=2)
        True
        >>> Quaternion(w=1, x=2, y=3, z=4) == Quaternion(w=1, x=2, y=3, z=4)
        True

        >>> Quaternion(1) == 2
        False
        >>> Quaternion(w=1, x=2, y=3, z=4) == Quaternion(w=1, x=2, y=3, z=5)
        False
        >>> Quaternion(w=1, x=2, y=3, z=4) == Quaternion(w=1, x=2, y=5, z=4)
        False
        >>> Quaternion(w=1, x=2, y=3, z=4) == Quaternion(w=1, x=5, y=3, z=4)
        False
        >>> Quaternion(w=1, x=2, y=3, z=4) == Quaternion(w=5, x=2, y=3, z=4)
        False
        """
        if isinstance(other, (int, long, float)):
            return self == Quaternion(w=other)
        if isinstance(other, complex):
            return self == Quaternion(w=other.real, x=other.imag)
        return self.w == other.w \
            and self.x == other.x \
            and self.y == other.y \
            and self.z == other.z

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """
        >>> Quaternion(1, 2, 3, 4).__hash__() == Quaternion(1, 2, 3, 4).__hash__()
        True
        """
        return (self.w, self.x, self.y, self.z).__hash__()

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        >>> Quaternion()
        0
        >>> Quaternion(w=1)
        1
        >>> Quaternion(w=-1)
        -1
        >>> Quaternion(w=2)
        2
        >>> Quaternion(x=1)
        i
        >>> Quaternion(x=-1)
        -i
        >>> Quaternion(x=2)
        2i
        >>> Quaternion(y=1)
        j
        >>> Quaternion(y=-1)
        -j
        >>> Quaternion(y=2)
        2j
        >>> Quaternion(z=1)
        k
        >>> Quaternion(z=-1)
        -k
        >>> Quaternion(z=2)
        2k

        >>> Quaternion(w=2, z=3)
        2 + 3k
        >>> Quaternion(w=2, z=-3)
        2 - 3k
        >>> Quaternion(2, 3, 5, 7)
        2 + 3i + 5j + 7k
        >>> Quaternion(2, 2, 2, 2)
        2 + 2i + 2j + 2k
        >>> Quaternion(0.5, -1.5, 3.25, 4)
        0.5 - 1.5i + 3.25j + 4k
        """
        if self == 0:
            return "0"

        v = [self.w, self.x, self.y, self.z]
        signs = ["+" if e > 0 else "-" for e in v]
        units = ["", "i", "j", "k"]
        abs_reps = [str(int(round(abs(e)))) if e == round(e) or str(abs(e)).endswith(".0") else str(abs(e)) for e in v]

        unit_repr = lambda sign, value, unit: sign + ("" if value == "1" and unit != "" else value) + unit
        unit_reps = [unit_repr(signs[i], abs_reps[i], units[i])
                     for i in range(4)
                     if v[i] != 0]
        head_rep = unit_reps[0][0 if unit_reps[0][0] == "-" else 1:]
        tail_reps = [" " + e[0] + " " + e[1:] for e in unit_reps[1:]]
        return head_rep + string.join(tail_reps, "")

    def __neg__(self):
        """
        Returns the additive inverse of this quaternion.

        >>> -Quaternion(1, 2, 3, 4)
        -1 - 2i - 3j - 4k
        """
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __invert__(self):
        """
        Returns the conjugate of this quaternion, with the non-real components negated.

        >>> ~Quaternion(2, 3, 5, 7)
        2 - 3i - 5j - 7k
        >>> ~~Quaternion(2, 3, -5, 7)
        2 + 3i - 5j + 7k
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __add__(self, other):
        """
        >>> Quaternion(2, 3, 5, 7) + Quaternion(11, 13, 17, 19)
        13 + 16i + 22j + 26k
        """
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z)

    def __sub__(self, other):
        """
        >>> Quaternion(2, 3, 5, 7) - Quaternion(11, 13, 17, 23)
        -9 - 10i - 12j - 16k
        """
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z)

    def __mul__(self, other):
        """
        Returns the product of two quaternions.

        :param other: The right hand side value.

        >>> Quaternion(2, 3, 5, 7) * 10
        20 + 30i + 50j + 70k
        >>> Quaternion(w=1) * Quaternion(w=1)
        1
        >>> Quaternion(x=1) * Quaternion(x=1)
        -1
        >>> Quaternion(y=1) * Quaternion(y=1)
        -1
        >>> Quaternion(z=1) * Quaternion(z=1)
        -1

        >>> Quaternion(x=1) * Quaternion(y=1)
        k
        >>> Quaternion(y=1) * Quaternion(x=1)
        -k
        >>> Quaternion(y=1) * Quaternion(z=1)
        i
        >>> Quaternion(z=1) * Quaternion(y=1)
        -i
        >>> Quaternion(z=1) * Quaternion(x=1)
        j
        >>> Quaternion(x=1) * Quaternion(z=1)
        -j

        >>> Quaternion(x=1) * Quaternion(y=1) * Quaternion(z=1)
        -1
        >>> Quaternion(x=1) * Quaternion(z=1) * Quaternion(y=1)
        1

        >>> Quaternion(x=1) * Quaternion(y=1) == Quaternion(z=1)
        True
        >>> Quaternion(y=1) * Quaternion(z=1) == Quaternion(x=1)
        True
        >>> Quaternion(z=1) * Quaternion(x=1) == Quaternion(y=1)
        True

        >>> Quaternion(2, 3, 5, 7) * Quaternion(11, 13, 17, 19) == Quaternion(-235, 35, 123, 101)
        True
        """
        if isinstance(other, (long, int, float)):
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other)
        v = self
        d = other
        return Quaternion(
            v.w*d.w - v.x*d.x - v.y*d.y - v.z*d.z,
            v.w*d.x + v.x*d.w + v.y*d.z - v.z*d.y,
            v.w*d.y - v.x*d.z + v.y*d.w + v.z*d.x,
            v.w*d.z + v.x*d.y - v.y*d.x + v.z*d.w)

    def left_divide_by(self, denominator):
        """
        Returns the value of x that satisfies self == x * denominator.
        :param denominator: When the result is left-multiplied onto it, the result is the receiving quaternion.

        >>> Quaternion(2, 3, 5, 7).left_divide_by(2)
        1 + 1.5i + 2.5j + 3.5k
        >>> all([a.left_divide_by(b) * b == a \
                 for a in [Quaternion(1), Quaternion(x=1), Quaternion(y=1), Quaternion(z=1)] \
                 for b in [Quaternion(1), Quaternion(x=1), Quaternion(y=1), Quaternion(z=1)]])
        True
        >>> Quaternion(2, 3, 5, 7).left_divide_by(Quaternion(1, 2, 3, 4))
        1.7 - 0.1j
        >>> Quaternion(2, 3, 5, 7).left_divide_by(Quaternion(1, 2, 3, 4)) * Quaternion(1, 2, 3, 4)
        2 + 3i + 5j + 7k
        """
        if isinstance(denominator, (int, float, long)):
            return self / denominator
        return (self * ~denominator) / denominator.norm()

    def right_divide_by(self, denominator):
        """
        Returns the value of x that satisfies self == denominator * x.
        :param denominator: When the result is right-multiplied onto it, the result is the receiving quaternion.

        >>> Quaternion(2, 3, 5, 7).right_divide_by(2)
        1 + 1.5i + 2.5j + 3.5k
        >>> all([b * a.right_divide_by(b) == a \
                 for a in [Quaternion(1), Quaternion(x=1), Quaternion(y=1), Quaternion(z=1)] \
                 for b in [Quaternion(1), Quaternion(x=1), Quaternion(y=1), Quaternion(z=1)]])
        True
        >>> abs(Quaternion(2, 3, 5, 7).right_divide_by(Quaternion(1, 2, 3, 4)) - Quaternion(1.7, -2/30, 1/30, -2/30)) \
                < 0.0001
        True
        >>> Quaternion(1, 2, 3, 4) * Quaternion(2, 3, 5, 7).right_divide_by(Quaternion(1, 2, 3, 4))
        2 + 3i + 5j + 7k
        """
        if isinstance(denominator, (int, float, long)):
            return self / denominator
        return (~denominator * self) / denominator.norm()

    def __div__(self, real_denominator):
        """
        Divides this quaternion by a scalar.
        :param real_denominator: The real number to divide by.
        """
        return self.__truediv__(real_denominator)

    def __truediv__(self, real_denominator):
        """
        Divides this quaternion by a scalar.
        :param real_denominator: The real number to divide by.

        >>> Quaternion(2, 3, 5, 7) / 2
        1 + 1.5i + 2.5j + 3.5k
        >>> Quaternion(1, 2, -3, 4) / 5.0
        0.2 + 0.4i - 0.6j + 0.8k
        """
        d = real_denominator
        if isinstance(d, (int, float, long)):
            if d == 0:
                raise ZeroDivisionError()
            return Quaternion(
                self.w / d,
                self.x / d,
                self.y / d,
                self.z / d)
        if isinstance(d, Quaternion):
            raise ValueError("Quaternion/quaternion division is ambiguous. Use right_divide_by or left_divide_by.")
        raise ValueError("Unsupported division")

    def rotate(self, other):
        """
        Uses this quaternion to rotate the vector component of another quaternion.
        :param other: The quaternion to be rotated.

        >>> Quaternion(0, 1, 0, 0).rotate(Quaternion(x=1)) == Quaternion(x=1)
        True
        >>> Quaternion(0, 1, 0, 0).rotate(Quaternion(y=1)) == Quaternion(y=-1)
        True
        >>> abs(Quaternion(math.sqrt(0.5), math.sqrt(0.5), 0, 0).rotate(Quaternion(y=1)) - Quaternion(z=1)) < 0.0001
        True
        """
        return self * other * ~self

    def dot(self, other):
        """
        Returns the dot product of the scalar coefficients of two quaternions.
        :param other: The other quaternion.

        >>> Quaternion(2, 3, 5, 7).dot(Quaternion(11, 13, 17, 19))
        279
        """
        return self.w * other.w \
            + self.x * other.x \
            + self.y * other.y \
            + self.z * other.z

    def slerp(self, other, t):
        """
        Spherical linear interpolation between two quaternions.
        :param other: Destination quaternion at t=1.
        :param t: Interpolation factor.

        >>> Quaternion(1).slerp(Quaternion(1), 0.5)
        1
        >>> abs(Quaternion(math.sin(0.1), math.cos(0.1)).slerp(Quaternion(math.sin(0.2), math.cos(0.2)), 0.5) \
                - Quaternion(math.sin(0.15), math.cos(0.15))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(2), math.cos(2)), 0) \
                - Quaternion(math.sin(1), math.cos(1))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(2), math.cos(2)), 0.1) \
                - Quaternion(math.sin(1.1), math.cos(1.1))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(2), math.cos(2)), 0.5) \
                - Quaternion(math.sin(1.5), math.cos(1.5))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(2), math.cos(2)), 0.9) \
                - Quaternion(math.sin(1.9), math.cos(1.9))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(2), math.cos(2)), 1) \
                - Quaternion(math.sin(2), math.cos(2))) < 0.00001
        True
        >>> abs(Quaternion(math.sin(1), math.cos(1)).slerp(Quaternion(math.sin(-1), math.cos(-1)), 0.5) \
                - Quaternion(0, 1)) < 0.00001
        True
        """
        theta = trig_tau.acos(self.dot(other))
        a = trig_tau.sin_scale_ratio(theta, 1-t)
        b = trig_tau.sin_scale_ratio(theta, t)
        return self*a + other*b

    def norm(self):
        """
        :return: The squared length of the quaternion.

        >>> Quaternion(1, 2, 3, 4).norm()
        30
        """
        return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z

    def __abs__(self):
        """
        :return: The length of the quaternion.

        >>> abs(Quaternion(-2, 2, 4, 5))
        7.0
        >>> abs(Quaternion(1, 0, 0, 0))
        1.0
        """
        return math.sqrt(self.norm())
