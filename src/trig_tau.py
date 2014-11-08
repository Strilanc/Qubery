#!/usr/bin/python
# coding=utf-8

"""
Trigonometric utility methods where angles are specified in fractions of a full turn, as opposed to in radians or
degrees.

The advantage of using fractions of a turn, instead of radians, is that more conversions between polar coordinates and
Euclidean coordinates can be done exactly (without introducing floating point error). For example, trig_tau.atan(1, 1)
returns exactly 0.125 instead of an approximation of τ/8.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
import math as math_rad


tau = math_rad.pi * 2


def acos(v):
    """
    Returns an angle fraction which, when given to trig_tau.cos, returns (approximately) the given value.
    :param v: A value between -1 and 1.

    >>> abs(acos(0.25) - 0.20978468837241687811452) < 0.1 ** 14
    True

    # inverts arbitrary values approximately
    >>> abs(cos(acos(1/3)) - 1/3) < 0.1 ** 14
    True
    >>> abs(acos(cos(1/3)) - 1/3) < 0.1 ** 14
    True

    # inverts special values exactly
    >>> cos(acos(1)) == 1
    True
    >>> cos(acos(0)) == 0
    True
    >>> cos(acos(-1)) == -1
    True
    """

    # Exact
    if v == 1:
        return 0
    if v == 0:
        return 0.25
    if v == -1:
        return 0.5

    # Approximate
    return math_rad.acos(v) / tau


def sin_scale_ratio(f, s):
    """
    Returns the ratio sin(f s) / sin(f), handling the near-zero values carefully to avoid singularities.

    :param f: The angle argument, in fractions of a turn, which will be scaled in the numerator but not the denominator.
    :param s: The scaling factor to apply to the angle before sin-ing it in the numerator.

    >>> sin_scale_ratio(0, 1/8) == 1/8
    True
    >>> sin_scale_ratio(0, 1/4) == 1/4
    True

    >>> abs(sin_scale_ratio(1/3, 1/8) - 0.298858490722684508034630) < 0.1 ** 14
    True

    # precision near zero and switch to approximation
    >>> abs(sin_scale_ratio(0.00110, 1/8) - 0.12500097964076610394) < 0.1 ** 14
    True
    >>> abs(sin_scale_ratio(0.00090, 1/8) - 0.12500065579137886541) < 0.1 ** 13
    True
    >>> abs(sin_scale_ratio(0.00021, 1/8) - 0.12500003570407218730) < 0.1 ** 13
    True
    >>> abs(sin_scale_ratio(0.00019, 1/8) - 0.12500002922714192262) < 0.1 ** 13
    True
    >>> abs(sin_scale_ratio(0.00011, 1/8) - 0.12500000979635397322) < 0.1 ** 14
    True
    >>> abs(sin_scale_ratio(0.00009, 1/8) - 0.12500000655788972983) < 0.1 ** 14
    True
    """

    # Near zero, switch to an approximation based on the first two Taylor series terms.
    # This avoids an explosion in floating point error due to the division by tiny values.
    if abs(f < 0.0002):
        d = (tau * f) ** 2 / 6
        return s * (1 - d * s ** 2) / (1 - d)

    return sin(f * s) / sin(f)


def cos(p):
    """
    Returns the cosine of the angle corresponding to a p'th of a turn.
    :param p: The proportion of a full turn to rotate <1, 0> by before returning the resulting vector's x coordinate.

    >>> cos(0)
    1
    >>> cos(0.25)
    0
    >>> cos(0.5)
    -1
    >>> cos(0.75)
    0
    >>> cos(1)
    1
    >>> abs(cos(0.125) - math_rad.sqrt(2)/2) < 0.1 ** 14
    True
    """
    p %= 1

    # Exact
    if p == 0:
        return 1
    if p == 0.25:
        return 0
    if p == 0.5:
        return -1
    if p == 0.75:
        return 0

    # Approximate
    return math_rad.cos(p*tau)


def sin(p):
    """
    Returns the sine of the angle corresponding to a p'th of a turn.
    :param p: The proportion of a full turn to rotate <1, 0> by before returning the resulting vector's y coordinate.

    >>> sin(0)
    0
    >>> sin(0.25)
    1
    >>> sin(0.5)
    0
    >>> sin(0.75)
    -1
    >>> sin(1)
    0
    >>> abs(sin(0.125) - math_rad.sqrt(2)/2) < 0.1 ** 14
    True
    """
    return cos(p-0.25)


def atan2(y, x):
    """
    Determines the angle, measured in fractions of a complete turn, that points toward the given coordinates.

    Results are canonicalized to be in the range [-1/2, 1/2). The zero angle points +x-ward, +y-ward is 1/4, -x-ward is
    -1/2, and -y-ward is -3/4.

    :param y: The y coordinate.
    :param x: The x coordinate.

    >>> atan2(y=0, x=1) == 0.125*0
    True
    >>> atan2(y=1, x=1) == 0.125*1
    True
    >>> atan2(y=1, x=0) == 0.125*2
    True
    >>> atan2(y=1, x=-1) == 0.125*3
    True
    >>> atan2(y=0, x=-1) == 0.125*-4
    True
    >>> atan2(y=-1, x=-1) == 0.125*-3
    True
    >>> atan2(y=-1, x=0) == 0.125*-2
    True
    >>> atan2(y=-1, x=1) == 0.125*-1
    True

    >>> abs(atan2(y=1, x=3) - 0.05120819117478336291229) < 0.1 ** 14
    True
    >>> abs(atan2(y=1, x=-4) - 0.46101043481131533726974) < 0.1 ** 14
    True
    >>> abs(atan2(y=-1, x=-5) - -0.46858352090549940809312) < 0.1 ** 14
    True
    >>> abs(atan2(y=-1, x=2) - -0.07379180882521663708770) < 0.1 ** 14
    True
    """

    # Exact
    if y == 0 and x > 0:
        return +0/8
    if x == y and y > 0:
        return +1/8
    if x == 0 and y > 0:
        return +2/8
    if x == -y and y > 0:
        return +3/8
    if y == 0 and x < 0:
        return -4/8
    if x == y and y < 0:
        return -3/8
    if x == 0 and y < 0:
        return -2/8
    if x == -y and y < 0:
        return -1/8

    # Approximate
    return math_rad.atan2(y, x) / tau


def expi(f):
    """
    Returns Euler's constant e to the power of i times tau times f.

    :param f: The angle argument, in fractions of a turn.

    >>> expi(0) == 1
    True
    >>> expi(0.25) == 1j
    True
    >>> expi(0.5) == -1
    True
    >>> expi(0.75) == -1j
    True
    >>> expi(1) == 1
    True
    >>> abs(expi(0.125) - math_rad.sqrt(2)/2 * (1 + 1j)) < 0.1 ** 14
    True

    >>> abs(expi(1/3) - (-0.5 + 0.86602540378443864676j)) < 0.1 ** 14
    True
    """
    return cos(f) + 1j * sin(f)


def sinc(f):
    """
    The cardinal sine function, but in fractions of a turn, equal to sin(f)/f.
    
    Note that trig_tau.sinc(x) is NOT just a horizontally-squeezed version of the usual radians-based sinc. The scalar
    division uses the fractional turn value instead of radians, so trig_tau.sinc(f) = math_rad.sinc(τ f) / τ.
    
    :param f: The angle argument, in fractions of a turn.

    # known values
    >>> abs(sinc(-0.75) - -4/3) < 0.1 ** 14
    True
    >>> sinc(-0.5) == 0
    True
    >>> sinc(-0.25) == 4
    True
    >>> abs(sinc(-0.125) - 4*math_rad.sqrt(2)) < 0.1 ** 14
    True
    >>> sinc(0) == tau
    True
    >>> abs(sinc(0.125) - 4*math_rad.sqrt(2)) < 0.1 ** 14
    True
    >>> sinc(0.25) == 4
    True
    >>> sinc(0.5) == 0
    True
    >>> abs(sinc(0.75) - -4/3) < 0.1 ** 7
    True

    # precision near switch to approximation
    >>> abs(sinc(0.00110) - 6.28313528383935370279) < 0.1 ** 12
    True
    >>> abs(sinc(0.00090) - 6.28315182045431291640) < 0.1 ** 12
    True
    >>> abs(sinc(0.00021) - 6.283183484010676381994) < 0.1 ** 12
    True
    >>> abs(sinc(0.00019) - 6.283183814744241947267) < 0.1 ** 12
    True
    >>> abs(sinc(0.00011) - 6.283184806945001315912) < 0.1 ** 13
    True
    >>> abs(sinc(0.00009) - 6.283184972311803683807) < 0.1 ** 14
    True
    """

    # Near zero, switch to an approximation based on the first two Taylor series terms.
    # This avoids an explosion in floating point error due to the division by tiny values.
    if abs(f) < 0.0002:
        return tau - f**2 * tau**3 / 6

    return sin(f) / f
