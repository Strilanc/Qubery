#!/usr/bin/python
# coding=utf-8

"""
Geometric utility methods used by cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
import numpy as np
import math
import string


def tensor_product(m1, m2):
    """
    Tiles a matrix over another's entries, scaling the tile's entries by the covered entry.
    If the input matrices have size w1 by h1 and w2 by h2, then the resulting matrix is w1*w2 by h1*h2.
    :param m1: A numpy matrix into which the other matrix is tiled.
    :param m2: A numpy matrix tiled into the other matrix.

    >>> tensor_product(np.mat([[0,1],[1,0]]), np.mat([[0,2],[1,0]]))
    matrix([[0, 0, 0, 2],
            [0, 0, 1, 0],
            [0, 2, 0, 0],
            [1, 0, 0, 0]])
    >>> tensor_product(np.mat([[0,3],[5,0]]), np.mat([[0,1],[1,0]]))
    matrix([[0, 0, 0, 3],
            [0, 0, 3, 0],
            [0, 5, 0, 0],
            [5, 0, 0, 0]])
    """
    return np.kron(m1, m2)


def tensor_power(m, p):
    """
    Tensor products a matrix against itself the given number of times (minus one).
    :param m: A numpy matrix.
    :param p: A non-negative integer.

    >>> tensor_power(np.mat([[0,1],[1,0]]), 0)
    matrix([[1]])
    >>> tensor_power(np.mat([[0,1],[1,0]]), 1)
    matrix([[0, 1],
            [1, 0]])
    >>> tensor_power(np.mat([[0,1],[1,0]]), 2)
    matrix([[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]])
    """
    return reduce(tensor_product, [m for _ in range(p)], np.mat([[1]]))


def sign(n):
    """
    Determines if a number is negative, zero, or positive.

    :param n: The number to be classified as positive, zero, or negative.
    :return: -1 for negative, 0 for zero, +1 for positive.

    >>> sign(1.5) == 1
    True
    >>> sign(0.01) == 1
    True
    >>> sign(0) == 0
    True
    >>> sign(-1) == -1
    True
    >>> sign(-100) == -1
    True
    """
    if n < 0:
        return -1
    if n > 0:
        return +1
    return 0


def vector_angle(d):
    """
    Determines the angle a vector is pointing along.

    :param d: The vector.
    :return: The angle.
    """
    return math.atan2(d[1], d[0])


def vector_dif(end, start):
    """
    Returns the difference between two vectors; the amount that must be added to the second to get the first.

    :param end: The left hand side of the subtraction.
    :param start: The right hand side of the subtraction.

    >>> vector_dif((2,3), (5,7)) == (-3, -4)
    True
    """
    return end[0] - start[0], end[1] - start[1]


def vector_sum(u, v):
    """
    Returns the sum of two vectors.

    :param u: The left hand side of the sum.
    :param v: The right hand side of the sum.

    >>> vector_sum((2,3), (5,7)) == (7, 10)
    True
    """
    return u[0] + v[0], u[1] + v[1]


def dot(u, v):
    """
    Returns the perpendicular dot product of two vectors, which is +-1 when they are parallel and 0 when they are
    perpendicular.

    :param u: The left hand side of the dot product.
    :param v: The right hand side of the dot product.

    >>> dot((1, 0), (0, 1)) == 0
    True
    >>> dot((0, 1), (1, 0)) == 0
    True
    >>> dot((0, 1), (1, 1)) == 1
    True
    >>> dot((0, 1), (0, 1)) == 1
    True
    >>> dot((1, 1), (-2, -2)) == -4
    True
    """
    return u[0] * v[0] + u[1] * v[1]


def perp(v):
    """
    Returns the input vector, but rotated x-to-y-to-minusX-to-minusY-ward by a quarter turn.

    :param v: The vector to rotate.

    >>> perp((0, 1)) == (-1, 0)
    True
    >>> perp((1, 0)) == (0, 1)
    True
    """
    return -v[1], v[0]


def perp_dot(u, v):
    """
    Returns the perpendicular dot product of two vectors, which is +-1 when they are perpendicular and 0 when they are
    parallel. Basically the result is the z component of the cross product of the two vectors.

    :param u: The vector rotated before applying the dot product.
    :param v: The vector kept the same (i.e. not rotated) before applying the dot product.

    >>> perp_dot((1, 0), (0, 1)) == 1
    True
    >>> perp_dot((0, 1), (1, 0)) == -1
    True
    >>> perp_dot((0, 1), (1, 1)) == -1
    True
    >>> perp_dot((0, 1), (0, 1)) == 0
    True
    >>> perp_dot((1, 1), (-2, -2)) == 0
    True
    """
    return dot(perp(u), v)


def vector_length(v):
    """
    Returns the euclidean length of a vector.

    :param v: An (x,y) vector.

    >>> vector_length((1, 0)) == 1
    True
    >>> vector_length((0, 2)) == 2
    True
    >>> vector_length((3, 4)) == 5
    True
    >>> vector_length((3, -4)) == 5
    True
    """
    return math.sqrt(dot(v, v))


def unit_dot(u, v):
    """
    Returns the dot product of the unit vectors parallel to the input vectors.
    Has a singularity near 0.

    :param u: The left hand side of the dot product.
    :param v: The right hand side of the dot product.

    >>> unit_dot((1, 1), (1, -1)) == 0
    True
    >>> unit_dot((1, 0), (2, 0)) == 1
    True
    >>> unit_dot((1, 0), (-0.5, 0)) == -1
    True
    >>> abs(unit_dot((1, 1), (2, 2)) - 1) < 0.0000001
    True
    >>> abs(unit_dot((1, 1), (-2, -2)) - -1) < 0.0000001
    True
    >>> abs(unit_dot((1, 1), (100, 0)) - math.sqrt(0.5)) < 0.0000001
    True
    """
    return dot(u, v) / vector_length(u) / vector_length(v)


def point_distance(p, q):
    """
    Returns the euclidean distance between two points.

    :param p: An (x,y) point.
    :param q: An (x,y) point.

    >>> point_distance((1, 1), (1, 1)) == 0
    True
    >>> point_distance((0, 0), (3, 4)) == 5
    True
    >>> point_distance((0, 0), (3, -4)) == 5
    True
    """
    return vector_length(vector_dif(p, q))


def ratio_distance(v1, v2):
    """
    Returns the larger of v1/v2 and v2/v1.

    :param v1: A positive number.
    :param v2: A positive number.

    >>> ratio_distance(1, 1) == 1
    True
    >>> ratio_distance(1, 2) == 2
    True
    >>> ratio_distance(2, 1) == 2
    True
    >>> ratio_distance(3, 2) == 1.5
    True
    >>> ratio_distance(5, 0.5) == 10
    True
    """
    if v2 > v1:
        return v2 / v1
    return v1 / v2


def lerp(r, s, t):
    """
    Linearly interpolates between two values.

    :param r: The result when t is zero.
    :param s: The result when t is one.
    :param t: The lerp slider number.

    >>> lerp(3, 7, 0)
    3
    >>> lerp(3, 7, 1)
    7
    >>> lerp(3, 7, -1)
    -1
    >>> lerp(3, 7, 2)
    11
    >>> lerp(3, 7, 0.5)
    5.0
    >>> (lerp(np.array([1, 2, 3]), np.array([16, 25, 35]), 0.5) == np.array([8.5, 13.5, 19])).all()
    True
    """
    return r + (s - r) * t


def vector_lerp(u, v, t):
    """
    Linearly interpolates between two vectors.

    :param u: The resulting (x,y) vector when t is zero.
    :param v: The resulting (x,y) vector when t is one.
    :param t: The lerp slider number.

    >>> vector_lerp((2,3), (5,7), 11)
    (35, 47)
    """
    return lerp(u[0], v[0], t), lerp(u[1], v[1], t)


def offset_point_by_vector_weighted(p, d, w):
    """
    Returns the result of adding a scaled vector to a point.

    :param p: The (x, y) point being offset.
    :param d: The (x, y) vector being scaled and added to the point.
    :param w: The numeric factor used to scale the vector.

    >>> offset_point_by_vector_weighted((2, 3), (5, 7), 11) == (57, 80)
    True
    """
    return p[0] + d[0] * w, p[1] + d[1] * w


def vector_sum_weighted(v1, w1, v2, w2):
    """
    Returns the weighted sum of two vectors.

    :param v1: The first (x, y) vector.
    :param w1: The numeric factor used to scale the first vector.
    :param v2: The second (x, y) vector.
    :param w2: The numeric factor used to scale the second vector.

    >>> vector_sum_weighted((2, 3), 5, (7, 11), 13) == (101, 158)
    True
    """
    return vector_sum(vector_scale(v1, w1), vector_scale(v2, w2))


def line_delta(line_seg):
    """
    Determines the displacement of a line segment; what needs to be added to the start point to get the end point.

    :param line_seg: An ((x1,y1),(x2,y2)) line segment.

    >>> line_delta(((0, 0), (1, 0))) == (1, 0)
    True
    >>> line_delta(((0, 0), (0, 1))) == (0, 1)
    True
    >>> line_delta(((0, 0), (1, -11))) == (1, -11)
    True
    >>> line_delta(((2, 3), (5, 7))) == (3, 4)
    True
    """
    return vector_dif(line_seg[1], line_seg[0])


def line_angle(line_seg):
    """
    Determines the direction of a line segment from start point to end point, with 0 being positive-X-ward and pi/2
    being positive-Y-ward.

    :param line_seg: An ((x1,y1),(x2,y2)) line segment.

    >>> line_angle(((0, 0), (1, 0))) == 0
    True
    >>> line_angle(((0, 0), (0, 1))) == math.pi/2
    True
    >>> line_angle(((0, 0), (-1, 0))) == math.pi
    True
    >>> line_angle(((0, 0), (0, -1))) == -math.pi/2
    True
    >>> line_angle(((0, 0), (1, 1))) == math.pi/4
    True
    >>> line_angle(((-1, 0), (0, 1))) == math.pi/4
    True
    """
    return vector_angle(vector_dif(line_seg[1], line_seg[0]))


def line_intersection(line1, line2, error=0.1, max_distance_from_segment_factor=None):
    """
    Determines if, how, and where two lines intersect.
    Each line is specified as a tuple of two points.

    :param line1: An ((x1,y1),(x2,y2)) line segment.
    :param line2: An ((x1,y1),(x2,y2)) line segment.
    :param error: Determines how strict the parallel check is. Larger values ignore more almost-parallel intersections.
    :param max_distance_from_segment_factor: Determines where along the extended lines the intersection can be.
     If None, then anywhere.
     If zero, then must be on the line segments.
     If one, the intersection point must be less than the segment's distance away from an endpoint of the segment.
     etc

    Returns the intersection of the two lines in two coordinates systems, or else (None, None) if the lines are
    ~parallel.
    The first component of the result is the intersection in euclidean space.
    The second component is the result is the distance you must travel along each line, with the distance between the
    two points being '1', to reach the intersection.

    >>> line_intersection(((0,0),(100,0)), ((50,50),(50,-50))) == ((50, 0), (0.5, 0.5))
    True
    >>> line_intersection(((0,0),(200,200)), ((0,100),(100,0))) == ((50, 50), (0.25, 0.5))
    True
    >>> line_intersection(((0,0),(20,20)), ((0,100),(100,0))) == ((50, 50), (2.5, 0.5))
    True
    >>> line_intersection(((0,0),(20,20)), ((0,100),(100,0)), max_distance_from_segment_factor = 1.51) \
        == ((50, 50), (2.5, 0.5))
    True
    >>> line_intersection(((0,0),(20,20)), ((0,100),(100,0)), max_distance_from_segment_factor = 1.49) \
        == (None, None)
    True
    >>> line_intersection(((0,0),(-20,-20)), ((0,100),(100,0))) == ((50, 50), (-2.5, 0.5))
    True
    >>> line_intersection(((0,0),(20,20)), ((0,1),(20,21))) == (None, None)
    True
    """
    d1 = line_delta(line1)
    d2 = line_delta(line2)

    den = perp_dot(d1, d2)
    if abs(den) <= error:
        return None, None

    du = vector_dif(line2[0], line1[0])
    ua = perp_dot(du, d2) / den
    ub = perp_dot(du, d1) / den
    c = offset_point_by_vector_weighted(line1[0], d1, ua)

    if max_distance_from_segment_factor is not None:
        f = max_distance_from_segment_factor
        if ua < -f or ub < -f or ua > 1 + f or ub > 1 + f:
            return None, None

    return c, (ua, ub)


def line_shift(line, sweep_factor):
    """
    Sweeps a line segment forwarded by its length, scaled by a sweep factor.

    :param line: An ((x1,y1),(x2,y2)) line segment.
    :param sweep_factor: How much to advance the line's first end point along its direction, using the line's length
      as the unit of measurement.

    >>> line_shift(((1, 1), (2, 2)), 0) == ((1, 1), (2, 2))
    True
    >>> line_shift(((1, 1), (2, 2)), 1) == ((2, 2), (3, 3))
    True
    >>> line_shift(((1, 1), (2, -2)), -2) == ((-1, 7), (0, 4))
    True
    """
    d = line_delta(line)
    b = offset_point_by_vector_weighted(line[0], d, sweep_factor)
    return b, vector_sum(b, d)


def sort_lines_by_slope(lines):
    """
    Sorts a list of lines so that lines with similar slopes are adjacent (except the first and last lines may be close).
    Line segments that are parallel, but going in opposite directions, are considered to have the same slope.

    :param lines: A list of ((x1,y1),(x2,y2)) line segments.

    >>> sort_lines_by_slope([((0,0),(1,1)), ((0,0),(-1,1)), ((0,10),(0,-1))]) \
        == [((0,0),(1,1)), ((0,10),(0,-1)), ((0,0),(-1,1))]
    True
    """
    return sorted(lines, key=lambda e: line_angle(e) % math.pi)


def are_lines_nearly_parallel(r, s, max_error=0.01):
    """
    Determines if two lines are nearly parallel.

    :param r: An ((x1,y1),(x2,y2)) line segment.
    :param s: An ((x1,y1),(x2,y2)) line segment.
    :param max_error: Determines how forgiving the check is.

    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((0,0), (1,1)))
    True
    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((0,0), (5,5)))
    True
    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((0,0), (-1,-1)))
    True
    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((0,0), (-1,1)))
    False
    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((0,0), (-1,-10)))
    False
    >>> are_lines_nearly_parallel(((0,0), (1,1)), ((10,10), (-11,-11)))
    True
    """
    return abs(unit_dot(perp(line_delta(r)), line_delta(s))) <= max_error


def point_distance_from_line_segment(point, line):
    """
    Determines the distance between a point and a line segment, accounting for edge cases.

    :param point: An (x, y) point.
    :param line: An ((x1,y1),(x2,y2)) line segment.

    >>> (point_distance_from_line_segment((-1, 1), ((0, 0), (5, 0))) - math.sqrt(2)) < 0.00001
    True
    >>> point_distance_from_line_segment((0, 1), ((0, 0), (5, 0))) == 1
    True
    >>> point_distance_from_line_segment((1, 1), ((0, 0), (5, 0))) == 1
    True
    >>> point_distance_from_line_segment((5, -2), ((0, 0), (5, 0))) == 2
    True
    >>> point_distance_from_line_segment((8, 4), ((0, 0), (5, 0))) == 5
    True
    >>> (point_distance_from_line_segment((2, 1), ((1, 1), (7, 7))) - math.sqrt(0.5)) < 0.00001
    True
    """
    d = line_delta(line)
    dp = vector_dif(point, line[0])
    s = line_length(line)

    du = dot(d, dp)
    if du < 0:
        return point_distance(point, line[0])
    if du > s:
        return point_distance(point, line[1])
    return abs(perp_dot(d, dp)) / s


def line_length(line):
    """
    Determines the length between a line segment's end points.

    :param line: An ((x1,y1),(x2,y2)) line segment.
    """
    return vector_length(line_delta(line))


def int_point(p):
    """
    Rounds a point's coordinates to be integers.

    :param p: An (x, y) point.
    """
    return int(round(p[0])), int(round(p[1]))


def vector_scale(v, c):
    """
    Scales a vector.

    :param v: An (x, y) vector.
    :param c: A numeric factor.
    """
    return v[0] * c, v[1] * c


def vector_average(vectors):
    """
    Returns the center of mass of a list of vectors.

    :param vectors: A length-able iterable collection of (x,y) vectors.

    >>> vector_average([(0,0), (6,6), (0, 6)]) == (2, 4)
    True
    """
    return vector_scale(reduce(lambda e1, e2: vector_sum(e1, e2), vectors, (0, 0)), 1 / len(vectors))


def avg(p, q):
    """
    Returns the average of two points.

    :param p: An (x,y) point.
    :param q: An (x,y) point.
    """
    return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2


def winded(points):
    """
    Returns the same points, but sorted into a clockwise order (as long as no point is inside the other three).

    :param points: A list of (x,y) points.
    """
    mid = np.average(np.array(points, np.float32), axis=0)
    return sorted(points, key=lambda p: vector_angle(vector_dif(mid, p)))


def partial_sums(items):
    """
    Adds up the items in a list, returning the sum before/after each item.

    :param items: An iterable list of items to add up.
    """
    total = 0
    result = [0]
    for e in items:
        total += e
        result.append(total)
    return result


def cycle_windows(cycle_list, span):
    """
    Returns all the contiguous sublists in a list of a specific length, assuming the first item is after the last item.
    :param cycle_list: The list from which to extract sublists.
    :param span: The size of the sublists.

    >>> cycle_windows([], 0)
    []
    >>> cycle_windows([1, 2], 0)
    [[], []]
    >>> cycle_windows([1, 2], 1)
    [[1], [2]]
    >>> cycle_windows([1, 2], 2)
    [[1, 2], [2, 1]]
    >>> cycle_windows([1, 2, 3], 2)
    [[1, 2], [2, 3], [3, 1]]
    """
    n = len(cycle_list)
    if span > n:
        raise ValueError("span > len(cycle_list)")
    return list([cycle_list[i:i+span] if i + span <= n else cycle_list[i:n] + cycle_list[0:i+span-n]
                 for i in range(n)])
