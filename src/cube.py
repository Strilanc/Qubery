#!/usr/bin/python

"""
Classes and utilities related to representing and tracking the orientation of a checkerboard cube with known colors.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
import math

import numpy as np

from rotation import Rotation


class Side(object):
    """
    Data about the side of a checkerboard cube: the colors, an index, and some names.

    The side indexes follow the following patterns:
    - Adding 3 gives the opposite side (mod 6)
    - Adding 1 goes around a corner (mod 3), giving a face next in ZYX axis order
    """
    def __init__(self, index, color1, color2, visual_name, side_name):
        self.index = index
        self.color1 = color1
        self.color2 = color2
        self.visual_name = visual_name
        self.side_name = side_name
        self.axis = index % 3
        self.sign = index // 3
        self.opposite_index = (index + 3) % 6

    def color(self, is_darker):
        return self.color1 \
            if is_darker == (np.max(np.array(self.color1)) < np.max(np.array(self.color2))) \
            else self.color2

    def __str__(self):
        return self.visual_name + " (" + self.side_name + ")"

    def __repr__(self):
        return self.side_name

# home
Front = Side(0, [130, 90, 70], [10, 30, 200], "BlueRed", "Front")
Top = Side(1, [50, 190, 220], [90, 150, 90], "YellowGreen", "Top")
Right = Side(2, [180, 200, 220], [50, 60, 70], "WhiteBlack", "Right")
Back = Side(3, [70, 50, 80], [10, 110, 210], "PurpleOrange", "Back")
Bottom = Side(4, [10, 110, 210], [50, 190, 220], "OrangeYellow", "Bottom")
Left = Side(5, [10, 30, 190], [30, 50, 70], "RedBlack", "Left")


def rgb_distance(rgb1, rgb2):
    """
    Determines a distance between two rgb colors.

    :param rgb1: An (r,g,b) or numpy [r,g,b] color triplet.
    :param rgb2: An (r,g,b) or numpy [r,g,b] color triplet.
    """
    return math.sqrt(np.sum((np.array(rgb1, np.float32) - np.array(rgb2, np.float32))**2))


def color_pair_distance(color_pair_1, color_pair_2):
    """
    Determines how similar two color pairs are, using rgb euclidean distance and ignoring order.
    :param color_pair_1: An rgb color pair.
    :param color_pair_2: Another rgb color pair.

    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 1))) == 0
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((1, 1, 1), (0, 0, 0))) == 0
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((1, 1, 2), (0, 0, 0))) == 0.125
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((1, 2, 1), (0, 0, 0))) == 0.125
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((2, 1, 1), (0, 0, 0))) == 0.125
    True
    """
    lux1 = np.average(color_pair_1)
    lux2 = np.average(color_pair_2)
    dux1 = (np.array(color_pair_1) / max(8, lux1)).tolist()
    dux2 = (np.array(color_pair_2) / max(8, lux2)).tolist()
    ds = list([rgb_distance(c1, c2) for c1 in dux1 for c2 in dux2])
    return min(ds[0] + ds[3], ds[1] + ds[2])


def classify_color_pair_as_side(color_pair):
    """
    Matches the given color pair against the expected color pairs, picking the closest match while trying to ignore
    the effects of lighting changes.
    :color_pair: The color pair to classify.
    :return: The most closely matching Side, by color.

    >>> red_blue_samples = [\
        [[131, 90, 70], [6, 29, 194]],\
        [[127, 87, 74], [7, 31, 192]],\
        [[125, 83, 66], [6, 26, 189]],\
        [[128, 83, 58], [3, 22, 189]],\
        [[130, 88, 71], [6, 27, 193]],\
        [[129, 90, 84], [12, 37, 192]],\
        [[128, 90, 83], [11, 37, 192]],\
        [[130, 83, 53], [2, 21, 190]]]
    >>> yellow_green_samples = [\
        [[53, 188, 223], [95, 124, 97]],\
        [[53, 187, 222], [95, 124, 98]],\
        [[52, 186, 221], [94, 123, 96]],\
        [[40, 187, 222], [95, 117, 78]],\
        [[55, 188, 223], [96, 126, 101]],\
        [[60, 184, 217], [96, 127, 105]],\
        [[60, 187, 221], [96, 127, 106]]]
    >>> white_black_samples = [\
        [[177, 205, 224], [48, 62, 74]],\
        [[177, 205, 225], [49, 62, 74]],\
        [[58, 114, 146], [26, 46, 59]],\
        [[177, 205, 225], [48, 62, 74]],\
        [[174, 203, 222], [44, 54, 66]],\
        [[168, 195, 216], [48, 60, 73]],\
        [[173, 200, 220], [38, 47, 59]]]
    >>> red_black_samples = [\
        [[4, 28, 189], [32, 47, 72]],\
        [[4, 28, 189], [33, 47, 72]],\
        [[4, 27, 188], [32, 46, 71]],\
        [[2, 21, 187], [30, 40, 57]],\
        [[5, 28, 188], [33, 47, 73]],\
        [[4, 26, 187], [33, 46, 72]],\
        [[4, 25, 188], [33, 44, 68]]]
    >>> orange_purple_samples = [\
        [[7, 111, 209], [70, 47, 73]],\
        [[6, 111, 209], [70, 48, 72]],\
        [[5, 110, 209], [72, 45, 69]],\
        [[3, 108, 207], [69, 41, 65]],\
        [[6, 110, 209], [73, 48, 75]],\
        [[10, 111, 208], [72, 53, 83]],\
        [[10, 111, 208], [72, 53, 83]],\
        [[9, 111, 207], [71, 53, 83]]]
    >>> yellow_orange_samples = [\
        [[50, 188, 224], [3, 108, 206]],\
        [[50, 187, 225], [3, 108, 206]],\
        [[50, 186, 223], [5, 110, 206]],\
        [[53, 184, 222], [6, 112, 206]],\
        [[51, 184, 223], [6, 112, 206]],\
        [[50, 184, 222], [6, 112, 206]],\
        [[48, 185, 223], [5, 109, 206]],\
        [[45, 187, 224], [3, 105, 206]],\
        [[47, 186, 223], [5, 104, 204]]]
    >>> [(e, classify_color_pair_as_side(e)) for e in red_blue_samples if classify_color_pair_as_side(e) != Front]
    []
    >>> [(e, classify_color_pair_as_side(e)) for e in yellow_green_samples if classify_color_pair_as_side(e) != Top]
    []
    >>> [(e, classify_color_pair_as_side(e)) for e in white_black_samples if classify_color_pair_as_side(e) != Right]
    []
    >>> [(e, classify_color_pair_as_side(e)) for e in red_black_samples if classify_color_pair_as_side(e) != Left]
    []
    >>> [(e, classify_color_pair_as_side(e)) for e in orange_purple_samples if classify_color_pair_as_side(e) != Back]
    []
    >>> [(e, classify_color_pair_as_side(e)) for e in yellow_orange_samples if classify_color_pair_as_side(e) != Bottom]
    []
    """
    return min(Sides, key=lambda e: color_pair_distance((e.color1, e.color2), color_pair))


# red black measurement:
#[[ 46  36 100]
# [ 46  22  13]]
Sides = [Front, Top, Right, Back, Bottom, Left]


class Facing(object):
    """
    A simplified axis-aligned checkerboard cube orientation.
    """

    def __init__(self, current_front, current_top):
        """
        >>> Facing(None, Front)
        Traceback (most recent call last):
            ...
        ValueError: current_front not in Cube.Sides
        >>> Facing(Front, None)
        Traceback (most recent call last):
            ...
        ValueError: current_top not in Cube.Sides
        >>> Facing(Front, Front)
        Traceback (most recent call last):
            ...
        ValueError: current_top on same axis as current_front
        >>> Facing(Right, Left)
        Traceback (most recent call last):
            ...
        ValueError: current_top on same axis as current_front
        """
        if current_front not in Sides:
            raise ValueError("current_front not in Cube.Sides")
        if current_top not in Sides:
            raise ValueError("current_top not in Cube.Sides")
        if current_front.axis == current_top.axis:
            raise ValueError("current_top on same axis as current_front")
        self.current_front = current_front
        self.current_top = current_top

    def is_top_right_darker(self):
        """
        Determines if the cube's current facing should result in the front having the darker color along the
        bottom-left-to-top-right diagonal, as opposed to along the bottom-right-to-top-left diagonal.

        >>> Facing(Front, Top).is_top_right_darker()
        True
        >>> Facing(Front, Right).is_top_right_darker()
        False
        >>> Facing(Left, Bottom).is_top_right_darker()
        False
        >>> Facing(Right, Back).is_top_right_darker()
        True
        """
        return (self.current_front.axis + 1) % 3 == self.current_top.axis

    def __eq__(self, other):
        return isinstance(other, Facing) \
            and self.current_front == other.current_front \
            and self.current_top == other.current_top

    def __repr__(self):
        """
        >>> Facing(Front, Top)
        Current front=BlueRed, top=YellowGreen
        """
        return "Current front=" + self.current_front.visual_name + ", top=" + self.current_top.visual_name

    def __str__(self):
        return self.__repr__()

    def x(self):
        """
        Rotates the cube's orientation a quarter turn counter-clockwise around the X world axis.

        The coordinate system is right-handed, and the positive X direction is rightward, so this operation moves the
        Top face to where the Front face is while preserving the Right face.

        >>> Facing(Front, Top).x() == Facing(Top, Back)
        True
        >>> Facing(Top, Back).x() == Facing(Back, Bottom)
        True
        >>> Facing(Back, Bottom).x() == Facing(Bottom, Front)
        True
        >>> Facing(Bottom, Front).x() == Facing(Front, Top)
        True

        >>> Facing(Top, Front).x() == Facing(Front, Bottom)
        True
        >>> Facing(Front, Bottom).x() == Facing(Bottom, Back)
        True
        >>> Facing(Bottom, Back).x() == Facing(Back, Top)
        True
        >>> Facing(Back, Top).x() == Facing(Top, Front)
        True

        >>> Facing(Top, Right).x() == Facing(Right, Bottom)
        True
        >>> Facing(Right, Bottom).x() == Facing(Bottom, Left)
        True
        >>> Facing(Bottom, Left).x() == Facing(Left, Top)
        True
        >>> Facing(Left, Top).x() == Facing(Top, Right)
        True

        >>> Facing(Right, Top).x() == Facing(Top, Left)
        True
        >>> Facing(Top, Left).x() == Facing(Left, Bottom)
        True
        >>> Facing(Left, Bottom).x() == Facing(Bottom, Right)
        True
        >>> Facing(Bottom, Right).x() == Facing(Right, Top)
        True
        """

        return Facing(self.current_top, Sides[self.current_front.opposite_index])

    def y(self):
        """
        Rotates the cube's orientation a quarter turn counter-clockwise around the Y world axis.

        The coordinate system is right-handed, and the positive Y direction is upward, so this operation moves the Left
        face to where the Front face is while preserving the Top face.

        >>> Facing(Front, Top).y() == Facing(Left, Top)
        True
        >>> Facing(Left, Front).y() == Facing(Top, Front)
        True
        """
        return self.x().z().x().x().x()

    def z(self):
        """
        Rotates the cube's orientation a quarter turn counter-clockwise around the Z world axis.

        The coordinate system is right-handed, and the positive Z direction is frontward, so this operation moves the
        Right face to where the Top face is while preserving the Front face.

        >>> Facing(Front, Top).z() == Facing(Front, Right)
        True
        >>> Facing(Front, Right).z() == Facing(Front, Bottom)
        True
        >>> Facing(Front, Bottom).z() == Facing(Front, Left)
        True
        >>> Facing(Front, Left).z() == Facing(Front, Top)
        True

        >>> Facing(Top, Right).z() == Facing(Top, Front)
        True
        >>> Facing(Top, Front).z() == Facing(Top, Left)
        True
        >>> Facing(Top, Left).z() == Facing(Top, Back)
        True
        >>> Facing(Top, Back).z() == Facing(Top, Right)
        True

        >>> Facing(Right, Front).z() == Facing(Right, Top)
        True
        >>> Facing(Right, Top).z() == Facing(Right, Back)
        True
        >>> Facing(Right, Back).z() == Facing(Right, Bottom)
        True
        >>> Facing(Right, Bottom).z() == Facing(Right, Front)
        True

        >>> Facing(Back, Right).z() == Facing(Back, Top)
        True
        >>> Facing(Back, Top).z() == Facing(Back, Left)
        True
        >>> Facing(Back, Left).z() == Facing(Back, Bottom)
        True
        >>> Facing(Back, Bottom).z() == Facing(Back, Right)
        True

        >>> Facing(Bottom, Right).z() == Facing(Bottom, Back)
        True
        >>> Facing(Bottom, Back).z() == Facing(Bottom, Left)
        True
        >>> Facing(Bottom, Left).z() == Facing(Bottom, Front)
        True
        >>> Facing(Bottom, Front).z() == Facing(Bottom, Right)
        True

        >>> Facing(Left, Top).z() == Facing(Left, Front)
        True
        >>> Facing(Left, Front).z() == Facing(Left, Bottom)
        True
        >>> Facing(Left, Bottom).z() == Facing(Left, Back)
        True
        >>> Facing(Left, Back).z() == Facing(Left, Top)
        True
        """
        new_axis = self.current_top.axis
        new_sign = self.current_top.sign

        s = 1 if self.current_front.sign == 0 else -1
        new_axis += s
        if new_axis % 3 == self.current_front.axis:
            new_sign += 1
            new_axis += s

        new_side = (new_axis % 3) + (new_sign % 2) * 3
        return Facing(self.current_front, Sides[new_side])

    def rotated_by(self, rotation):
        """
        Returns the result of rotating this Facing by the given rotation.
        :param rotation: The Rotation to rotate by. Must be an axis-aligned rotation.

        >>> Facing(Front, Top).rotated_by(Rotation(x=0.25)) == Facing(Top, Back)
        True
        >>> Facing(Front, Top).rotated_by(Rotation(x=-0.25)) == Facing(Bottom, Front)
        True
        >>> Facing(Front, Top).rotated_by(Rotation(y=0.25)) == Facing(Left, Top)
        True
        >>> Facing(Front, Top).rotated_by(Rotation(y=-0.25)) == Facing(Right, Top)
        True
        >>> Facing(Front, Top).rotated_by(Rotation(z=0.25)) == Facing(Front, Right)
        True
        >>> Facing(Front, Top).rotated_by(Rotation(z=-0.25)) == Facing(Front, Left)
        True
        """
        if rotation == Rotation():
            return self

        if rotation == Rotation(x=0.25):
            return self.x()
        if rotation == Rotation(x=0.5):
            return self.x().x()
        if rotation == Rotation(x=0.75):
            return self.x().x().x()

        if rotation == Rotation(y=0.25):
            return self.y()
        if rotation == Rotation(y=0.5):
            return self.y().y()
        if rotation == Rotation(y=0.75):
            return self.y().y().y()

        if rotation == Rotation(z=0.25):
            return self.z()
        if rotation == Rotation(z=0.5):
            return self.z().z()
        if rotation == Rotation(z=0.75):
            return self.z().z().z()

        raise ValueError("Can't perform rotations that break axis alignment.")


class FrontMeasurement(object):
    """
    Cube side details extracted from a single image or frame.
    """

    def __init__(self, current_front, is_top_right_darker):
        self.current_front = current_front
        self.is_top_right_darker = is_top_right_darker

    def __eq__(self, other):
        return isinstance(other, FrontMeasurement) \
            and self.current_front == other.current_front \
            and self.is_top_right_darker == other.is_top_right_darker

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """
        >>> str(FrontMeasurement(Back, False))
        'front: PurpleOrange, is_top_right_darker: False'
        """
        return "front: %s, is_top_right_darker: %s" % (self.current_front.visual_name, self.is_top_right_darker)


class PoseMeasurement(object):
    """
    Cube pose details extracted from a single image or frame.
    """

    def __init__(self, front_measurement, angle, center, corners, color_pair):
        self.front_measurement = front_measurement
        self.angle = angle
        self.corners = corners
        self.center = center
        self.color_pair = color_pair

    def __str__(self):
        """
        >>> str(PoseMeasurement(FrontMeasurement(Top, True), 0.124, (251.125, 935.875), None, [[1, 2, 3], [4, 5, 6]]))
        'front: YellowGreen, is_top_right_darker: True, turn: 0.12, color: [[1, 2, 3], [4, 5, 6]], center: (251, 936)'
        """
        return "%s, turn: %.2f, color: %s, center: (%.0f, %.0f)" \
               % (self.front_measurement,
                  self.angle,
                  np.array(self.color_pair, np.int32).tolist(),
                  self.center[0],
                  self.center[1])


class PoseTrack(object):
    """
    Accumulated cube pose details from measurements over time.
    """

    def __init__(self,
                 facing,
                 stable_pose_measurement,
                 last_pose_measurement,
                 last_pose_measurement_stability,
                 rotations):
        self.facing = facing
        self.stable_pose_measurement = stable_pose_measurement
        self.last_pose_measurement = last_pose_measurement
        self.last_pose_measurement_stability = last_pose_measurement_stability
        self.rotations = rotations

    def quantum_operation(self):
        """
        Returns the aggregate quantum operation from the tracked rotations so far.

        >>> (PoseTrack(None, None, None, None, rotations=[]).quantum_operation() == np.mat( \
                [[1, 0], \
                 [0, 1]])).all()
        True
        >>> (PoseTrack(None, None, None, None, rotations=[Rotation(x=0.5)]).quantum_operation() == np.mat( \
                [[0, 1], \
                 [1, 0]])).all()
        True
        >>> (PoseTrack(None, None, None, None, rotations=[Rotation(x=0.5), \
                                                          Rotation(y=0.5), \
                                                          Rotation(z=0.5)]).quantum_operation() == np.mat( \
                [[-1j, 0], \
                 [0, -1j]])).all()
        True
        """
        operations = [r.as_pauli_operation() for r in self.rotations]
        return reduce(lambda e1, e2: e2 * e1, operations, Rotation().as_pauli_operation())

    def then(self, new_pose_measurement):
        """
        Uses the given measurement to continue tracking the cube, and returns an updated PoseTrack instance.
        :param new_pose_measurement The latest measurement of where the cube is and how it is oriented.
        """
        new_pose_measurement_stability = self.last_pose_measurement_stability + 1 \
            if new_pose_measurement.front_measurement == self.last_pose_measurement.front_measurement \
            else 0

        new_facing = self.facing
        new_stable_measurement = self.stable_pose_measurement
        new_rotations = self.rotations

        if new_pose_measurement_stability >= 3:
            new_stable_measurement = new_pose_measurement

            # if the front side changed, use the new side to figure out which quarter X or Y rotation happened
            # TODO: what about quick 180s?
            if new_facing.current_front != new_pose_measurement.front_measurement.current_front:
                adj = [(new_facing.rotated_by(r), r)
                       for r in [Rotation(x=0.25), Rotation(x=-0.25), Rotation(y=0.25), Rotation(y=-0.25)]]
                for facing, rotation in adj:
                    if facing.current_front == new_pose_measurement.front_measurement.current_front:
                        new_facing = facing
                        new_rotations = Rotation.plus_rotation_simplified(new_rotations, rotation)

            # if diagonals are flipped, guess which Z rotation happened based on the last stable measured angle
            if new_facing.is_top_right_darker() != new_pose_measurement.front_measurement.is_top_right_darker:
                advance = self.stable_pose_measurement.angle < 0
                r = Rotation(z=0.25 if advance else -0.25)
                new_rotations = Rotation.plus_rotation_simplified(new_rotations, r)
                new_facing = new_facing.z()
                if not advance:
                    new_facing = new_facing.z().z()

        return PoseTrack(new_facing,
                         new_stable_measurement,
                         new_pose_measurement,
                         new_pose_measurement_stability,
                         new_rotations)

PoseMeasurement.Empty = PoseMeasurement(FrontMeasurement(Front, False),
                                        0,
                                        (0, 0),
                                        [(0, 0), (0, 0), (0, 0), (0, 0)],
                                        [[0, 0, 0], [0, 0, 0]])

PoseTrack.Empty = PoseTrack(Facing(Front, Top),
                            PoseMeasurement.Empty,
                            PoseMeasurement.Empty,
                            0,
                            [])
