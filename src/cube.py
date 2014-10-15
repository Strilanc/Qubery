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

    def __str__(self):
        return self.visual_name + " (" + self.side_name + ")"

    def __repr__(self):
        return self.side_name

# home
Front = Side(0, [220, 85, 15], [160, 80, 190], "BlueRed", "Front")
Top = Side(1, [129, 216, 198], [165, 133, 10], "YellowGreen", "Top")
Right = Side(2, [240, 220, 175], [113, 87, 63], "WhiteBlack", "Right")
Back = Side(3, [185,  60,  50], [125, 120, 170], "PurpleOrange", "Back")
Bottom = Side(4, [128, 138, 209], [136, 215, 194], "OrangeYellow", "Bottom")
Left = Side(5, [145, 104, 181], [145, 87, 61], "RedBlack", "Left")


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
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((1, 1, 2), (0, 0, 0))) == 1
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((1, 2, 1), (0, 0, 0))) == 1
    True
    >>> color_pair_distance(((0, 0, 0), (1, 1, 1)), ((2, 1, 1), (0, 0, 0))) == 1
    True
    """
    lux1 = np.average(color_pair_1)
    lux2 = np.average(color_pair_2)
    dux1 = (np.array(color_pair_1) / max(1, lux1)).tolist()
    dux2 = (np.array(color_pair_2) / max(1, lux2)).tolist()
    ds = list([rgb_distance(c1, c2) for c1 in dux1 for c2 in dux2])
    return min(ds[0] + ds[3], ds[1] + ds[2])


def classify_color_pair_as_side(color_pair):
    """
    Matches the given color pair against the expected color pairs, picking the closest match while trying to ignore
    the effects of lighting changes.
    :color_pair: The color pair to classify.
    :return: The most closely matching Side, by color.

    >>> red_blue_samples = [\
        [[101, 78, 192], [222, 83, 14]],\
        [[100, 77, 192], [222, 83, 13]],\
        [[100, 76, 192], [220, 83, 13]],\
        [[100, 77, 192], [221, 83, 13]],\
        [[99, 77, 192], [222, 84, 13]],\
        [[110, 84, 194], [228, 89, 14]],\
        [[117, 89, 197], [233, 94, 17]],\
        [[118, 90, 197], [232, 95, 16]],\
        [[120, 91, 197], [233, 96, 17]],\
        [[123, 93, 198], [234, 98, 18]],\
        [[124, 94, 199], [234, 98, 18]],\
        [[125, 94, 199], [235, 99, 18]],\
        [[129, 98, 199], [239, 102, 19]],\
        [[135, 102, 201], [239, 104, 21]],\
        [[132, 100, 199], [238, 104, 21]],\
        [[135, 102, 201], [239, 105, 22]],\
        [[134, 101, 200], [239, 105, 22]],\
        [[137, 103, 201], [240, 106, 22]],\
        [[140, 105, 201], [241, 108, 23]],\
        [[138, 105, 201], [240, 108, 23]],\
        [[140, 105, 201], [240, 108, 24]],\
        [[139, 104, 201], [242, 108, 25]],\
        [[141, 106, 202], [241, 109, 24]],\
        [[141, 106, 202], [241, 109, 23]],\
        [[144, 108, 203], [242, 111, 25]],\
        [[161, 121, 218], [253, 123, 28]],\
        [[159, 119, 219], [254, 122, 27]],\
        [[192, 137, 221], [255, 136, 25]],\
        [[191, 137, 221], [255, 136, 26]],\
        [[201, 142, 222], [255, 141, 28]],\
        [[209, 148, 224], [255, 146, 31]],\
        [[212, 150, 224], [255, 145, 31]],\
        [[175, 129, 216], [255, 120, 13]],\
        [[158, 118, 212], [255, 109, 7]],\
        [[144, 110, 208], [254, 103, 4]],\
        [[168, 122, 215], [254, 118, 16]],\
        [[168, 122, 215], [254, 118, 16]],\
        [[163, 119, 214], [254, 118, 17]],\
        [[160, 118, 214], [254, 117, 16]],\
        [[158, 116, 213], [254, 116, 15]],\
        [[117, 88, 204], [241, 93, 6]],\
        [[111, 82, 203], [238, 88, 4]],\
        [[111, 83, 203], [238, 88, 4]],\
        [[108, 81, 203], [237, 86, 3]],\
        [[106, 80, 203], [237, 85, 3]],\
        [[121, 89, 206], [246, 90, 3]],\
        [[131, 97, 207], [250, 95, 4]],\
        [[151, 109, 209], [254, 105, 8]],\
        [[198, 139, 217], [255, 133, 24]],\
        [[207, 145, 218], [255, 140, 29]],\
        [[60, 34, 204], [212, 73, 0]],\
        [[62, 34, 204], [213, 74, 0]],\
        [[63, 35, 204], [213, 74, 0]],\
        [[63, 35, 204], [214, 74, 0]],\
        [[62, 35, 203], [214, 75, 0]],\
        [[63, 35, 203], [214, 74, 0]],\
        [[64, 35, 203], [214, 74, 0]],\
        [[64, 35, 203], [215, 75, 0]],\
        [[81, 42, 203], [217, 78, 14]],\
        [[193, 92, 35], [106, 90, 188]],\
        [[196, 138, 212], [254, 135, 35]]]
    >>> yellow_green_samples = [\
        [[146, 190, 154], [170, 127, 16]],\
        [[134, 194, 167], [162, 114, 0]],\
        [[107, 180, 162], [141, 102, 0]],\
        [[110, 180, 163], [140, 102, 0]],\
        [[106, 179, 163], [139, 101, 0]],\
        [[110, 181, 163], [142, 103, 0]],\
        [[111, 182, 163], [143, 103, 0]],\
        [[110, 181, 164], [142, 103, 0]],\
        [[110, 181, 164], [141, 103, 0]],\
        [[113, 183, 165], [145, 104, 0]],\
        [[116, 185, 166], [148, 106, 0]],\
        [[122, 188, 165], [152, 108, 0]],\
        [[124, 188, 165], [152, 108, 0]],\
        [[131, 191, 166], [161, 111, 0]],\
        [[158, 203, 167], [183, 123, 0]],\
        [[159, 204, 167], [185, 125, 0]],\
        [[157, 203, 168], [185, 125, 0]],\
        [[163, 206, 168], [188, 127, 0]],\
        [[167, 211, 169], [194, 130, 0]],\
        [[167, 211, 168], [194, 130, 0]],\
        [[165, 211, 170], [192, 130, 0]],\
        [[167, 211, 167], [193, 130, 0]],\
        [[156, 208, 169], [188, 127, 0]],\
        [[158, 208, 168], [187, 127, 0]],\
        [[161, 210, 169], [191, 129, 0]],\
        [[159, 210, 169], [191, 130, 0]],\
        [[153, 207, 168], [185, 127, 0]],\
        [[141, 202, 166], [177, 123, 0]],\
        [[131, 197, 166], [166, 117, 0]],\
        [[126, 194, 166], [161, 114, 0]],\
        [[119, 190, 164], [155, 110, 0]],\
        [[115, 188, 164], [152, 109, 0]],\
        [[113, 187, 164], [148, 107, 0]],\
        [[110, 185, 163], [146, 106, 0]],\
        [[110, 185, 165], [145, 105, 0]],\
        [[109, 185, 165], [145, 105, 0]],\
        [[113, 185, 164], [147, 106, 0]],\
        [[115, 185, 165], [146, 106, 0]],\
        [[112, 181, 164], [143, 103, 0]],\
        [[112, 181, 164], [143, 103, 0]],\
        [[110, 181, 163], [142, 103, 0]],\
        [[112, 182, 164], [144, 104, 0]],\
        [[113, 182, 163], [144, 104, 0]],\
        [[115, 182, 164], [145, 104, 0]],\
        [[116, 183, 164], [147, 105, 0]],\
        [[119, 185, 165], [150, 107, 0]],\
        [[128, 189, 165], [158, 110, 0]],\
        [[135, 193, 165], [165, 114, 0]],\
        [[138, 195, 166], [167, 116, 0]],\
        [[143, 197, 166], [171, 118, 0]],\
        [[141, 196, 166], [171, 118, 0]],\
        [[130, 191, 165], [162, 113, 0]],\
        [[134, 192, 165], [165, 114, 0]],\
        [[137, 194, 166], [167, 116, 0]],\
        [[77, 170, 184], [117, 104, 0]],\
        [[77, 169, 184], [114, 103, 0]],\
        [[76, 169, 183], [115, 103, 0]],\
        [[75, 169, 184], [115, 103, 0]],\
        [[75, 169, 184], [115, 103, 0]],\
        [[74, 169, 185], [114, 103, 0]],\
        [[140, 194, 166], [169, 116, 0]]]
    >>> white_black_samples = [\
        [[253, 188, 152], [70, 34, 20]],\
        [[254, 196, 157], [82, 40, 24]],\
        [[254, 206, 168], [80, 46, 29]],\
        [[255, 219, 184], [92, 57, 39]],\
        [[255, 222, 184], [95, 60, 41]],\
        [[255, 223, 186], [97, 62, 44]],\
        [[255, 227, 186], [102, 66, 46]],\
        [[255, 230, 187], [106, 69, 47]],\
        [[255, 240, 191], [116, 79, 51]],\
        [[255, 242, 196], [96, 68, 44]],\
        [[255, 243, 195], [95, 68, 44]],\
        [[255, 243, 196], [98, 70, 45]],\
        [[255, 245, 197], [101, 72, 45]],\
        [[255, 245, 197], [100, 70, 45]],\
        [[255, 245, 197], [100, 70, 45]],\
        [[255, 247, 199], [102, 72, 46]],\
        [[255, 250, 202], [111, 78, 49]],\
        [[255, 248, 205], [114, 83, 60]],\
        [[255, 244, 206], [123, 89, 66]],\
        [[255, 247, 206], [136, 99, 73]],\
        [[255, 252, 207], [151, 110, 80]],\
        [[255, 252, 207], [154, 113, 81]],\
        [[255, 250, 202], [136, 100, 69]],\
        [[255, 246, 199], [123, 89, 60]],\
        [[255, 237, 195], [115, 82, 56]],\
        [[255, 233, 193], [107, 76, 52]],\
        [[255, 231, 193], [112, 79, 55]],\
        [[254, 227, 195], [126, 93, 69]],\
        [[255, 227, 188], [119, 86, 59]],\
        [[254, 227, 190], [107, 76, 53]],\
        [[255, 229, 185], [99, 66, 43]],\
        [[255, 232, 187], [93, 62, 41]],\
        [[254, 227, 186], [77, 49, 37]],\
        [[254, 215, 191], [45, 28, 23]],\
        [[253, 208, 194], [14, 12, 15]],\
        [[250, 207, 188], [13, 12, 14]],\
        [[247, 207, 187], [13, 12, 14]],\
        [[247, 207, 187], [13, 12, 14]],\
        [[254, 215, 194], [16, 14, 14]],\
        [[254, 212, 180], [47, 28, 27]]]
    >>> red_black_samples = [\
        [[106, 73, 172], [106, 57, 29]],\
        [[114, 81, 175], [115, 62, 31]],\
        [[116, 81, 176], [117, 63, 32]],\
        [[119, 85, 177], [121, 66, 34]],\
        [[125, 88, 178], [126, 69, 34]],\
        [[131, 92, 180], [132, 72, 35]],\
        [[147, 104, 184], [146, 83, 41]],\
        [[148, 106, 184], [150, 87, 42]],\
        [[139, 100, 183], [141, 79, 38]],\
        [[141, 100, 185], [141, 79, 38]],\
        [[141, 100, 185], [141, 79, 38]],\
        [[133, 96, 185], [132, 73, 35]],\
        [[124, 89, 183], [119, 65, 30]],\
        [[120, 85, 183], [112, 60, 28]],\
        [[112, 80, 182], [103, 52, 25]],\
        [[107, 77, 181], [97, 48, 24]],\
        [[113, 79, 182], [103, 52, 26]],\
        [[124, 87, 184], [116, 62, 30]],\
        [[117, 82, 182], [109, 58, 28]],\
        [[117, 82, 181], [116, 62, 31]],\
        [[117, 83, 181], [119, 63, 31]],\
        [[123, 87, 181], [124, 67, 33]],\
        [[132, 94, 183], [132, 74, 36]],\
        [[142, 101, 184], [142, 79, 39]],\
        [[158, 112, 184], [159, 91, 44]],\
        [[169, 119, 184], [170, 98, 47]],\
        [[156, 111, 180], [156, 91, 39]],\
        [[152, 108, 178], [152, 88, 37]],\
        [[161, 115, 179], [162, 93, 42]],\
        [[167, 119, 181], [166, 97, 43]],\
        [[175, 123, 185], [175, 102, 48]],\
        [[174, 123, 186], [174, 102, 48]],\
        [[164, 116, 186], [164, 95, 45]],\
        [[159, 114, 186], [159, 93, 44]],\
        [[129, 95, 183], [127, 73, 36]],\
        [[110, 81, 181], [107, 58, 29]],\
        [[87, 61, 178], [76, 36, 21]],\
        [[79, 55, 177], [63, 29, 19]],\
        [[74, 51, 176], [58, 25, 18]],\
        [[71, 50, 175], [55, 24, 18]],\
        [[75, 52, 177], [56, 24, 19]],\
        [[74, 51, 177], [56, 24, 18]],\
        [[79, 56, 178], [64, 29, 18]],\
        [[86, 61, 180], [71, 33, 19]],\
        [[84, 61, 179], [71, 33, 18]],\
        [[84, 60, 178], [71, 33, 18]],\
        [[91, 65, 179], [79, 38, 20]],\
        [[101, 71, 181], [90, 44, 24]],\
        [[106, 75, 181], [99, 49, 25]],\
        [[116, 81, 181], [114, 60, 29]],\
        [[122, 87, 181], [122, 66, 31]],\
        [[143, 102, 183], [143, 80, 39]],\
        [[56, 31, 200], [29, 14, 12]],\
        [[53, 30, 198], [26, 12, 10]],\
        [[52, 31, 198], [25, 12, 10]],\
        [[52, 30, 198], [23, 11, 10]],\
        [[51, 30, 197], [25, 12, 10]],\
        [[51, 30, 198], [24, 11, 10]],\
        [[49, 31, 196], [25, 12, 12]],\
        [[66, 34, 199], [39, 20, 24]],\
        [[124, 90, 180], [124, 69, 34]]]
    >>> orange_purple_samples = [\
        [[125, 120, 174], [186, 58, 49]],\
        [[134, 125, 177], [192, 63, 51]],\
        [[136, 126, 177], [196, 67, 54]],\
        [[141, 130, 180], [202, 73, 58]],\
        [[146, 133, 182], [206, 74, 59]],\
        [[151, 137, 184], [210, 77, 60]],\
        [[155, 139, 186], [215, 79, 62]],\
        [[154, 139, 187], [213, 77, 61]],\
        [[158, 146, 196], [222, 82, 66]],\
        [[170, 157, 211], [234, 89, 74]],\
        [[170, 157, 211], [234, 89, 74]],\
        [[163, 159, 222], [235, 79, 70]],\
        [[163, 159, 222], [235, 79, 70]],\
        [[160, 158, 222], [232, 75, 66]],\
        [[159, 159, 222], [232, 73, 65]],\
        [[160, 161, 223], [233, 76, 66]],\
        [[158, 158, 221], [230, 75, 66]],\
        [[141, 140, 201], [209, 68, 60]],\
        [[145, 146, 212], [214, 74, 70]],\
        [[172, 161, 219], [226, 89, 79]],\
        [[172, 161, 219], [226, 89, 79]],\
        [[187, 171, 223], [234, 100, 84]],\
        [[189, 172, 223], [236, 102, 85]],\
        [[193, 175, 225], [239, 104, 86]],\
        [[203, 181, 228], [249, 113, 91]],\
        [[206, 182, 228], [251, 115, 93]],\
        [[216, 186, 229], [254, 117, 95]],\
        [[200, 179, 227], [252, 112, 91]],\
        [[191, 174, 226], [250, 104, 86]],\
        [[64, 91, 210], [136, 14, 31]],\
        [[64, 90, 211], [136, 14, 30]],\
        [[64, 90, 212], [136, 13, 30]],\
        [[63, 90, 212], [136, 13, 29]],\
        [[64, 90, 213], [135, 13, 29]],\
        [[64, 90, 213], [135, 13, 29]],\
        [[64, 90, 213], [135, 12, 29]],\
        [[65, 90, 213], [136, 12, 29]],\
        [[66, 90, 213], [136, 12, 29]],\
        [[64, 90, 214], [136, 12, 29]],\
        [[171, 162, 222], [230, 88, 76]]]
    >>> yellow_orange_samples = [\
        [[102, 167, 149], [89, 99, 165]],\
        [[109, 174, 151], [99, 104, 167]],\
        [[111, 174, 151], [99, 105, 167]],\
        [[110, 173, 152], [99, 105, 168]],\
        [[113, 175, 153], [102, 107, 169]],\
        [[118, 179, 154], [108, 110, 171]],\
        [[117, 178, 154], [106, 109, 170]],\
        [[115, 177, 154], [106, 108, 170]],\
        [[118, 178, 153], [104, 108, 170]],\
        [[111, 175, 153], [99, 105, 169]],\
        [[104, 171, 152], [89, 100, 168]],\
        [[104, 171, 152], [89, 100, 168]],\
        [[102, 170, 153], [87, 99, 168]],\
        [[107, 174, 154], [93, 102, 170]],\
        [[108, 174, 154], [92, 102, 170]],\
        [[111, 176, 154], [98, 104, 170]],\
        [[119, 178, 155], [102, 107, 171]],\
        [[119, 178, 155], [102, 107, 171]],\
        [[122, 181, 155], [110, 110, 172]],\
        [[146, 193, 156], [141, 128, 175]],\
        [[156, 197, 157], [154, 135, 177]],\
        [[149, 196, 155], [150, 134, 177]],\
        [[142, 193, 154], [140, 130, 175]],\
        [[135, 190, 153], [136, 127, 175]],\
        [[125, 186, 152], [126, 122, 173]],\
        [[117, 184, 149], [116, 118, 171]],\
        [[114, 182, 149], [112, 116, 171]],\
        [[99, 171, 147], [92, 105, 166]],\
        [[87, 165, 146], [79, 98, 165]],\
        [[87, 165, 146], [79, 98, 165]],\
        [[93, 169, 147], [85, 102, 166]],\
        [[99, 173, 149], [95, 106, 169]],\
        [[102, 175, 151], [98, 107, 170]],\
        [[115, 178, 152], [110, 112, 171]],\
        [[128, 182, 153], [114, 113, 171]],\
        [[126, 180, 153], [116, 113, 170]],\
        [[112, 172, 151], [101, 106, 167]],\
        [[104, 168, 150], [96, 102, 166]],\
        [[81, 158, 148], [74, 92, 164]],\
        [[81, 158, 148], [74, 92, 164]],\
        [[76, 155, 148], [69, 89, 162]],\
        [[76, 155, 148], [69, 89, 162]],\
        [[89, 160, 149], [80, 94, 164]],\
        [[103, 166, 149], [91, 99, 165]],\
        [[125, 177, 152], [117, 112, 168]],\
        [[130, 180, 152], [121, 116, 169]],\
        [[132, 181, 152], [123, 117, 169]],\
        [[139, 185, 153], [132, 121, 171]],\
        [[87, 178, 194], [53, 78, 211]],\
        [[85, 176, 191], [51, 77, 209]],\
        [[86, 175, 190], [51, 77, 207]],\
        [[85, 175, 190], [51, 76, 207]],\
        [[86, 176, 191], [52, 76, 208]],\
        [[86, 176, 191], [52, 77, 208]],\
        [[87, 177, 191], [53, 78, 208]],\
        [[87, 177, 191], [53, 78, 208]],\
        [[88, 178, 191], [54, 79, 209]],\
        [[138, 185, 153], [132, 121, 170]]]
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
                advance = self.stable_pose_measurement.angle > 0
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
