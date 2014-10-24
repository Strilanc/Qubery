#!/usr/bin/python
# coding=utf-8

"""
Image related utility methods used by cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
from geom import *
import cv2
import rotation
import cube


def draw_lines(monitor, line_segments, color=(0, 255, 0), end_point_color=(0, 255, 0)):
    """
    Draws lines onto an image.

    :param monitor: An rgb image to draw lines onto.
    :param line_segments: An iterable of ((x1,y1),(x2,y2)) line segments.
    :param color: The color to draw the line segments with.
    :param end_point_color: The color to draw the end point circles with.
    """
    for (p, q) in line_segments:
        p = int_point(p)
        q = int_point(q)
        cv2.line(monitor, p, q, color)
        cv2.circle(monitor, p, 3, end_point_color)
        cv2.circle(monitor, q, 3, end_point_color)


def rgb_max_to_gray(rgb):
    """
    Converts an rgb input image to a gray-scale image by using the largest color component of each pixel as its
    resulting gray. This brightens areas that are mostly a single primary color compared to the normal translation.

    :param rgb: An opencv image in rgb format.
    """
    b, g, r = cv2.split(rgb)
    return np.maximum(np.maximum(b, g), r)


CIRCLE_SAMPLE_DELTAS_7x7 = [(-3, -1), (-2, -2), (-1, -3), (0, -3),
                            (+1, -3), (+2, -2), (+3, -1), (+3, 0),
                            (+3, +1), (+2, +2), (+1, +3), (0, +3),
                            (-1, +3), (-2, +2), (-3, +1), (-3, 0)]
CIRCLE_SAMPLE_DELTAS_5x5 = [(-2, 0), (-1, -1),
                            (0, -2), (+1, -1),
                            (+2, 0), (+1, +1),
                            (0, +2), (-1, +1)]
CIRCLE_SAMPLE_DELTAS_3x3 = [(-1, 0),
                            (0, -1),
                            (+1, 0),
                            (0, +1)]


def valley_transform(image, circle_deltas=None, sample_radius_factor=2):
    """
    Edge detection transform, favoring long straight boundaries between homogeneous areas.

    :param image: An rgb or gray-scale opencv image.
    :param circle_deltas: The points to sample between. Assumes points half the list length apart are opposites.
    :param sample_radius_factor: How much to expand the sample points, making them sparser but deeper.

    >>> valley_transform(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,1,1,1,1,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,1,1,1,1,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0]]) * 255, \
                                  circle_deltas = CIRCLE_SAMPLE_DELTAS_5x5, \
                                  sample_radius_factor = 1)
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,  63,  63,  63,  63,   0,   0,   0,   0,   0,   0,   0],
           [  0,  63, 127, 191, 191, 127,  63,   0,   0,   0,   0,   0,   0],
           [ 63, 127, 191, 255, 255, 191, 127,  63,   0,   0,   0,   0,   0],
           [ 63, 191, 255, 127, 127, 255, 191,  63,   0,   0,   0,   0,   0],
           [ 63, 191, 191,  63,  63, 191, 191,  63,   0,   0,   0,   0,   0],
           [ 63, 191, 191,  63,  63, 191, 191,  63,   0,   0,   0,   0,   0],
           [ 63, 191, 255, 127, 127, 255, 191, 127,  63,  63,  63,   0,   0],
           [ 63, 127, 191, 255, 255, 191,  63, 191, 191, 191, 127,  63,   0],
           [  0,  63, 127, 191, 191, 191,  63, 191, 255, 255, 191, 127,  63],
           [  0,   0,  63,  63,  63, 127, 191, 255, 127, 127, 255, 191,  63],
           [  0,   0,   0,   0,   0,  63, 191, 191,  63,  63, 191, 191,  63],
           [  0,   0,   0,   0,   0,  63, 191, 191,  63,  63, 191, 191,  63],
           [  0,   0,   0,   0,   0,  63, 191, 255, 127, 127, 255, 191,  63],
           [  0,   0,   0,   0,   0,  63, 127, 191, 255, 255, 191, 127,  63],
           [  0,   0,   0,   0,   0,   0,  63, 127, 191, 191, 127,  63,   0],
           [  0,   0,   0,   0,   0,   0,   0,  63,  63,  63,  63,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    if circle_deltas is None:
        circle_deltas = CIRCLE_SAMPLE_DELTAS_7x7

    circle_deltas = [vector_scale(c, sample_radius_factor)
                     for c in circle_deltas]
    n = len(circle_deltas)
    h = n // 2

    float_image = np.array(image, np.float32)
    rolls = [np.roll(np.roll(float_image, c[0], 0), c[1], 1) for c in circle_deltas]

    t = float_image * 0
    for i in range(h):
        t += np.abs(rolls[i] - rolls[i - h])

    float_scores = t / h
    scores = np.array(np.clip(float_scores, 0, 255), np.uint8)
    return scores


def saddle_transform(image, circle_deltas=None, sample_radius_factor=2):
    """
    Saddle point detection transform, favoring ninety-degree transitions.

    :param image: An rgb or gray-scale opencv image.
    :param circle_deltas: The points to sample between. Points a quarter further should be 90 degrees apart.
    :param sample_radius_factor: How much to expand the sample points, making them sparser but deeper.

    >>> saddle_transform(np.array([[0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,1,1,1,0,0,0,0,0,0], \
                                  [0,0,0,1,1,1,0,0,0,0,0,0], \
                                  [0,0,0,1,1,1,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,1,1,1,0,0,0], \
                                  [0,0,0,0,0,0,1,1,1,0,0,0], \
                                  [0,0,0,0,0,0,1,1,1,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0], \
                                  [0,0,0,0,0,0,0,0,0,0,0,0]]) * 255, \
                                  circle_deltas = CIRCLE_SAMPLE_DELTAS_5x5, \
                                  sample_radius_factor = 1)
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,  63,  63,  63,   0,   0,   0,   0,   0,   0],
           [  0,   0,  63, 127, 127, 127,  63,   0,   0,   0,   0,   0],
           [  0,  63, 127, 127, 127, 127, 127,  63,   0,   0,   0,   0],
           [  0,  63, 127, 127,   0, 127, 127,  63,  63,   0,   0,   0],
           [  0,  63, 127, 127, 127, 191, 191, 127, 127,  63,   0,   0],
           [  0,   0,  63, 127, 127, 191, 191, 127, 127, 127,  63,   0],
           [  0,   0,   0,  63,  63, 127, 127,   0, 127, 127,  63,   0],
           [  0,   0,   0,   0,  63, 127, 127, 127, 127, 127,  63,   0],
           [  0,   0,   0,   0,   0,  63, 127, 127, 127,  63,   0,   0],
           [  0,   0,   0,   0,   0,   0,  63,  63,  63,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    if circle_deltas is None:
        circle_deltas = CIRCLE_SAMPLE_DELTAS_7x7

    # The number of sample points is a tradeoff between radial accuracy and performance
    n = len(circle_deltas)
    q = n // 4

    # Sampling too close causes rounding issues
    # Sampling too far away causes blurring issues
    circle_deltas = [vector_scale(c, sample_radius_factor)
                     for c in circle_deltas]
    quarter_turn_deltas = [vector_dif(circle_deltas[i], circle_deltas[i - q])
                           for i in range(n)]

    float_image = np.array(image, np.int16)

    rolls_x = {dx: np.roll(float_image, dx, 0)
               for dx in set([d[0] for d in quarter_turn_deltas])}
    rolls_xy = {d: np.roll(rolls_x[d[0]], d[1], 1)
                for d in quarter_turn_deltas
                if d[0] >= 0 or d[1] > 0}
    difs_xy = {d: float_image - rolls_xy[d]
               for d in quarter_turn_deltas
               if d[0] >= 0 or d[1] > 0}
    difs_xy = {d: difs_xy[d]
               if d[0] >= 0 or d[1] > 0
               else np.roll(np.roll(-difs_xy[vector_scale(d, -1)], d[0], 0), d[1], 1)
               for d in quarter_turn_deltas}

    # At a saddle point, points 90 degrees off should disagree by roughly +-d for some d
    # Since half the time it's +d and half the time it's -d, there should be a large standard deviation
    float_scores = float_image * 0
    for i in range(n):
        float_scores += np.abs(
            np.roll(np.roll(difs_xy[quarter_turn_deltas[i]], -circle_deltas[i][1], 1), -circle_deltas[i][0], 0))
    scores = np.array(np.clip(float_scores / n, 0, 255), np.uint8)

    return scores


def saddle_score(image, point, radius=10):
    """
    A more refined determination of if a point is a saddle point, less biased towards 90 degree transitions.

    :param image: An rgb opencv image.
    :param point: The point to score.
    :param radius: How far to look for saddle-y-ness.
    """
    h, w, _ = image.shape
    x, y = point

    # Discard points near the edge because they can't be checked appropriately
    # (Also the faster saddle transform rolls the image and creates artifacts near the edge)
    if x < radius or x >= w - radius or y < radius or y >= h - radius:
        return 0, 0, 0, (0, 0, 0), (0, 0, 0)

    pixels_at_radius = int(math.ceil(math.pi * radius / 4) * 4)
    samples = []
    for i in range(pixels_at_radius):
        theta = i * 2 * math.pi / pixels_at_radius
        dx, dy = int(round(radius * math.cos(theta))), int(round(radius * math.sin(theta)))
        samples.append(image[y + dy][x + dx])
    samples = np.array(samples, np.float32)

    spread, best_labels, centers = cv2.kmeans(
        samples,
        K=2,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.1),
        attempts=1,
        flags=cv2.KMEANS_RANDOM_CENTERS)
    best_labels = np.ndarray.flatten(best_labels)
    spread = max(1, math.log(max(spread, 1)))

    half_turn = pixels_at_radius // 2
    half_labels = (best_labels + np.roll(best_labels, half_turn))[:half_turn]
    candidates = []
    for offset in range(half_turn):
        roll_tally = partial_sums(np.roll(half_labels, -offset))
        d = roll_tally[half_turn]
        for length in range(0, half_turn):
            miss = d - 2 * roll_tally[length] + 2 * length
            candidates.append((offset, length, miss))
    offset, length, miss = min(candidates, key=lambda e: e[2])
    score = pixels_at_radius / (1 + miss)

    return (offset * 2 * math.pi / pixels_at_radius,
            (offset + length) * 2 * math.pi / pixels_at_radius,
            score / spread,
            (int(centers[0][0]), int(centers[0][1]), int(centers[0][2])),
            (int(centers[1][0]), int(centers[1][1]), int(centers[1][2])))


def spread_local_maxima(image, spread_log_base_3=(3, 3), do_padding=True):
    """
    Rolls an image around while maxing it against itself, to spread local maximas' values to their nearby area.

    :param image: A grey-scale opencv image.
    :param spread_log_base_3: The maximas are spread around by three to the power of this value in row and col.
    :param do_padding: Whether or not to let the maximums wrap around, so the left column is next to the right column.

    >>> (spread_local_maxima(np.array([[0, 0, 0, 0, 0], \
                                       [0, 1, 0, 0, 0], \
                                       [0, 0, 0, 0, 0], \
                                       [0, 0, 0, 2, 0], \
                                       [0, 0, 0, 3, 0], \
                                       [0, 0, 0, 4, 0]]), spread_log_base_3=(1, 1)) \
            == np.array([[1, 1, 1, 0, 0], \
                         [1, 1, 1, 0, 0], \
                         [1, 1, 2, 2, 2], \
                         [0, 0, 3, 3, 3], \
                         [0, 0, 4, 4, 4], \
                         [0, 0, 4, 4, 4]])).all()
    True
    >>> (spread_local_maxima(np.array([[0, 0, 0, 0, 0], \
                                       [0, 1, 0, 0, 0], \
                                       [0, 0, 0, 0, 0], \
                                       [0, 0, 0, 2, 0], \
                                       [0, 0, 0, 3, 0], \
                                       [0, 0, 0, 4, 0]]), spread_log_base_3=(2, 1)) \
            == np.array([[1, 1, 1, 1, 1], \
                         [1, 1, 1, 1, 1], \
                         [2, 2, 2, 2, 2], \
                         [3, 3, 3, 3, 3], \
                         [4, 4, 4, 4, 4], \
                         [4, 4, 4, 4, 4]])).all()
    True
    >>> (spread_local_maxima(np.array([[0, 0, 0, 0, 0], \
                                       [0, 1, 0, 0, 0], \
                                       [0, 0, 0, 0, 0], \
                                       [0, 0, 0, 2, 0], \
                                       [0, 0, 0, 3, 0], \
                                       [0, 0, 0, 4, 0]]), spread_log_base_3=(2, 2)) \
            == np.array([[3, 3, 3, 3, 3], \
                         [4, 4, 4, 4, 4], \
                         [4, 4, 4, 4, 4], \
                         [4, 4, 4, 4, 4], \
                         [4, 4, 4, 4, 4], \
                         [4, 4, 4, 4, 4]])).all()
    True
    """
    h = image.shape[0]
    w = image.shape[1]
    row_padding = int((math.pow(3, spread_log_base_3[0]) - 1) / 2)
    col_padding = int((math.pow(3, spread_log_base_3[1]) - 1) / 2)
    if row_padding >= w:
        row_padding = 0
    if col_padding >= h:
        col_padding = 0
    if not do_padding:
        row_padding = 0
        col_padding = 0
    total = np.pad(image, ((row_padding, row_padding), (col_padding, col_padding)), 'minimum')
    for axis in range(2):
        d = 1
        for i in range(spread_log_base_3[1-axis]):
            total = np.maximum(total, np.roll(total, d, axis))
            total = np.maximum(total, np.roll(total, -d, axis))
            d *= 3

    if not do_padding:
        return total
    return total[row_padding:-row_padding, col_padding:-col_padding]


def find_isolated_local_maxima(grey_scale_image, spread_log_base_3=(3, 3), do_padding=True):
    """
    Finds local maxima that aren't too close to a higher local maxima.

    :param grey_scale_image: A grey-scale opencv image.
    :param spread_log_base_3: The maximas are spread around by three to the power of this value, occluding other ones.
    :param do_padding: Whether or not to let the maximums wrap around, so the left column is next to the right column.

    >>> find_isolated_local_maxima(np.array([[0, 1, 2, 3, 4], \
                                             [5, 1, 2, 3, 5], \
                                             [6, 2, 3, 4, 6], \
                                             [7, 8, 9, 8, 7], \
                                             [6, 7, 8, 7, 8], \
                                             [5, 6, 7, 8, 10]]), spread_log_base_3=(1, 1))
    [(2, 3), (4, 5)]
    >>> find_isolated_local_maxima(np.array([[0, 1, 2, 3, 4], \
                                             [5, 1, 2, 3, 5], \
                                             [6, 2, 3, 4, 6], \
                                             [7, 8, 9, 8, 7], \
                                             [6, 7, 8, 7, 8], \
                                             [5, 6, 7, 8, 10]]), spread_log_base_3=(2, 2))
    [(4, 5)]
    """
    total = spread_local_maxima(grey_scale_image, spread_log_base_3, do_padding)
    c, r = (total == grey_scale_image).nonzero()
    return zip(r, c)


def find_corner_votes(center, a1, a2, valley_trans_fine):
    """
    Uses the given axis angles to find probable face cells to flood fill in an attempt to find corners.

    :param center: The (x,y) saddle point.
    :param a1: Estimated angle of the axis of transition.
    :param a2: Estimated angle of the other axis of transition.
    :param valley_trans_fine: Precomputed valley transform of input frame.
    """
    h, w = valley_trans_fine.shape[:2]
    d1 = (math.cos(a1), math.sin(a1))
    d2 = (math.cos(a2), math.sin(a2))
    cards = [
        (+1, 0),
        (+1, +1),
        (0, +1),
        (-1, +1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (+1, -1)]
    dirs = [vector_sum_weighted(d1, w1, d2, w2) for (w1, w2) in cards]

    corner_votes = [[], [], [], [], [], [], [], []]
    for i in range(4):
        i_prev_diag = i * 2 - 1
        i_prev = i * 2 + 0
        i_diag = i * 2 - 7
        i_next = i * 2 - 6
        i_next_diag = i * 2 - 5

        diag = dirs[i_diag]

        x, y = offset_point_by_vector_weighted(center, diag, 10)
        x = int(min(max(x, 0), w - 1))
        y = int(min(max(y, 0), h - 1))

        flood_terrain = rgb_max_to_gray(valley_trans_fine)
        flood_terrain[y][x] = 0
        flood_dst = np.zeros([h + 2, w + 2], dtype=np.uint8)
        cv2.floodFill(flood_terrain, flood_dst, (x, y), 1, 10, 10, cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
        contours, _ = cv2.findContours(flood_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        contour = contours[0]
        flood_area = cv2.contourArea(contour)
        if flood_area < 5:
            continue
        fit_rect = cv2.cv.BoxPoints(cv2.minAreaRect(contour))
        fit_rect_area = vector_length(vector_dif(fit_rect[0], fit_rect[1])) * vector_length(
            vector_dif(fit_rect[1], fit_rect[2]))
        rect_waste = fit_rect_area / (flood_area + 1)
        if rect_waste > 1.66:
            continue

        for (j, k) in [(i_prev_diag, i_prev), (i_diag, i_diag), (i_next_diag, i_next)]:
            best_pt_1 = max(contour[:, 0], key=lambda e: dot(e, dirs[j]))
            best_pt_2 = max(fit_rect, key=lambda e: dot(e, dirs[j]))
            best_pt = vector_lerp(best_pt_1, best_pt_2, 0.5)
            corner_votes[k].append(best_pt)
    return corner_votes


def vote_and_infer_corners(corner_votes, center):
    """
    Aggregates the various corner votes into a guess at where the entire face of the cube is.

    :param corner_votes: A list of lists of votes for each corner, with evens being sides and odds being diagonals.
    :param center: The (x,y) saddle point.
    :return: The corners of the cube's face.
    """
    corners = list([None if len(e) == 0 else vector_average(e) for e in corner_votes])
    diag_corners = corners[1::2]
    if len([e for e in diag_corners if e is not None]) < 2:
        return None

    more_corner_votes = list([list(e) for e in corner_votes])
    for i in range(4):
        i_prev_diag = i * 2 - 1
        i_prev = i * 2 + 0
        i_diag = i * 2 - 7
        i_next = i * 2 - 6
        i_next_diag = i * 2 - 5
        i_opp_diag = i * 2 - 3
        if corners[i_diag] is not None:
            continue
        if corners[i_next] is not None and corners[i_next_diag] is not None:
            b = corners[i_next]
            d = vector_dif(b, corners[i_next_diag])
            more_corner_votes[i_diag].append(vector_sum(b, d))
        if corners[i_prev] is not None and corners[i_prev_diag] is not None:
            b = corners[i_prev]
            d = vector_dif(b, corners[i_prev_diag])
            more_corner_votes[i_diag].append(vector_sum(b, d))
        if corners[i_opp_diag] is not None:
            b = center
            d = vector_dif(b, corners[i_opp_diag])
            more_corner_votes[i_diag].append(vector_sum(b, d))
    corners = list([None if len(e) == 0 else vector_average(e) for e in more_corner_votes])
    diag_corners = corners[1::2]
    if any([e is None for e in diag_corners]):
        return None
    return diag_corners


def integrate_rows(frame):
    """
    Computes the row-wise integral of an image. The row-wise integral changes each entry to the partial sum of all
    elements in the row up to the given entry. For example, [[1,2,3],[2,3,4]] becomes [[1,3,6],[2,5,9]].

    :param frame: The image to integrate row-wise.

    >>> (integrate_rows(np.array([[1, 2, 3], \
                                  [2, 3, 4]], dtype=np.float32)) \
           == np.array([[1, 3, 6], \
                        [2, 5, 9]])).all()
    True
    >>> (integrate_rows(np.array([[1, 2, 3, 4], \
                                  [5, 6, 7, 8], \
                                  [9, 10, 11, 12]], dtype=np.float32)) \
           == np.array([[1, 3, 6, 10], \
                        [5, 11, 18, 26], \
                        [9, 19, 30, 42]])).all()
    True
    """
    integral = cv2.integral(frame)
    reduced = integral - np.roll(integral, +1, axis=0)
    return reduced[1:, 1:]


def log_polar_transform(frame, center, distance_factor):
    """
    Transforms the image so that angles emanating from a point become rows in the result, and distances are on a log
    scale.

    :param frame: The image to transform.
    :param center: The center from which the angles emanate.
    :param distance_factor: What to multiply distances by after log-ing them. Larger values are "more linear".
    """
    dst = cv2.cv.fromarray(np.zeros_like(frame))
    cv2.cv.LogPolar(cv2.cv.fromarray(frame), dst, center, distance_factor)
    return np.array(dst)


def measure_cross_at(frame, color, center):
    """
    Estimates the end points of a cross of the given color centered on the given point.

    :param frame: The image containing the cross.
    :param color: The color of the cross.
    :param center: The center of the cross.
    :return ((p1, p2), (q1, q2)), score
    """

    # Turn angles into rows of color error, and combine opposite angles
    h, w, _ = frame.shape
    frame_float = np.array(frame, np.float32)
    solid_color = (frame_float * 0 + 1) * color
    color_difference = np.abs(frame_float - solid_color) - 128
    ray_space = log_polar_transform(color_difference, center, 80) + 128
    line_space = ray_space[0:h // 2] + ray_space[h // 2:h]

    # Prefer longer crosses with less color error, with a linear tradeoff in log-space
    # If the cross is too skewed by perspective, giving opposite legs different lengths, this won't work well
    line_noise_accumulation_space = integrate_rows(np.float32(rgb_max_to_gray(line_space)))
    col_indexes = np.indices(line_noise_accumulation_space.shape)[1]
    line_signal_space = col_indexes * 35 - line_noise_accumulation_space

    # There should be two lines standing out in the signal space
    line_scores = []
    for p in find_isolated_local_maxima(line_signal_space, spread_log_base_3=(5, 3), do_padding=False):
        # Recover cartesian offsets
        e, r = p
        d = math.exp(e/80)
        dx, dy = d * rotation.cos_tau(r/h), d * rotation.sin_tau(r/h)
        x1, y1 = center[0] + dx, center[1] + dy
        x2, y2 = center[0] - dx, center[1] - dy

        # If leg is too small or too large, ignore it as a possible error
        if d < 5 or d > 100:
            continue
        # If leg runs off the side of the image, ignore it as a possible error
        if x1 < 0 or y1 < 0 or x1 > w - 1 or y1 > h - 1:
            continue
        if x2 < 0 or y2 < 0 or x2 > w - 1 or y2 > h - 1:
            continue
        # If leg signal is not strong enough, ignore it as a possible error
        signal = line_signal_space[p[1], p[0]]
        if signal < 2000:
            continue

        line = (x1, y1), (x2, y2)
        line_scores.append((line, signal))
    if len(line_scores) != 2:
        return None
    lines = list([e[0] for e in line_scores])
    score = min([e[1] for e in line_scores])
    return lines, score


def measure_cross_near(frame, color, center):
    """
    Estimates the end points of a cross of the given color centered very near to the given point.

    :param frame: The image containing the cross.
    :param color: The color of the cross.
    :param center: The center of the cross.
    :return ((p1, p2), (q1, q2)), score
    """
    winners = [measure_cross_at(frame, color, vector_sum(center, d))
               for d in [(0, 0)]]  # "Exact" counts as "near", right? Seems to be accurate enough, for now...
    keepers = [k for k in winners if k is not None]
    if len(keepers) == 0:
        return None
    return max(keepers, key=lambda e: e[1])


def measure_checkerboard_color_inside(frame, corners):
    """
    Uses the given corners to perspective-correct the image and average colors across where the checkerboard faces are
    expected to be.

    :param frame: The image containing the checkerboard cube to measure.
    :param corners: Positions of the corners of the face of a checkerboard cube.
    """
    d = 50
    r = d // 2
    u, v = 0, d
    standard_corners = np.float32([[v, v], [u, v], [u, u], [v, u]])

    corners = np.float32(corners)
    corners = np.array(winded(corners))

    perspective = cv2.getPerspectiveTransform(corners, standard_corners)
    normalized = cv2.warpPerspective(frame, perspective, (d, d))
    x = np.array(normalized, np.float32)
    t = 5
    v1 = x[t:r-t, t:r-t] + x[r+t:r+r-t, r+t:r+r-t]
    v2 = x[t:r-t, r+t:r+r-t] + x[r+t:r+r-t, t:r-t]
    color1 = np.average(np.average(v1, axis=0), axis=0) / 2
    color2 = np.average(np.average(v2, axis=0), axis=0) / 2
    return [color1, color2]


def distance_from_cross_points_to_frame(cross_end_points, frame_corners):
    """
    Determines how closely four cross end points fall on the four sides of the border of a cube side.

    :param cross_end_points: The four end points of a cross.
    :param frame_corners: The four corners of a checkerboard cube side.
    """
    # TODO: match each point to one line, so a tiny cross near one edge doesn't score highly
    return max([distance_from_point_to_cycle_path(e, frame_corners)
                for e in cross_end_points])


def find_checkerboard_cube_faces(input_frame, draw_frame):
    """
    Tries to find faces of checkerboard cubes.

    :param input_frame: A raw rgb image of reasonable size.
    :param draw_frame: A copy of the input image to draw debug information on.
    :return: A list of cube.PoseMeasurement instances; one for each found face.
    """
    valley_trans = valley_transform(input_frame)
    valley_trans_fine = valley_transform(input_frame, circle_deltas=CIRCLE_SAMPLE_DELTAS_5x5, sample_radius_factor=1)
    saddle_trans = saddle_transform(input_frame)
    saddle_trans_fine = saddle_transform(input_frame, circle_deltas=CIRCLE_SAMPLE_DELTAS_5x5, sample_radius_factor=1)
    combined = np.maximum(saddle_trans, valley_trans) - valley_trans

    # find centers
    gray_saddle_trans = rgb_max_to_gray(combined)
    local_maximas = find_isolated_local_maxima(gray_saddle_trans)
    candidates = []
    for center in local_maximas:
        if gray_saddle_trans[center[1]][center[0]] < 30:
            continue
        a1, a2, s, c1, c2 = saddle_score(input_frame, center)
        if s < 1:
            continue

        corner_votes = find_corner_votes(center, a1, a2, valley_trans_fine)
        diag_corners = vote_and_infer_corners(corner_votes, center)
        if diag_corners is None:
            continue

        fit_rect = cv2.cv.BoxPoints(cv2.minAreaRect(np.array(diag_corners, dtype=np.float32)))
        fit_rect_area = vector_length(vector_dif(fit_rect[0], fit_rect[1])) * vector_length(
            vector_dif(fit_rect[1], fit_rect[2]))
        infer_area = cv2.contourArea(np.array(diag_corners, dtype=np.float32))
        usage2 = infer_area / (fit_rect_area + 0.1)
        if usage2 < 0.6:
            continue

        cross_lines_score = measure_cross_near(saddle_trans_fine, saddle_trans[center[1]][center[0]] // 2, center)
        if cross_lines_score is None:
            continue
        ((p1, p2), (q1, q2)), score = cross_lines_score
        if distance_from_cross_points_to_frame([p1, p2, q1, q2], diag_corners) > 50:
            continue

        color_pair = measure_checkerboard_color_inside(input_frame, diag_corners)
        side = cube.classify_color_pair_as_side(color_pair)
        is_top_right_darker = np.max(color_pair[0]) > np.max(color_pair[1])

        mid = np.average(diag_corners, axis=0)
        right_topward_corner = winded(diag_corners)[0]
        turns = vector_angle(vector_dif(right_topward_corner, mid))/math.pi/2 - 0.125
        pose = cube.PoseMeasurement(cube.FrontMeasurement(side, is_top_right_darker),
                                    turns,
                                    mid,
                                    diag_corners,
                                    color_pair)

        candidates.append(pose)

    return candidates


def distance_from_point_to_cycle_path(point, path_points):
    """
    Returns the minimum distance between a point and a cyclical path.

    :param path_points: The path, specified as a sequence of (x,y) points defining line segments in order.
    :param point: The (x,y) point.

    >>> abs(distance_from_point_to_cycle_path((0, 0), [(1, 1), (-1, 1), (0, -1)]) - 1/math.sqrt(5)) < 0.000001
    True
    >>> abs(distance_from_point_to_cycle_path((1, 1), [(1, 1), (-1, 1), (0, -1)])) < 0.000001
    True
    >>> abs(distance_from_point_to_cycle_path((1, 10), [(1, 1), (-1, 1), (0, -1)]) - 9) < 0.000001
    True
    >>> abs(distance_from_point_to_cycle_path((0, 10), [(1, 1), (-1, 1), (0, -1)]) - 9) < 0.000001
    True
    >>> abs(distance_from_point_to_cycle_path((-1, 10), [(1, 1), (-1, 1), (0, -1)]) - 9) < 0.000001
    True
    >>> abs(distance_from_point_to_cycle_path((-2, 10), [(1, 1), (-1, 1), (0, -1)]) - math.sqrt(82)) < 0.000001
    True
    """
    return min([point_distance_from_line_segment(point, line) for line in cycle_windows(path_points, 2)])


def scale_shape_points(points, scale_factor):
    """
    Scales a set of points, keeping their center of mass fixed.

    :param points: A list of (x,y) points.
    :param scale_factor: A numeric factor to scale the distance-from-center vectors by.
    """
    center = vector_average(points)
    return list([vector_lerp(center, e, scale_factor) for e in points])
