#!/usr/bin/python
# coding=utf-8

"""
Cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
from rotation import *
import cv2
import cube
import geom
import imag
from gates import QuantumOperation


def scale_around(points, factor, center):
    """
    Returns the same set of points, but proportionally closer or further from a center point.
    Also they get rounded into integers because this method is used for drawing mostly.

    :param points: numpy array of points to scale.
    :param factor: Scaling factor.
    :param center: Center point to scale around.
    """
    return np.array([geom.lerp(center, pt, factor) for pt in points], np.int32)


def draw_tracked_pose_top(tracked_pose, draw_frame):
    """
    Draws the latest measured front face and the stable top face of a tracked pose measurement.

    :param tracked_pose: The track to draw.
    :param draw_frame: The image to draw the information into.
    """
    pose_measurement = tracked_pose.last_pose_measurement
    ordered_corners = np.array(geom.winded(pose_measurement.corners), np.float32)
    mid = pose_measurement.center

    # only show top side when tracking is stable
    if tracked_pose.stable_pose_measurement != tracked_pose.last_pose_measurement:
        return

    bottom_right = ordered_corners[0]
    bottom_left = ordered_corners[1]

    top_left = ordered_corners[2]
    top_right = ordered_corners[3]
    top_mid = (top_left + top_right)/2

    past_left = geom.lerp(bottom_left, top_left, 1.5)
    past_right = geom.lerp(bottom_right, top_right, 1.5)
    past_mid = (past_left + past_right)/2

    whole_contour = scale_around([past_left, past_right, top_right, top_left], 0.5, mid)
    left_contour = scale_around([past_left, past_mid, top_mid, top_left], 0.5, mid)
    right_contour = scale_around([past_right, past_mid, top_mid, top_right], 0.5, mid)

    left_color = tracked_pose.facing.current_top.color(not pose_measurement.front_measurement.is_top_right_darker)
    right_color = tracked_pose.facing.current_top.color(pose_measurement.front_measurement.is_top_right_darker)
    whole_color = (0, 255, 255)

    cv2.drawContours(draw_frame, [left_contour], 0, left_color, -1)
    cv2.drawContours(draw_frame, [right_contour], 0, right_color, -1)
    cv2.drawContours(draw_frame, [whole_contour], 0, whole_color, 1)


def draw_pose(pose_measurement, draw_frame):
    """
    Draws the front face of a pose measurement.

    :param pose_measurement: The pose measurement to draw.
    :param draw_frame: The image to draw the information into.
    """
    ordered_corners = np.array(geom.winded(pose_measurement.corners), np.float32)
    mid = pose_measurement.center

    # fill part of corners with colors from measured side
    for corner_index in range(4):
        side1 = (ordered_corners[corner_index] + ordered_corners[(corner_index + 1) % 4])/2
        side2 = (ordered_corners[corner_index] + ordered_corners[(corner_index - 1) % 4])/2
        corner_contour = scale_around(
            [mid, side1, ordered_corners[corner_index], side2],
            0.5,
            mid)
        clr = pose_measurement.front_measurement.current_front.color(
            is_darker=(corner_index % 2 == 0) != pose_measurement.front_measurement.is_top_right_darker)
        cv2.drawContours(draw_frame, [corner_contour], 0, clr, -1)

    # yellow border
    cv2.drawContours(draw_frame, [scale_around(ordered_corners, 0.5, mid)], 0, (0, 255, 255), 1)


class TrackSquare(object):
    """
    An area corresponding to a qubit. Tracks checkerboard cube within for operations to apply.
    """
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.track = cube.PoseTrack.Empty
        self.is_tracking = False
        self.is_controlled = False
        self.op = Rotation().as_pauli_operation()

    def update(self, pose_measurements):
        """
        Performs tracking using the new set of pose measurements.
        Updates this area's track using the single measurement in the tracking area (or else does nothing).

        :param pose_measurements: Probable checkerboard cube face locations.
        """
        matching = [pose_measurement for pose_measurement in pose_measurements
                    if self.x <= pose_measurement.center[0] <= self.x + self.w
                    if self.y <= pose_measurement.center[1] <= self.y + self.h]
        self.op = unitary_lerp(self.op, self.track.quantum_operation(), 0.5)
        self.is_tracking = len(matching) == 1
        if self.is_tracking:
            matched_pose = matching[0]
            self.track = self.track.then(matched_pose)
            if self.track.stable_pose_measurement == matched_pose:
                self.is_controlled = matched_pose.center[1] >= self.y + self.h / 2

    def draw(self, draw_frame, qubit_size):
        """
        Draws this tracking square as well as the tracked pose and qubit value within it (if any).

        :param draw_frame: The image to draw into.
        :param qubit_size: The radius of circles used to show the qubit's state.
        """
        if self.is_tracking:
            draw_tracked_pose_top(self.track, draw_frame)
        cv2.rectangle(draw_frame, (self.x, self.y), (self.x + qubit_size * 2, self.y + qubit_size * 2), (0, 0, 0), -1)
        for i in range(2):
            c = self.op[i, 0]
            p = (self.x + qubit_size, self.y + qubit_size + qubit_size * 2 * i)
            p2 = (int(round(self.x + qubit_size + c.real * qubit_size)),
                  int(round(self.y + qubit_size + qubit_size * 2 * i + c.imag * qubit_size)))
            cv2.circle(draw_frame, p, qubit_size, (150, 150, 150))
            cv2.circle(draw_frame, p, int(round(np.abs(c * qubit_size))), (0, 255, 255), -1)
            cv2.line(draw_frame, p, p2, (0, 255, 0), 2)
        cv2.rectangle(draw_frame,
                      (self.x, self.y + self.h // 2),
                      (self.x + self.w, self.y + self.h),
                      (0, 0, 0) if not self.is_controlled else (255, 255, 255),
                      1)
        cv2.rectangle(draw_frame,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h // 2),
                      (0, 0, 0) if self.is_controlled else (255, 255, 255),
                      1)


def run_loop():
    """
    Read frame, process frame, repeat.
    """
    margin = 1
    size = 75
    tracks = [TrackSquare(margin + size*i, 50, size - margin*2, 100) for i in range(4)]

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open video capture.")

    operations_in_progress = []

    while True:
        # Read next frame
        _, frame = capture.read()
        h, w = frame.shape[:2]

        # Shrink and mirror
        reduction = 12
        h, w = (h // reduction)*2, (w // reduction)*2
        frame = cv2.resize(frame, (w, h))

        draw_frame = np.copy(frame)
        frame_pose_measurements = imag.find_checkerboard_cube_faces(frame, draw_frame)
        for pose in frame_pose_measurements:
            draw_pose(pose, draw_frame)
        for tracked in tracks:
            tracked.update(frame_pose_measurements)
            tracked.draw(draw_frame, 5)

        operations_to_apply = []
        for i in range(len(tracks)):
            t = tracks[i]
            for r in t.track.rotations:
                op = QuantumOperation(r.as_pauli_operation(), [c.is_controlled for c in tracks], i)
                print op.__str__()
                operations_in_progress.append(op)
            t.track.rotations = []

        cv2.imshow('debug', cv2.resize(draw_frame, (w*3, h*3)))

        if cv2.waitKey(1) == 27 or capture is None:
            break

    cv2.destroyAllWindows()
    capture.release()

run_loop()
