#!/usr/bin/python

"""
Cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0

import cv2
import itertools
import geom

import imag
from rotation import *
import cube


capture = None or cv2.VideoCapture(1)
if not capture.isOpened():
    raise RuntimeError("Failed to open video capture.")
frame = None

def scale_around(points, factor, center):
    return [center + (pt-center)*factor for pt in points]


def draw_tracked_pose_top(tracked_pose, draw_frame):
    pose_measurement = tracked_pose.last_pose_measurement
    ordered_corners = np.array(geom.winded(pose_measurement.corners), np.float32)
    mid = pose_measurement.center

    # show top, based on tracking
    if tracked_pose.stable_pose_measurement == tracked_pose.last_pose_measurement:
        top_left = ordered_corners[2]
        top_right = ordered_corners[3]
        bottom_left = ordered_corners[1]
        bottom_right = ordered_corners[0]
        past_left = bottom_left + (top_left - bottom_left)*1.5
        past_right = bottom_right + (top_right - bottom_right)*1.5
        past_mid = (past_left + past_right)/2
        top_mid = (top_left + top_right)/2
        upside = [ordered_corners[2], past_left, past_right, ordered_corners[3]]
        left_contour = np.array(scale_around([past_left, past_mid, top_mid, top_left], 0.5, mid), np.int32)
        right_contour = np.array(scale_around([past_right, past_mid, top_mid, top_right], 0.5, mid), np.int32)
        left_color = tracked_pose.facing.current_top.color(not pose_measurement.front_measurement.is_top_right_darker)
        right_color = tracked_pose.facing.current_top.color(pose_measurement.front_measurement.is_top_right_darker)
        cv2.drawContours(draw_frame, [left_contour], 0, left_color, -1)
        cv2.drawContours(draw_frame, [right_contour], 0, right_color, -1)
        cv2.drawContours(draw_frame, [np.array(scale_around(upside, 0.5, mid), np.int32)], 0, (0, 255, 255), 1)


def draw_pose(pose_measurement, draw_frame):
    ordered_corners = np.array(geom.winded(pose_measurement.corners), np.float32)
    mid = pose_measurement.center

    # fill part of corners with colors from measured side
    for corner_index in range(4):
        side1 = (ordered_corners[corner_index] + ordered_corners[(corner_index + 1) % 4])/2
        side2 = (ordered_corners[corner_index] + ordered_corners[(corner_index - 1) % 4])/2
        corner_contour = np.array(scale_around(
            [mid, side1, ordered_corners[corner_index], side2],
            0.5,
            mid), np.int32)
        clr = pose_measurement.front_measurement.current_front.color(
            is_darker=(corner_index % 2 == 0) != pose_measurement.front_measurement.is_top_right_darker)
        cv2.drawContours(draw_frame, [corner_contour], 0, clr, -1)

    # yellow border
    cv2.drawContours(draw_frame, [np.array(scale_around(ordered_corners, 0.5, mid), np.int32)], 0, (0, 255, 255), 1)


class TrackSquare:
    def __init__(self, x, y, w, h, control_x, control_y):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.control_x = control_x
        self.control_y = control_y
        self.track = cube.PoseTrack.Empty
        self.is_tracking = False
        self.is_controlled = False
        self.op = Rotation().as_pauli_operation()

    def update(self, pose_measurements):
        matching = [e for e in pose_measurements
                    if self.x <= e.center[0] <= self.x + self.w
                    if self.y <= e.center[1] <= self.y + self.h]
        self.op = unitary_lerp(self.op, self.track.quantum_operation(), 0.5)
        self.is_tracking = len(matching) > 0
        if len(matching) > 0:
            pose = matching[0]
            self.track = self.track.then(pose)
            if self.track.stable_pose_measurement == pose:
                self.is_controlled = False
        else:
            controls = [e for e in pose_measurements
                        if self.control_x <= e.center[0] <= self.control_x + self.w
                        if self.control_y <= e.center[1] <= self.control_y + self.h]
            if len(controls) > 0:
                control_pose = controls[0]
                self.track = self.track.then(control_pose)
                self.is_tracking = True
                if self.track.stable_pose_measurement == control_pose:
                    self.is_controlled = True

    def draw(self, draw_frame, s):
        if self.is_tracking:
            draw_tracked_pose_top(self.track, draw_frame)
        cv2.rectangle(drawFrame, (self.x, self.y), (self.x + s*2, self.y + s*2), (0, 0, 0), -1)
        for i in range(2):
            c = self.op[i, 0]
            p = (self.x + s, self.y + s + s*2*i)
            p2 = (int(round(self.x + s + c.real*s)), int(round(self.y + s + s*2*i + c.imag*s)))
            cv2.circle(draw_frame, p, s, (150, 150, 150))
            cv2.circle(draw_frame, p, int(round(np.abs(c * s))), (0, 255, 255), -1)
            cv2.line(draw_frame, p, p2, (0, 255, 0), 2)
        #print track.rotations
        #print "        " + str(np.array([color1, color2], np.int32).tolist()) + ",\\"
        #va, vb, vc, vd = interop[0,0],interop[0,1],interop[1,0],interop[1,1]
        #print "[|%s\n           %s|\n |%s\n           %s|]" % (Quaternion(round(va.real, 3), round(va.imag, 3)),
        #                               Quaternion(round(vb.real, 3), round(vb.imag, 3)),
        #                               Quaternion(round(vc.real, 3), round(vc.imag, 3)),
        #                               Quaternion(round(vd.real, 3), round(vd.imag, 3)))
        cv2.rectangle(drawFrame,
                      (self.control_x, self.control_y),
                      (self.control_x + self.w, self.control_y + self.h),
                      (0, 0, 0) if not self.is_controlled else (255, 255, 255),
                      1)
        cv2.rectangle(drawFrame,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h),
                      (0, 0, 0) if self.is_controlled else (255, 255, 255),
                      1)


margin = 1
size = 50
tracks = [TrackSquare(margin + size*i, 50, size - margin*2, size, margin + size*i, 100) for i in range(4)]

while True:
    # Read next frame
    _, frame = capture.read()
    H, W = frame.shape[:2]

    # Shrink and mirror
    reduction = 3
    H, W = H // reduction, W // reduction
    frame = cv2.resize(frame, (W, H))

    drawFrame = np.copy(frame)
    pose_measurements = imag.find_checkerboard_cube_faces(frame, drawFrame)
    for pose in pose_measurements:
        draw_pose(pose, drawFrame)
    for tracked in tracks:
        tracked.update(pose_measurements)
        tracked.draw(drawFrame, 5)

    opyops = []
    for t in tracks:
        for r in t.track.rotations:
            q = r.as_pauli_operation()
            m = np.identity(1, np.float32)
            for t2 in tracks.__reversed__():
                if t == t2:
                    m = geom.tensor_product(m, r.as_pauli_operation())
                elif t2.is_controlled:
                    print "controlled"
                    m = geom.controlled_by_next_qbit(m)
                else:
                    m = geom.tensor_product(m, np.identity(2, np.float32))
            opyops.append(m)
        t.track.rotations = []
    if len(opyops) > 0:
        print reduce(lambda e1, e2: e2 * e1, opyops).__repr__()

    cv2.imshow('debug', cv2.resize(drawFrame, (W*3, H*3)))

    if cv2.waitKey(1) == 27 or capture is None:
        break

cv2.destroyAllWindows()
if capture is not None:
    capture.release()
