#!/usr/bin/python

"""
Cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0

import cv2
import geom

import imag
from rotation import *
import cube


capture = None or cv2.VideoCapture(1)
frame = None

track = cube.PoseTrack.Empty
interop = Rotation().as_pauli_operation()


def scale_around(points, factor, center):
    return [center + (pt-center)*factor for pt in points]


def draw_tracked_pose(tracked_pose, draw_frame):
    pose_measurement = tracked_pose.last_pose_measurement
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

    # yellow border
    cv2.drawContours(draw_frame, [np.array(scale_around(ordered_corners, 0.5, mid), np.int32)], 0, (0, 255, 255), 1)


while True:
    # Read next frame
    _, frame = capture.read()
    H, W = frame.shape[:2]

    # Shrink and mirror
    reduction = 3
    H, W = H // reduction, W // reduction
    frame = cv2.resize(frame, (W, H))

    drawFrame = np.copy(frame)
    for pose in imag.find_checkerboard_cube_faces(frame, drawFrame):

        track = track.then(pose)
        draw_tracked_pose(track, drawFrame)
        interop = unitary_lerp(interop, track.quantum_operation(), 0.75)

        #print track.rotations
        #print "        " + str(np.array([color1, color2], np.int32).tolist()) + ",\\"
        #va, vb, vc, vd = interop[0,0],interop[0,1],interop[1,0],interop[1,1]
        #print "[|%s\n           %s|\n |%s\n           %s|]" % (Quaternion(round(va.real, 3), round(va.imag, 3)),
        #                               Quaternion(round(vb.real, 3), round(vb.imag, 3)),
        #                               Quaternion(round(vc.real, 3), round(vc.imag, 3)),
        #                               Quaternion(round(vd.real, 3), round(vd.imag, 3)))

    s = 10
    cv2.rectangle(drawFrame, (s, s), (s*3, s*5), (0, 0, 0), -1)
    for i in range(2):
        c = interop[i, 0]
        p = (s*2, s*2 + s*2*i)
        p2 = (int(round(s*2 + c.real*s)), int(round(s*2 + s*2*i + c.imag*s)))
        cv2.circle(drawFrame, p, s, (150, 150, 150))
        cv2.circle(drawFrame, p, int(round(np.abs(c * s))), (0, 255, 255), -1)
        cv2.line(drawFrame, p, p2, (0, 255, 0), 2)
    cv2.imshow('debug', cv2.resize(drawFrame, (W*3, H*3)))


    if cv2.waitKey(1) == 27 or capture is None:
        break

cv2.destroyAllWindows()
if capture is not None:
    capture.release()
