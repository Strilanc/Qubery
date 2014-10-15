#!/usr/bin/python

"""
Cube finding program.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
import cv2
import numpy as np
import imag
import cube
from rotation import *

capture = None or cv2.VideoCapture(1)
frame = None

track = cube.PoseTrack.Empty
interop = Rotation().as_pauli_operation()

color_pair_total = np.array([[0, 0, 0], [0, 0, 0]], np.int32)
color_pair_count = 0

while True:
    # Read next frame
    _, frame = capture.read()
    H, W = frame.shape[:2]

    # Shrink and mirror
    reduction = 2
    H, W = H // reduction, W // reduction
    frame = cv2.resize(frame, (W, H))
    frame = cv2.flip(frame, 1)

    drawFrame = np.copy(frame)
    for pose in imag.find_checkerboard_cube_faces(frame, drawFrame):
        cv2.drawContours(drawFrame, [np.int0(pose.corners)], 0, (0, 255, 255), 1)
        track = track.then(pose)
        print pose
        #print track.facing

        color1, color2 = pose.color_pair
        if np.average(color1) < np.average(color2):
            color2, color1 = color1, color2
        #print "        " + str(np.array([color1, color2], np.int32).tolist()) + ",\\"
        color_pair_total += np.array([color1, color2])
        color_pair_count += 1
        #print color_pair_total // color_pair_count
        interop = unitary_lerp(interop, track.quantum_operation(), 0.75)

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
    cv2.imshow('debug', drawFrame)


    if cv2.waitKey(1) == 27 or capture is None:
        break

cv2.destroyAllWindows()
if capture is not None:
    capture.release()
