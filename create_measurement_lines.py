#!/usr/bin/python

import cv2
import sys


current_line = []
line_num = 0
prev_point = None
ignore_next_click = False

def mouse_callback(event, x, y, flags, param):
    global current_line, line_num, prev_point, ignore_next_click

    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if ignore_next_click:
            ignore_next_click = False
            return
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        if prev_point is not None:
            cv2.line(img, prev_point, (x, y), (0, 0, 255), 2)

        current_line.append((x, y))
        prev_point = (x, y)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        line_str = ','.join("%d %d" % (p[0], p[1]) for p in current_line)
        print("%d;line_%d;%s" % (line_num, line_num, line_str))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "line_%d" % (line_num), (x, y - 2), font, 0.6 , (0, 0, 255), 1)
        current_line = []
        prev_point = None
        line_num += 1
        ignore_next_click = True


if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: %s <input video name> [<offset frame number>]" % (sys.argv[0]))
    sys.exit(1)


input_filename = sys.argv[1]
offset = 0
if len(sys.argv) > 2:
    offset = int(sys.argv[2])

cap = cv2.VideoCapture(input_filename)

for i in range(0, offset):
    ret = cap.grab()
    if not ret or not cap.isOpened():
        print("End input stream during of skipping to offset")
        sys.exit(1)


ret, frame = cap.read()
if not ret:
    print("Cannot read video frame")
    sys.exit(1)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback, frame)

while(1):
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xff == 27:
        break


cv2.destroyAllWindows

