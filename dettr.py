#!/usr/bin/python

import argparse

from ctypes import *
import math
import random
import sys
from tr import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.offsetbox as offsetbox
import matplotlib
from skimage import io
import cv2
import pprint

import shapely.geometry as geom

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("tr_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
pp = pprint.PrettyPrinter()

print(sys.argv[0])
lib = CDLL(os.path.join(os.path.dirname(sys.argv[0]), "../libdet.so"), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_tr = lib.do_nms_tr
do_nms_tr.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def np_array_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, np_image, thresh=.5, hier_thresh=.5, nms=.45):
    im = np_array_to_image(np_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x - b.w/2, b.y - b.h/2, b.x + b.w/2, b.y + b.h/2)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def load_measurement_lines(filename, scale=1):
    result = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    for l in lines:
        line = l.split(';')
        m_line = {}
        m_line['num'] = int(line[0])
        m_line['name'] = line[1]
        line_points = map(lambda p: p.split(' '), line[2].split(','))
        m_line['points'] = map(lambda p: (int(int(p[0]) * scale), int(int(p[1]) * scale)), line_points)
        m_line['count'] = 0
        result[m_line['num']] = m_line

    return result

def draw_measurement_lines(img, measurement_lines):
    for i in measurement_lines:
        line = measurement_lines[i]
        color = (255, 0, 0) if line['shoot'] else (0, 255, 0)
        for i in range(1, len(line['points'])):
            cv2.line(img, line['points'][i-1], line['points'][i], color, 2)

        x, y = line['points'][0]
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, "%s: %d" % (line['name'], line['count']), (x, y - 2), font, 1 , color, 1)

def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def is_lines_intersects(line1, line2):
    l1 = geom.LineString(line1)
    l2 = geom.LineString(line2)
    p = l1.intersection(l2)
    return not p.is_empty

def print_counters(m_lines):
    for i in m_lines:
        line = m_lines[i]
        print("%d\t%s\t%d" %(line['num'], line['name'], line['count']))

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description = "Process video and count of objects' traectories intersections")
    ap.add_argument("data_cfg", help="Path to data config file, i.e. cfg/data.data")
    ap.add_argument("net_cfg", help="Path to neural network configuration, i.e. cfg/data.cfg")
    ap.add_argument("weights", help="Neural network weights file")
    ap.add_argument("input_video", help="Input video file")
    ap.add_argument("measurement_lines", help="Measurement lines description file")
    ap.add_argument("--divisor", help="Framerate divisor, integer", type=int, default=1)
    ap.add_argument("--show-tracks", help="Show object tracks (slow)", action="store_true")
    ap.add_argument("--gui", help="Display frames during processing", action="store_true")
    ap.add_argument("--scale-factor", help="Resize frames before detection by specified scale", type=float, default=1)
    ap.add_argument("--debug", help="Enable debugging messages", action="store_true")

    args = ap.parse_args()

    data_cfg = args.data_cfg
    net_cfg = args.net_cfg
    net_weights = args.weights
    input_filename = args.input_video
    measurement_lines_file = args.measurement_lines
    frame_div = args.divisor

    colours = np.random.random_integers(0, 255, (32,3))

    mot_tracker = tr(max_age=10, min_hits=1)

    net = load_net(net_cfg, net_weights, 0)
    meta = load_meta(data_cfg)

    output_template = "output/frame-%05d.jpg"


    cap = cv2.VideoCapture(input_filename)

    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    scale_factor = args.scale_factor

    measurement_lines = load_measurement_lines(measurement_lines_file, scale_factor)
    print measurement_lines


    seq = 1

    print(input_filename)

    last_matched = {}
    track_history = {}
    last_points = {}
    last_counted_at = {}

    red = (0, 0, 255)

    while cap.isOpened():
        ret, orig_frame = cap.read()

        if not ret:
            break

        if seq % frame_div != 0:
            seq += 1
            continue

        out_file = output_template % (seq)

        print('Processing frame %s...' % (seq))

        h, w = orig_frame.shape[:2]

        if scale_factor != 1:
            h = int(scale_factor * h)
            w = int(scale_factor * w)
            frame = cv2.resize(orig_frame, (w, h))
        else:
            frame = orig_frame


        if args.debug:
            print(orig_frame.shape[:2])
            print(frame.shape[:2])

        for key in measurement_lines:
            measurement_lines[key]['shoot'] = False

        r = detect(net, meta, frame)

        dets = []
        for detection in r:
            name, prob, bbox = detection
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], prob])

        np_dets = np.array(dets)
        print("Detected %d objects" % (len(np_dets)))
        trackers, unmatched = mot_tracker.update(np_dets)
        print("Tracked %d objects" % (len(trackers)))
        if args.debug:
            print("%d tracked objects are not detected" % (len(unmatched)))
            print(unmatched)

        det_frame = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        for d in trackers:
            d = d.astype(np.int32)

            color = colours[d[4] % 32]

            if track_history.get(d[4]) is None:
                track_history[d[4]] = []

            th = track_history[d[4]]
            th.append((int(d[0] + (d[2] - d[0])/2), int(d[1] + (d[3] - d[1])/2)))
            if len(th) > 2 and not args.show_tracks:
                th.pop(0)

            color = colours[d[4] % 32]

            shoot = False
            if len(th) >= 2:
                for key in measurement_lines:
                    ml = measurement_lines[key]
                    # don't count one line twice
                    if is_lines_intersects(ml['points'], [th[-2], th[-1]]) and (last_counted_at.get(d[4]) is None or last_counted_at[d[4]] != key):
                        ml['shoot'] = True
                        ml['count'] += 1
                        last_counted_at[d[4]] = key
                        shoot = True

            if shoot:
                color = (0, 0, 255)
            cv2.rectangle(det_frame,(d[0],d[1]), (d[2], d[3]), color, 1)
            cv2.circle(det_frame, (d[0] + (d[2]-d[0])/2, d[1] + (d[3] - d[1]) / 2), 3, (0, 0, 255), cv2.FILLED)
            cv2.putText(det_frame, str(d[4]), (d[0], d[1] - 2), font, 0.5, color, 1)

        if args.show_tracks:
            for t in track_history:
                fp = None
                color = colours[t % 32]
                th = track_history[t]
                for p in th:
                    if fp is None:
                        fp = p
                        continue
                    cv2.line(det_frame, fp, p, color, 1)
                    fp = p

        draw_measurement_lines(det_frame, measurement_lines)


        if args.gui:
            cv2.imshow('det_frame', det_frame)
            if cv2.waitKey(100) == 27:
                print_counters(measurement_lines)
                break

        print("Saving to %s" % (out_file))
        cv2.imwrite(out_file, det_frame)


        seq += 1
        latest_matched = trackers

    print_counters(measurement_lines)


