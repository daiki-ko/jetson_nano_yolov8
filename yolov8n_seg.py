#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019-2021 Tsutomu Furuse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
from ultralytics import YOLO
import wget
import tarfile
import time
import argparse

FPS = 30
GST_STR_CSI = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1, sensor-id=%d \
    ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
WINDOW_NAME = 'YOLO v8 segmentation'
INPUT_RES = (416, 416)


# Draw bounding boxes on the screen from the YOLO inference result
def draw_bboxes(image, bboxes, confidences, categories, all_categories, message=None):
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        img_height, img_width, _ = image.shape
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(img_width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(img_height, np.floor(y_coord + height + 0.5).astype(int))
        cv2.rectangle(image, \
            (left, top), (right, bottom), (0, 0, 255), 3)
        info = '{0} {1:.2f}'.format(all_categories[category], score)
        cv2.putText(image, info, (right, top), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        print(info)
    if message is not None:
        cv2.putText(image, message, (32, 32), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

# Draw the message on the screen
def draw_message(image, message):
    cv2.putText(image, message, (32, 32), \
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

# Reshape the image from OpneCV to Tiny YOLO v2
def reshape_image(img):
    # Convert 8-bit integer to 32-bit floating point
    img = img.astype(np.float32)
    # Convert HWC to CHW
    img = np.transpose(img, [2, 0, 1])
    # Convert CHW to NCHW
    img = np.expand_dims(img, axis=0)
    # Convert to row-major
    img = np.array(img, dtype=np.float32, order='C')
    return img

# Main function
def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='YOLO v8 segmentation')
    parser.add_argument('--camera', '-c', \
        type=int, default=0, metavar='CAMERA_NUM', \
        help='Camera number')
    parser.add_argument('--csi', \
        action='store_true', \
        help='Use CSI camera')
    parser.add_argument('--width', \
        type=int, default=1280, metavar='WIDTH', \
        help='Capture width')
    parser.add_argument('--height', \
        type=int, default=720, metavar='HEIGHT', \
        help='Capture height')
    parser.add_argument('--objth', \
        type=float, default=0.6, metavar='OBJ_THRESH', \
        help='Threshold of object confidence score (between 0 and 1)')
    parser.add_argument('--nmsth', \
        type=float, default=0.3, metavar='NMS_THRESH', \
        help='Threshold of NMS algorithm (between 0 and 1)')
    args = parser.parse_args()

    if args.csi or (args.camera < 0):
        if args.camera < 0:
            args.camera = 0
        # Open the MIPI-CSI camera
        gst_cmd = GST_STR_CSI \
            % (args.width, args.height, FPS, args.camera, args.width, args.height)
        cap = cv2.VideoCapture(gst_cmd, cv2.CAP_GSTREAMER)
    else:
        # Open the V4L2 camera
        cap = cv2.VideoCapture(args.camera)
        # Set the capture parameters
        #cap.set(cv2.CAP_PROP_FPS, FPS)     # Comment-out for OpenCV 4.1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get the actual frame size
    # OpenCV 4.1 does not get the correct frame size
    #act_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #act_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    act_width = args.width
    act_height = args.height
    frame_info = 'Frame:%dx%d' %  (act_width, act_height)

    WINDOW_NAME = 'YoloV8 seg'

    model = YOLO('yolov8n-seg.pt')

    if not cap.isOpened():
    	print('Can not open camera.')
    	sys.exit()

    while True:
        ret, img = cap.read()

        if ret != True:
            break

        results = model(img)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        #cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(10)
        if key == 27: # ESC
            break

    # Release the capture object
    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
