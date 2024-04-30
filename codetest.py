#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import pprint
import graphviz

import cv2 as cv
import numpy as np
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import CvFpsCalc
from model import HandSignClassifier
from model import PointHistoryClassifier
import landmark_functions as lf


data_dir = 'datasets/test'
use_static_image_mode = True
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

use_brect = True

datagen = ImageDataGenerator(dtype='uint8')
image_iterator = datagen.flow_from_directory(data_dir,
                                             class_mode='categorical',
                                             batch_size=1)
class_indices = image_iterator.class_indices
indices_to_class = {v: k for k, v in class_indices.items()}

# Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

#  ########################################################################
frame = 0
for image, class_array in image_iterator:
    frame = frame + 1
    class_index = np.argmax(class_array)
    letter = indices_to_class[class_index]
    image = np.squeeze(image)

    # Process Key (ESC: end) #################################################
    key = cv.waitKey(1)
    if key == 27 or frame > 1:  # ESC
        break

    image.flags.writeable = False
    results = hands.process(image)

    # print("Type:", type(results))
    #
    # print("Attributes:", vars(results))
    #
    # print("Attributes:", dir(results))

    mhl = results.multi_hand_world_landmarks
    print("Stufe 0: ", mhl)
    pprint.pprint(mhl)
    for n1, l_hand in enumerate(mhl):
        print("Stufe 1: ", n1)
        pprint.pprint(l_hand)
        for n2, l in enumerate(l_hand.landmark):
            print("Stufe 2: ", n2)
            pprint.pprint(l)

    mh = results.multi_handedness
    print("results.multi_handedness")
    pprint.pprint(mh[0])
    for n1, h_hand in enumerate(mh):
        print("for hand in mh", n1)
        pprint.pprint(h_hand.classification)
        print("hc = hand.classification[0]")
        hc = h_hand.classification[0]
        pprint.pprint(hc)
        print("index = hc.index")
        index = hc.index
        pprint.pprint(index)