#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import time
from collections import Counter
from collections import deque
from PIL import Image, ImageTk

import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox

from model.keypoint_classifier.control_classifier import ControlClassifier
from utils import CvFpsCalc
from model import HandSignClassifier
# from model import PointHistoryClassifier

import landmark_functions as lf

recorded_word = ""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():

    # Spelling settings
    spelling_cooldown = 1
    global recorded_word

    # Create GUI
    root = tk.Tk()
    root.title("Yebra")

    frame_imgs = tk.Frame(root)
    frame_imgs.pack(side='top')

    tk_guide = ImageTk.PhotoImage(Image.open("ASLalphabet.jpg"))
    label_guide = tk.Label(frame_imgs, image=tk_guide)
    label_guide.pack(side='left')

    label_video = tk.Label(frame_imgs)
    label_video.pack(side='right')

    frame_shlr = tk.Frame(root)
    frame_shlr.pack(side='top', padx=15)

    button_del = tk.Button(frame_shlr, text="Reset Text", command=reset_word,
                           activebackground='red', activeforeground='white')
    button_del.pack(side='right', padx=100)

    info_text = ("Select your preferred hand for spelling down at the bottom.\n"
                 "The other hand will be used for entering a letter.\n"
                 "Perform a thumbs up gesture to enter the letter currently depicted by your spelling hand.")
    button_info = tk.Button(frame_shlr, text="How to use",
                            command=lambda: messagebox.showinfo("How to use", info_text))
    button_info.pack(side='left', padx=100)

    label_sh = tk.Label(frame_shlr, text="Spelling Hand")
    label_sh.pack(side='top', pady=2)

    frame_lr = tk.Frame(frame_shlr)
    frame_lr.pack(side='top')

    var_sh = tk.IntVar(value=1)
    rbutton_left = tk.Radiobutton(frame_lr, text="Left", variable=var_sh, value=0)
    rbutton_left.pack(side='left', padx=5, pady=1)
    rbutton_right = tk.Radiobutton(frame_lr, text="Right", variable=var_sh, value=1)
    rbutton_right.pack(side='left', padx=5, pady=1)



    # Argument parsing
    args = get_args()

    guide = cv.imread('ASLalphabet.jpg')
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    use_3D = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    hand_sign_classifier = HandSignClassifier()
    control_classifier = ControlClassifier()

    # point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/customModel/ASLclassifier_label.csv',
              encoding='utf-8-sig') as f:
        letter_labels = csv.reader(f)
        letter_labels = [row[0] for row in letter_labels]
    with open('model/customModel/control_label.csv',
              encoding='utf-8-sig') as f:
        control_labels = csv.reader(f)
        control_labels = [row[0] for row in control_labels]
    # with open(
    #         'model/point_history_classifier/point_history_classifier_label.csv',
    #         encoding='utf-8-sig') as f:
    #     point_history_classifier_labels = csv.reader(f)
    #     point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # FPS measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    # history_length = 16  # 16
    # point_history = deque(maxlen=history_length)

    # Finger gesture history
    # finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    # mode = 0
    record_letter = False
    last_activation_time = 0
    max_word_len = 22

    while True:

        try:
            root.update()
            if not root.winfo_exists():
                break
        except tk.TclError as e:
            break

        spelling_hand = var_sh.get()

        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(1)
        if key == 27:   # ESC
            break
        if key == 8:    # Del
            recorded_word = ""
        # number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                handedness_id = handedness.classification[0].index
                spelling = handedness_id == spelling_hand

                # Bounding box calculation
                brect = lf.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = lf.calc_landmark_list(debug_image, hand_landmarks, include_z=use_3D)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = lf.pre_process_landmark(landmark_list, include_z=use_3D)
                # pre_processed_point_history_list = lf.pre_process_point_history(debug_image, point_history)

                # Hand sign classification
                classifier_input = [handedness_id] + pre_processed_landmark_list

                hand_color = (255, 255, 255)
                current_time = time.time()
                if spelling:
                    hand_sign_id = hand_sign_classifier(classifier_input)
                    keypoint_classifier_labels = letter_labels
                    if record_letter and current_time - last_activation_time > spelling_cooldown:
                        hand_color = (0, 255, 0)
                        if hand_sign_id == 26:  # [del]
                            if recorded_word:
                                recorded_word = recorded_word[:-1]
                        elif hand_sign_id == 27:    # [space]
                            recorded_word = recorded_word + " "
                        else:
                            recorded_word = recorded_word + keypoint_classifier_labels[hand_sign_id]

                        if len(recorded_word) > max_word_len:
                            recorded_word = recorded_word[1:]

                        last_activation_time = current_time

                else:
                    hand_sign_id = control_classifier(classifier_input)
                    keypoint_classifier_labels = control_labels
                    record_letter = hand_sign_id == 0
                    if record_letter:
                        hand_color = (0, 255, 0)
                    else:
                        hand_color = (0, 255, 255)

                # Point history
                # if hand_sign_id == 200:  # Point gesture
                #     point_history.append(landmark_list[8])
                # else:
                #     point_history.append([0, 0])

                # Finger gesture classification
                # finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list)
                # if point_history_len == (history_length * 2):
                #     finger_gesture_id = point_history_classifier(
                #         pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                # finger_gesture_history.append(finger_gesture_id)
                # most_common_fg_id = Counter(
                #     finger_gesture_history).most_common()

                # Drawing part
                landmarks_2D = lf.conv_to_2D(landmark_list)
                debug_image = lf.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = lf.draw_landmarks(debug_image, landmarks_2D, primary_color=hand_color)
                debug_image = lf.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    # point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            # point_history.append([0, 0])
            pass

        # debug_image = lf.draw_point_history(debug_image, point_history)
        debug_image = lf.draw_info(debug_image, fps)
        debug_image = lf.draw_letters(debug_image, recorded_word)

        # Screen reflection #############################################################
        tk_video = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        tk_video = ImageTk.PhotoImage(Image.fromarray(tk_video))
        label_video.config(image=tk_video)
        label_guide.config(image=tk_guide)

    try:
        root.destroy()
    except tk.TclError as e:
        pass

    cap.release()
    cv.destroyAllWindows()


def reset_word():
    global recorded_word
    recorded_word = ""


if __name__ == '__main__':
    print("Loading ...")
    main()
