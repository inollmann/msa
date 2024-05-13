import csv
import numpy as np
import itertools
import copy
import cv2 as cv


# CSV-related functions

def read_csv(csv_path, datatype='list'):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    match datatype:
        case 'list':
            pass
        case 'npfloat':
            data = np.array(data, dtype=float)
        case 'npstring':
            data = np.array(data)
        case _:
            pass

    return data


def write_csv(csv_path, data):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def logging_csv(letter, handedness, landmark_list, csv_path):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter, handedness, *landmark_list])
    return


def array2list(arr):
    if arr.shape == (1, 42):
        arr.reshape(21, 2)
        return arr
    else:
        print("Wrong shape of landmark array")


# Landmark Preprocessing

def pre_process_landmark(landmark_list, include_z=False):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
            if include_z:
                base_z = landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        if include_z:
            temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def conv_to_2D(landmark_list):
    landmarks_2D = []
    for row in landmark_list:
        landmarks_2D.append(row[:2])

    return landmarks_2D


# Transform to Image Coordinates

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks, include_z=False):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        if include_z:
            landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y, landmark_z])
        else:
            landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def delete_wrong_classifications(landmark_array, key):
    if isinstance(key, int):
        key = str(key)
    landmark_array = np.array(landmark_array)
    mask = landmark_array[:, 1] != key
    cleaned_array = landmark_array[mask]

    return cleaned_array


def encode_handedness(landmark_array, column):
    landmark_array[landmark_array[:, 1] == 'Left', column] = '0'
    landmark_array[landmark_array[:, 1] == 'Right', column] = '1'

    return landmark_array


# Augmentation

def random_rotation(landmark_array, max_angle):
    lm_array = copy.deepcopy(landmark_array)
    aug_array = landmark_array[:, 3:].reshape(landmark_array.shape[0], 21, 2)
    aug_array = aug_array.astype(float)
    for i, hand in enumerate(aug_array):
        angle = np.random.uniform(0, np.radians(max_angle))
        rot_mat = np.array([[np.cos(angle), np.sin(angle)*-1],
                            [np.sin(angle), np.cos(angle)]])
        for j, xy in enumerate(hand):
            xy_rot = np.dot(rot_mat, xy.T)
            aug_array[i, j] = xy_rot.T
    aug_array = aug_array.reshape(landmark_array.shape[0], 42)
    lm_array[:, 3:] = aug_array.astype('<U22')

    return landmark_array


# Landmark Visualization

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text="", finger_gesture_text=""):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ': ' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps):
    x = 10
    y = 470
    cv.putText(image, "FPS: " + str(fps), (x, y), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS: " + str(fps), (x, y), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2, cv.LINE_AA)

    # mode_string = ['Logging Key Point', 'Logging Point History']
    # if 1 <= mode <= 2:
    #     cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
    #                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    #                cv.LINE_AA)
    #     if 0 <= number <= 9:
    #         cv.putText(image, "NUM:" + str(number), (10, 110),
    #                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    #                    cv.LINE_AA)
    return image


def draw_letters(image, letters):
    x = 10
    y = 30
    cv.putText(image, "Text: " + str(letters) + "_", (x, y), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "Text: " + str(letters) + "_", (x, y), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 255, 0), 2, cv.LINE_AA)

    return image


def draw_landmarks(image, landmark_point, primary_color=(255, 255, 255), secondary_color=(0, 0, 0)):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), primary_color, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), primary_color, 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), primary_color, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), primary_color, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), primary_color, 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), primary_color, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), primary_color, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), primary_color, 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), primary_color, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), primary_color, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), primary_color, 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), primary_color, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), primary_color, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), primary_color, 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), primary_color, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), primary_color, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), primary_color, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), primary_color, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), primary_color, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), primary_color, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), secondary_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), primary_color, 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, secondary_color, 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, secondary_color, 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, secondary_color, 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, secondary_color, 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, secondary_color, 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, primary_color, -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, secondary_color, 1)

    return image
