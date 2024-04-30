import csv
from collections import Counter
import numpy as np


def read_csv(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def write_csv(csv_path, data):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def dataset_size(arr, name):
    users = Counter(arr[:, 1])
    users = dict(sorted(users.items(), key=lambda x: x[0], reverse=False))

    letters = Counter(arr[:, 2])
    letters = dict(sorted(letters.items(), key=lambda x: x[0], reverse=False))

    print("\n\n", name, "\n")
    for element, count in users.items():
        print(f"{element}: {count} occurrences")
    # print("\n")
    for element, count in letters.items():
        print(f"{element}: {count} occurrences")


csv_file_original = 'keypoints/keypointB_1.csv'
csv_file_noDuplicates = 'keypoints/keypointB_noDuplicates.csv'
landmarkArray = np.array(read_csv(csv_file_original))

unique_frames, frame_occ = np.unique(landmarkArray[:, 0], return_counts=True)
duplicate_frames = np.where(frame_occ > 1)[0]
duplicate_frame_idx = []
for duplicate_value in unique_frames[duplicate_frames]:
    duplicate_indices = np.where(landmarkArray[:, 0] == duplicate_value)[0]
    duplicate_frame_idx.append(duplicate_indices)
# duplicate_frame_idx = np.sort(np.concatenate(duplicate_frame_idx))
# landmarks_noDuplicates = np.delete(landmarkArray, duplicate_frame_idx, axis=0)
# write_csv(csv_file_noDuplicates, landmarks_noDuplicates)
# write_csv(csv_file_noDuplicates, landmarkArray)
# print("Indices of duplicates:\n", duplicate_frame_idx)

dataset_size(landmarkArray, "Original")
# dataset_size(landmarks_noDuplicates, "No Duplicates")

