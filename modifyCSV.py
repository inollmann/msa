import numpy as np
import landmark_functions as lf


# print('Starting...')
# lm_left = lf.read_csv('keypoints/keypointB_3D_left.csv')
# lm_left = lf.delete_wrong_classifications(lm_left, 1)
#
# lm_right = lf.read_csv('keypoints/keypointB_3D_right.csv')
# lm_right = lf.delete_wrong_classifications(lm_right, 0)
#
# lm_full = np.concatenate((lm_left, lm_right), axis=0)
#
# lf.write_csv('keypoints/keypointB_3D_full.csv', lm_full)
# print('Finished')

lm_full = lf.read_csv('keypoints/keypointB_3D_full.csv', datatype='npstring')
lm_aug = lf.random_rotation(lm_full)

lf.write_csv('keypoints/keypointB_3D_augmented.csv', lm_aug)

