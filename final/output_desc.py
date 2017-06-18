from sampler_functions import *
import pickle
import numpy as np
import cv2
from os.path import join
import os


sampler = SR()
category = sampler.get_category("/Users/qiuwshou/Downloads/UCF50")
sampler.split_data(category,"/Users/qiuwshou/Desktop/train","/Users/qiuwshou/Desktop/test")

sampler.create_key_frame("/Users/qiuwshou/Desktop/train","/Users/qiuwshou/Desktop/train_frame")
sampler.create_key_frame("/Users/qiuwshou/Desktop/test","/Users/qiuwshou/Desktop/test_frame")


label=sampler.frame_category("/Users/qiuwshou/Desktop/keyframe")
print(label)

sampler.cal_dof("/Users/qiuwshou/Desktop/train_frame","/Users/qiuwshou/Desktop/train_dof")
sampler.cal_dof("/Users/qiuwshou/Desktop/test_frame","/Users/qiuwshou/Desktop/test_dof")

sampler.codebook_dof("/Users/qiuwshou/Desktop/train_dof","/Users/qiuwshou/Desktop/train_hist/",200,2)
sampler.codebook_dof("/Users/qiuwshou/Desktop/test_dof","/Users/qiuwshou/Desktop/test_hist/",200,2)


sampler.codebook_dof_all("/Users/qiuwshou/Desktop/train_hist","/Users/qiuwshou/Desktop/train_cen.txt",200)
sampler.codebook_dof_all("/Users/qiuwshou/Desktop/test_hist","/Users/qiuwshou/Desktop/test_cen.txt",200)

sampler.hist_dof("/Users/qiuwshou/Desktop/train_cen.txt","/Users/qiuwshou/Desktop/train_hist",
                 "/Users/qiuwshou/Desktop/train_motion.txt","/Users/qiuwshou/Desktop/train_category.txt")
sampler.hist_dof("/Users/qiuwshou/Desktop/test_cen.txt","/Users/qiuwshou/Desktop/test_hist",
                 "/Users/qiuwshou/Desktop/test_motion.txt","/Users/qiuwshou/Desktop/test_category.txt")

sampler.codebook_csift("/Users/qiuwshou/Desktop/train_csift","/Users/qiuwshou/Desktop/train_dof",
                       "/Users/qiuwshou/Desktop",200,2)
sampler.codebook_csift("/Users/qiuwshou/Desktop/train_csift","/Users/qiuwshou/Desktop/train_dof",
                       "/Users/qiuwshou/Desktop",200,2)

sampler.hist_csift("/Users/qiuwshou/Desktop","/Users/qiuwshou/Desktop/train_csift",
                   "/Users/qiuwshou/Desktop/train_dof","/Users/qiuwshou/Desktop",200,2)
sampler.hist_csift("/Users/qiuwshou/Desktop","/Users/qiuwshou/Desktop/test_csift",
                   "/Users/qiuwshou/Desktop/test_dof","/Users/qiuwshou/Desktop",200,2)

sampler.svm_cls("/Users/qiuwshou/Desktop/train_motion.txt","/Users/qiuwshou/Desktop/test_motion.txt")
sampler.svm_cls("/Users/qiuwshou/Desktop/hits_all.txt","/Users/qiuwshou/Desktop/hist_all(1).txt")


