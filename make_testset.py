import numpy as np
import cv2
import os
import glob
from skimage.util import view_as_windows

for i in range(7,10):
    if (i <= 6):
        file = './test_images/test'+str(i)+'.jpg'
    else:
        file = './test_images/test'+str(i)+'.png'
    print(file)
    img = cv2.imread(file)
    crop = img[400:675,:,:]
    h,w = crop.shape[0:2]
    crop2 = cv2.resize(crop, (int(w/2),int(h/2)))
    A = view_as_windows(crop, (64,64,3), 16)
    B = view_as_windows(crop2, (64,64,3), 16)
    dir = './test_set/test'+str(i)+'/'
    k = 1
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            window = A[i,j,0,:,:,:]
            out_file = dir+'image'+str(k)+'.png'
            cv2.imwrite(out_file,window)
            k = k + 1
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            window = B[i,j,0,:,:,:]
            out_file = dir+'image'+str(k)+'.png'
            cv2.imwrite(out_file,window)
            k = k + 1
