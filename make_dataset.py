import numpy as np
import cv2
import os
import glob
import re
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def make_hog(channel):
    features = hog(channel, orientations=10, pixels_per_cell=(8, 8), 
        cells_per_block=(2,2), visualise=False, feature_vector=True)
    return features.reshape((1,-1))    

def make_pixels(img):
    img_small = cv2.resize(img, (32,32))
    features = img_small.ravel()
    return features.reshape((1,-1))

def make_colorhist(channel):
    hist = np.histogram(channel, bins = 32, range = (0,256)) 
    return hist[0].reshape((1,-1))

def make_features(file):
    img = cv2.imread(file)
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    hog_featuresL = make_hog(hls[:,:,1])
    hog_featuresS = make_hog(hls[:,:,2])
    color_features = make_pixels(img)
    Bcolor = make_colorhist(img[:,:,0])
    Gcolor = make_colorhist(img[:,:,1])
    Rcolor = make_colorhist(img[:,:,2])
    features = np.hstack((color_features,hog_featuresL, hog_featuresS, 
        Bcolor, Gcolor, Rcolor))
    return features

vehicle_paths = ['./vehicles/GTI_Far','./vehicles/GTI_Left',
    './vehicles/GTI_MiddleClose', './vehicles/GTI_Right', 
    './vehicles/KITTI_extracted']

non_vehicle_paths = ['./non-vehicles/Extras','./non-vehicles/GTI']

paths = vehicle_paths + non_vehicle_paths


# There are 17760 total images in the dataset
# We have 2152 features
# 1568 * 2 + 1024*3
num_features = 7088 
X = np.zeros((17760,num_features))
count = 0
for path in paths:
    for file in glob.glob(os.path.join(path,'*.png')):
        features = make_features(file)
        X[count] = features
        count = count + 1
print(count)

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# make the target vector
# number of vehicle images is 8792
# number of non-vehicle images is 8698
y = np.concatenate((np.ones(8792),np.zeros(8968)))

np.save('X.npy',scaled_X)
np.save('y.npy',y)

cars=[[16,17,18,19,2,21,22,23,24,25,26,42,43,44,45,47,48,49,50,51,52],
      [1],
      [17,18,19],
      [10,16,17,18,19,21,22,23,24,25,42,43,44,45,47,48,49,50,51],
      [16,17,18,19,21,22,23,24,25,42,43,44,45,47,48,49,50,51],
      [16,17,18,19,20,21,22,23,24,42,43,44,45,46,47,48,49,50],
      [],
      [],
      []]

# make the test set
print('Starting the test')
test_size = 4725
X_test = np.zeros((test_size,num_features))
y_test = np.zeros((test_size,1))
count = 0
for i in range(1,10):
    path = './test_set/test'+str(i)
    for file in glob.glob(os.path.join(path,'*.png')):
        nums = re.findall(r'\d+',file)
        if (int(nums[1]) in cars[int(nums[0])-1]):
            y_test[count] = 1    
        features = make_features(file)
        X_test[count] = features
        count = count + 1

scaled_X_test = X_scaler.transform(X_test)

joblib.dump(X_scaler,'scaler.pkl')
np.save('X_test.npy',scaled_X_test)
np.save('y_test.npy',y_test)

