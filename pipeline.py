import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from skimage.util import view_as_windows
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from collections import deque
from scipy.ndimage.measurements import label



# global variables
clf = joblib.load('clf.pkl')
X_scaler = joblib.load('scaler.pkl')
H = deque(maxlen=10)
thresh = 50 

def make_colorhist(channel):
    hist = np.histogram(channel, bins = 32, range = (0,256))
    return hist[0].reshape((1,-1))

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), 
	        (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img

def get_bboxes(img,size,stride_eighths):
    '''
    This function scans over the image with square sliding windows of side length size, and 
    with stride = (stride_eighths/8)*size.  stride_eighths should be an integer. 
    It runs a classifier on each window and returns x,y coordinates
    of bounding boxes for windows in which a car is found.
    '''
    # this is the scale factor to bring a window of side length size to size 64
    # this is used for the HOG and color histogram features
    fac64 = 64/size
    # this is the scale factor to bring a window of side length size down to size 32
    # this is used for the raw pixel values.
    fac32 = 32/size
    h,w = img.shape[:2]
    img64 = cv2.resize(img, (int(w*fac64), int(h*fac64)))
    img32 = cv2.resize(img, (int(w*fac32), int(h*fac32)))
    hls = cv2.cvtColor(img64, cv2.COLOR_BGR2HLS)
    featL = hog(hls[:,:,1], orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                visualise=False, feature_vector=False)
    featS = hog(hls[:,:,2], orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                visualise=False, feature_vector=False)
    stride64 = int(stride_eighths*64/8)
    stride32 = int(stride_eighths*32/8)
    # view_as_windows creates an array of windows with the specified size and stride
    windows64 = view_as_windows(img64, (64,64,3),stride64)
    windows32 = view_as_windows(img32, (32,32,3),stride32)
    # these arrays should have the same number of rows and columns 
    assert windows64.shape[0:2] == windows32.shape[0:2]
    # if assertion passes we get the number of rows and columns from windows64
    num_rows = windows64.shape[0]
    num_cols = windows64.shape[1]
    # the number of features is 7088
    X = np.zeros((num_rows*num_cols, 7088))
    # for loop to iterate over each window
    count = 0
    for i in range(num_rows):
        for j in range(num_cols):
            window1 = windows64[i,j,0,:,:,:]
            Bcolor = make_colorhist(window1[:,:,0])
            Gcolor = make_colorhist(window1[:,:,1])
            Rcolor = make_colorhist(window1[:,:,2])
            window2 = windows32[i,j,0,:,:,:]
            pixels = window2.reshape(1,-1)
            LHOG = featL[stride_eighths*i:stride_eighths*i+7,
	            stride_eighths*j:stride_eighths*j+7,:,:,:].reshape(1,-1)
            SHOG = featS[stride_eighths*i:stride_eighths*i+7,
	            stride_eighths*j:stride_eighths*j+7,:,:,:].reshape(1,-1)
            features = np.hstack((pixels,LHOG,SHOG,Bcolor,Gcolor,Rcolor))
            X[count]=features
            count = count + 1
    X_scaled = X_scaler.transform(X)
    y = clf.predict(X_scaled)
    y = y.reshape((num_rows,num_cols))
    stride = int(size*stride_eighths/8)
    bboxes = []
    for i in range(num_rows):
        for j in range(num_cols):
            if (y[i,j] == 1):
                bboxes.append([(j*stride, i*stride),(j*stride+size,i*stride+size)])
    return bboxes

def process_image(img):
    # crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    y_top = 400
    x_left = 400 
    crop = img[y_top:675,x_left:,:]
    bboxes1 = get_bboxes(crop,64,2) 
    bboxes2 = get_bboxes(crop,96,2) 
    bboxes3 = get_bboxes(crop,128,2) 
    bboxes = bboxes1 + bboxes2 + bboxes3
    hotboxes = np.zeros_like(img)
    #img_copy = img.copy()
    for box in bboxes:
        #cv2.rectangle(img_copy, (x_left + box[0][0], y_top + box[0][1]),
	#        (x_left+box[1][0],y_top+box[1][1]),(0,255,0),2)
        hotboxes[y_top + box[0][1]:y_top + box[1][1], x_left+box[0][0]:x_left+box[1][0],1] += 1
    #cv2.imshow('image',img_copy)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    H.append(hotboxes)
    heatmap = sum(H)
    print(np.max(heatmap))
    heatmap = (heatmap > thresh)
    labels = label(heatmap)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_labeled_bboxes(img, labels)
    return img 

def main():
    output_movie = 'annotated.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(process_image)
    project_clip.write_videofile(output_movie, audio=False)

if __name__ == "__main__":
    main()
