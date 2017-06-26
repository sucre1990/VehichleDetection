import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, \
        pixels_per_cell=(pix_per_cell, pix_per_cell), \
        cells_per_block=(cell_per_block, cell_per_block), \
        visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features= hog(img, orientations=orient, \
        pixels_per_cell=(pix_per_cell, pix_per_cell), \
        cells_per_block=(cell_per_block, cell_per_block), \
        visualise=False, feature_vector=feature_vec)
        return features
    
cspace='LUV'
orient=8 
pix_per_cell=8
cell_per_block=2
hog_channel=0
spatial_size = (16,16)
hist_bins = 32
hist_bins_range = (0, 256)
bin_spatial_f = True
color_hist_f = True
hog_f = True
flip_aug = True


import pickle
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(X_scaler, f)
    
with open('scaler.pkl', 'rb') as f:
    X_scaler = pickle.load(f)
    
import pickle
# save model
# with open('svm_model.pkl', 'wb') as f:
#     pickle.dump(svc, f)
    
with open('svm_model.pkl', 'rb') as f:
    svc = pickle.load(f)
    
# This Function takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched 
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    
    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB', \
                   spatial_size=(32, 32), hist_bins=32, \
                   hist_range=(0, 256), orient=9, \
                   pix_per_cell=8, cell_per_block=2, \
                   hog_channel=0, spatial_feat=True, \
                   hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    img = img.copy()
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        
        #4) Extract features for that window using single_img_features()
        features = image_extract_features(test_img, cspace=color_space, orient = orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,\
                           hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_bins_range = (0, 256),\
                           bin_spatial_f=spatial_feat, color_hist_f=hist_feat, hog_f=hog_feat)

        #5) Scale extracted features to be fed to classifier
        features = ((np.concatenate(features)).astype(np.float64)).reshape(1,-1)
#         print(features)
#         print("...................")
        test_features = scaler.transform(features)
#         print(test_features)
#         print(">>>>>>>>>>>>>>>>>>>")

        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

from scipy.ndimage.measurements import label
print(2)
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_boxes(img, bboxes, thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], bbox[2], thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function that contains color space converting
def convert_color(img, cspace='LUV'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)  
    return feature_image

# Function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, cspace, spatial_size, hist_bins, hog_channel,\
             box_color = (0,0,255), cells_per_step = 4):
    hot_windows = []
    draw_img = np.copy(img)
    image = img.copy()
#     img = (img/255.).astype(np.float32)

    
    img_tosearch = image[ystart:ystop,:,:]


    ctrans_tosearch = convert_color(img_tosearch, cspace=cspace)


    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_list = []
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_list.append(hog_feat1)
            hog_list.append(hog_feat2)
            hog_list.append(hog_feat3)
            if hog_channel == 'ALL':
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = np.array(hog_list[hog_channel])

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)

            hist_features = color_hist(subimg, nbins=hist_bins)

            features = np.concatenate((spatial_features, hist_features, hog_features)).reshape(1, -1)
            # Scale features and make a prediction
            test_features = X_scaler.transform(features)    

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))   

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                
                hot_windows.append(((xbox_left, ytop_draw + ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),box_color))
    return hot_windows


def pipeline_vehicle_detection(image):
    ori_image = image
    image = (image/255.).astype(np.float32)

    hot_windows64 = find_cars(image, ystart = 380, ystop = 500, scale = 1, svc = svc, box_color = (255,0,0), X_scaler = X_scaler, orient = orient, pix_per_cell = pix_per_cell,\
                            cell_per_block = cell_per_block, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins,\
                            hog_channel = hog_channel, cells_per_step = 2)

    hot_windows128 = find_cars(image, ystart = 358, ystop = 582, scale = 2, svc = svc, box_color = (0,255,0), X_scaler = X_scaler, orient = orient, pix_per_cell = pix_per_cell,\
                            cell_per_block = cell_per_block, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins,\
                            hog_channel = hog_channel, cells_per_step = 2)

    hot_windows192 = find_cars(image, ystart = 357, ystop = 668, scale = 3, svc = svc, box_color = (0,0,255), X_scaler = X_scaler, orient = orient, pix_per_cell = pix_per_cell,\
                            cell_per_block = cell_per_block, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins,\
                            hog_channel = hog_channel, cells_per_step = 2)

    hot_windows256 = find_cars(image, ystart = 300, ystop = 720, scale = 4, svc = svc, box_color = (0,255,255), X_scaler = X_scaler, orient = orient, pix_per_cell = pix_per_cell,\
                            cell_per_block = cell_per_block, cspace = cspace, spatial_size = spatial_size, hist_bins = hist_bins,\
                            hog_channel = hog_channel, cells_per_step =2)


    hot_windows = hot_windows64 + hot_windows128 + hot_windows192 + hot_windows256

    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(ori_image), labels)
    return draw_img

 


def main():
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    project_output = 'project_out2.mp4'
    clip = VideoFileClip("project_video.mp4")
    project_clip = clip.fl_image(pipeline_vehicle_detection) 
    project_clip.write_videofile(project_output, audio=False)
    
if __name__ == '__main__':
    main()
