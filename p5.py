
# code copied from P5 Q&A video on U-t00b

import glob
import os
import sys
import time
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# when I write the outermost loop, these are the parameters
# that will be varied systematically to optimize the algorithm
# but as David Byrne once said: I ain't got time for that now

### TODO: Tweak these parameters and see how the results change.
#color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cells_per_block = 2 # HOG cells per block
#hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins

# used for false-pos rejection
heatmap_frame = []
frame_count = 0


def load_class_images(class_name):
    basedir = 'examples/'+class_name+'s/'  # this singular/plural thing is silly, I know
    image_types = os.listdir(basedir)
    image_files = []
    for imtyp in image_types:
        print(basedir + imtyp + '/*')
        image_files.extend(glob.glob(basedir + imtyp + '/*'))
    
    print( '{0} {1}s found'.format(len(image_files), class_name) )

    with open(class_name+'-imglist.txt', 'w') as f:
        for fname in image_files:
            f.write( fname + '\n' )
            
    return image_files
    
    
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
    
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    

# Define a function that takes an image,
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
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis=False, hist_bins=32, orient=9):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                if vis==True:
                    hogfeet, hog_image = get_hog_features(feature_image[:,:,channel], 
                                                          orient, pix_per_cell, cell_per_block, 
                                                          vis=vis, feature_vec=True)
                else:
                    hogfeet = get_hog_features(feature_image[:,:,channel], 
                                               orient, pix_per_cell, cell_per_block, 
                                               vis=vis, feature_vec=True)
                hog_features.extend(hogfeet)      
        else:
            if vis==True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis==True:
        return np.concatenate(img_features), hog_image
        
    return np.concatenate(img_features)


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, vis=False,
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
def cvt_color(img, conversion):
    if conversion == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        
        
def apply_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] = 0
    return heatmap
    
    
def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,50,255), 6)
    return img
    
    
def find_cars(image, scale):
    
    global X_Scaler
    global svc
 
    ### TODO: Tweak these parameters and see how the results change.
    global orient #= 9  # HOG orientations
    global pix_per_cell #= 8 # HOG pixels per cell
    global cells_per_block #= 2 # HOG cells per block
    #hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    global spatial_size #= (32, 32) # Spatial binning dimensions
    global hist_bins #= 32    # Number of histogram bins
    
    #color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    color_conversion = 'RGB2YCrCb'
    winsize = 64
    ystart = image.shape[0]//2
    ystop = image.shape[0] - winsize
   
    draw_img = np.copy(image)
    heatmap = np.zeros_like(image[:,:,0])
    img = image.astype(np.float32)/255
    img_search_region = img[ystart:ystop,:,:]
    ctrans = cvt_color(img_search_region, color_conversion)
    if scale != 1:
        imshape = ctrans.shape
        ctrans = cv2.resize(ctrans, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch0 = ctrans[:,:,0] # Y
    ch1 = ctrans[:,:,1] # Cr
    ch2 = ctrans[:,:,2] # Cb
    
    num_xblx = (ch0.shape[1] // pix_per_cell) - 1
    num_yblx = (ch0.shape[0] // pix_per_cell) - 1
        
    feat_per_blk = orient * cells_per_block**2
    blk_per_win = (winsize // pix_per_cell) - 1
    cells_per_step = 2
    xsteps = (num_xblx - blk_per_win) // cells_per_step
    ysteps = (num_yblx - blk_per_win) // cells_per_step

    hog0 = get_hog_features(ch0, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cells_per_block, feature_vec=False)
    
    for xb in range(xsteps):
        for yb in range(ysteps):
            
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            
            hogfeat0 = hog0[ypos:ypos+blk_per_win,xpos:xpos+blk_per_win].ravel()
            hogfeat1 = hog1[ypos:ypos+blk_per_win,xpos:xpos+blk_per_win].ravel()
            hogfeat2 = hog2[ypos:ypos+blk_per_win,xpos:xpos+blk_per_win].ravel()
            hog_features = np.hstack((hogfeat0,hogfeat1, hogfeat2))
            
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            
            subimg = cv2.resize(ctrans[ytop:ytop+winsize, xleft:xleft+winsize], (64,64))
            
            spatial_feat = bin_spatial(subimg, size=spatial_size)
            hist_feat = color_hist(subimg, nbins=hist_bins)
            test_features = X_scaler.transform(np.hstack((spatial_feat, hist_feat, hog_features)).reshape(1, -1))
            
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(winsize*scale) # but no 'lose or' yuk yuk
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart),(0,50,255))
                #img_boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
                    
    return draw_img, heatmap           
            
    
def pre_hogified_finder():
    out_images = []
    out_maps = []
    #out_boxes = []
    scale = 1.5
    
    searchpath = 'test_images/*'
    test_imgs = glob.glob(searchpath)
    
    for img_src in test_imgs:
        img_boxes = []
        t = time.time()
        count = 0
        image = mpimg.imread(img_src)
        out_img, heat_map = find_cars(image, scale)
        labels = label(heat_map)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        out_images.append(draw_img)
        out_maps.append(heat_map)
        
        
def integrate(hmap, frames=5):
    global heatmap_frame
    global frame_count
    
    # init only once
    if frame_count == 0:
        for i in range(frames):
            heatmap_frame.append(np.zeros_like(hmap))
    
    # add in the new frame
    heatmap_frame[frame_count % frames] = hmap
    integrated_heatmap = hmap
    # sum the frames
    for frame in range(frames):
        if frame != (frame_count % frames):
            integrated_heatmap = np.add(integrated_heatmap, heatmap_frame[frame])
    # apply threshold
    return apply_threshold(integrated_heatmap, 2)
        
        
def process_image(img):
    out_img, out_heatmap = find_cars(img, 1.5)
    heatmap = integrate( out_heatmap, 5 )
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(img), labels)

        
def train_clf():
    
    cars = load_class_images( 'vehicle' )
    uncars = load_class_images( 'non-vehicle' )

    ### TODO: Tweak these parameters and see how the results change.
    
    global orient #= 9  # HOG orientations
    global pix_per_cell #= 8 # HOG pixels per cell
    global cells_per_block #= 2 # HOG cells per block
    global spatial_size #= (32, 32) # Spatial binning dimensions
    global hist_bins #= 32    # Number of histogram bins

    # -------- danger, Will Robinson: if change these, change find_cars() too !!!
    
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    
    # -------- end danger zone
        
    # only needed to do this a finite # of times, now done with it
    if False:
        randomly_selected_car = np.random.randint(0,len(cars))
        randomly_selected_notcar = np.random.randint(0,len(uncars))
        
        car_pic = mpimg.imread(cars[randomly_selected_car])
        notta_car_pic = mpimg.imread(uncars[randomly_selected_notcar])
        # the lines below were used to create "figure1" for the writeup
        mpimg.imsave('output_images/randomcar.png', car_pic)
        mpimg.imsave('output_images/randomnoncar.png', notta_car_pic)
        
        print('extracting features from a random car image')
        car_features, car_hog_img = single_img_features(car_pic, 
                                                        hog_feat=hog_feat,
                                                        hog_channel=hog_channel, 
                                                        spatial_feat=spatial_feat, 
                                                        hist_feat=hist_feat, 
                                                        color_space=color_space, 
                                                        spatial_size=spatial_size, 
                                                        hist_bins=hist_bins, 
                                                        orient=orient, 
                                                        pix_per_cell=pix_per_cell, 
                                                        cell_per_block=cells_per_block,
                                                        vis=True
                                                        )
                                
        print('extracting features from a random not-a-car image')
        uncar_features, uncar_hog_img = single_img_features(notta_car_pic, 
                                color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cells_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                vis=True,
                                hist_feat=hist_feat, hog_feat=hog_feat)
                                
        # the lines below were used to create "figure2" for the writeup
        mpimg.imsave('output_images/randomcarhog.png', car_hog_img)
        mpimg.imsave('output_images/randomnoncarhog.png', uncar_hog_img)
                                                   
    print('extracting features from all the car images')
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cells_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
                            
    print('extracting features from all the "uncar" images')
    uncar_features = extract_features(uncars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cells_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    print('normalizing the extracted features')
    X = np.vstack((car_features, uncar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    print('labeling the 2 classes')
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(uncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('fitting a classifier')
    print('Using:', orient, 'orientations', pix_per_cell,
        'pixels per cell and', cells_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    if False:
        searchpath = 'test_images/*'
        test_imgs = glob.glob(searchpath)
        images = []
        titles = []
        overlap = 0.5
    
    #for img_src in test_imgs:
    if False:
        
        # Check the prediction time for a single sample
        t=time.time()

        image = mpimg.imread(img_src)
        draw_image = np.copy(image)
        # verify that scaling is 0..255
        print(image.shape, image[0][0][0], np.min(image), np.max(image))

        # need the following line because we extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image we are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32)/255

        y_start_stop = [image.shape[0]//2, image.shape[0]] # Min and max in y to search in slide_window()

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                                xy_window=(96, 96), xy_overlap=(overlap, overlap))

        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cells_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)                       

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 50, 255), thick=6)                    
        mpimg.imsave('output_images/'+img_src, window_img)
        print(time.time() - t, 'seconds to process {0} windows'.format(len(windows)))
    
    return X_scaler, svc
    

if __name__ == '__main__':
    
    # train the classifier
    global X_Scaler
    global svc
    X_scaler, svc = train_clf()
    
    print('classifier trained, processing video')
    if len(sys.argv) > 1:
        print(len(sys.argv), '==> test video')
        vid_out = 'test.mp4'
        clip = VideoFileClip('test_video.mp4')
    else:
        print(len(sys.argv), '==> full video')
        vid_out = 'p5.mp4'
        clip = VideoFileClip('project_video.mp4')
        
    test_clip = clip.fl_image(process_image)
    test_clip.write_videofile(vid_out, audio=False)
    print('submit it!')
    
    