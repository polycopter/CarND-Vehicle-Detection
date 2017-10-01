from skimage.feature import hog
orient = 9
pix_per_cell = 8
cell_per_block = 2

feature_array = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=False)

# The output feature_array will have a shape of (n_yblocks, n_xblocks, 2, 2, 9), 
# where n_yblocks and n_xblocks are determined by the shape of your region of interest 
# (i.e. how many blocks fit across and down your image in x and y).

# So, for example, if you used cells_per_block=2 in extracting features from the 64x64 pixel training images, 
# then you would want to extract subarrays of shape (7, 7, 2, 2, 9) from feature_array 
# and then use np.ravel() to unroll the feature vector.