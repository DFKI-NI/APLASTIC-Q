import numpy as np
import os
import time

from matplotlib import pyplot
from PIL import Image
from skimage import io

# only evaluate a section of input image
def cut_im_to_sections(image, cut_im_sect):
	'''
	Parameters
	----------
	image : NP.ARRAY
	cut_im_sect : List with 4 floats = [top, bot, left, right] OR None
		If None, function return image	
		Each float is smaller than 1
		
	Returns
	-------
	image : NP.ARRAY
	'''
	if type(cut_im_sect) is list:
		image = image[round(image.shape[0]*cut_im_sect[0]):round(image.shape[0]*cut_im_sect[1]),
					round(image.shape[1]*cut_im_sect[2]):round(image.shape[1]*cut_im_sect[3]),
					:]
	return image

#  Function for creating the input which can be passed to a neural network
def imageslicer_modelinput(path_and_name_of_image,
                           image_size,
                           file_format = 'JPG',
						   cut_im_sect = None,
						   image_size_PLD = None):
	'''
	feeds an image as grid into a keras model

	Parameters
	----------
	path_and_name_of_image : str
		path and name to str.
	image_size : int
		tile size of PLD or PLQ.
	file_format : str, optional
		either 'JPG', 'PNG', 'TIF'. The default is 'JPG'.
	cut_im_sect : List with 4 floats = [top, bot, left, right] OR None
		If None, function return image	
		Each float is >=0 and <=1
	image_size_PLD : BOOL, optional
		1 if modelinput for PLD. The default is None.

	Returns
	-------
	X : np.arr
		dim = (image_size, image_size, 3, grid_row_len * grid_col_len )
	(grid_row_len, grid_col_len) : tuple with 2 ints
		is needed to rescale to Classificationmatrix later

	'''
	if file_format not in ['JPG', 'PNG', 'TIF', 'ARR']:
		return(print('File Format is not yet supported, only: ', ['JPG', 'PNG', 'TIF', 'ARR']))
	
	if file_format in ['JPG', 'PNG']:
		image = pyplot.imread(path_and_name_of_image)
	
	if file_format in ['TIF']:
		image = io.imread(path_and_name_of_image)

	if file_format in ['ARR']:
		image = path_and_name_of_image.copy()
	
	image = cut_im_to_sections(image, cut_im_sect)

	shape = image.shape
	if type(image_size_PLD) == int:
		grid_row_len = shape[0]//image_size - (shape[0]//image_size)%2
		grid_col_len = shape[1]//image_size - (shape[1]//image_size)%2
	if type(image_size_PLD) != int:
		grid_row_len = shape[0]//image_size
		grid_col_len = shape[1]//image_size
		
	#array where results get saved
	X = np.zeros((grid_row_len*grid_col_len,
			   image_size,
			   image_size,
			   shape[2]))

	if file_format in ['TIF']:
		X_alpha = np.ones((grid_row_len*grid_col_len))
	
		for i in range(grid_row_len):
			for j in range(grid_col_len):
				X[i*grid_col_len+j] = image[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :]
				if np.count_nonzero(image[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, 3] ==0) > 144:
					X_alpha[i*grid_col_len+j] = 0 # so that alpha squares do not get evaluated
	
	if file_format in ['JPG', 'PNG', 'ARR']:
		for i in range(grid_row_len):
			for j in range(grid_col_len):
				X[i*grid_col_len+j] = image[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :]
			
	if file_format in ['JPG', 'TIF', 'ARR']:
		X = X.astype('float32')
		X = X[:,:,:,:3]
		X /= 255
		if file_format in ['TIF']:
			return (X, X_alpha), (grid_row_len, grid_col_len)

	if file_format in ['PNG']:
		X = X[:,:,:,:3]
		
	return X, (grid_row_len, grid_col_len)



