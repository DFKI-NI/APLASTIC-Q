# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:59:44 2020

@author: Mattis Wolf
"""
from skimage.transform import resize
from matplotlib import pyplot
from PIL import Image

def resize_logo(image_arr, logo_arr, logo_size):
    """
    image_arr = np.array with 3 colorchannels
    logo_arr  = np.array with 3 colorchannels
    logo_size = float, >0 and <1
    """
    h, w, d       = image_arr.shape
    h_l, w_l, d_l = logo_arr.shape
    rel_fac  = min(h,w)//(1/logo_size)
    sizing_f = rel_fac / max(h_l, w_l)
    logo_arr = resize(logo_arr, (int(h_l*sizing_f), int(w_l*sizing_f), d_l))
    return logo_arr    

def imprint_logo(image_arr, logo_arr, edge_d = 0.05):
    """
    image_arr = np.array with 3 colorchannels
    logo_arr  = np.array with 3 colorchannels
    edge_d    = float, >0 and <1
    """
    h, w, d       = image_arr.shape
    h_l, w_l, d_l = logo_arr.shape
    edge_d = int(min(h,w)*edge_d)
    #image_arr[-(h_l+edge_d):-edge_d, -(w_l+edge_d):-edge_d, :] = logo_arr[:,:,:3]
    image_arr[edge_d:(h_l+edge_d), -(w_l+edge_d):-edge_d, :3] = logo_arr[:,:,:3]
    return image_arr

def watermark_png(path_input_png, path_logo_png, logo_size):
	'''
	Parameters
	----------
	path_input_png : STRING
		path to PNG image which will be imprinted with logo.
	path_logo_png : STRING
		path to PNG logo which will be imprinted.
	logo_size : FLOAT
		ranging between 0 and 1. higher values mean big logo size.

	Returns nothing
	-------
	input image will be corrupted.

	'''
	
	im   = pyplot.imread(path_input_png)*255
	im   = im.astype('uint8')
	im_  = im.copy()
	logo = pyplot.imread(path_logo_png)*255
	
	logo = resize_logo(im, logo, logo_size)
	im_  = imprint_logo(im_, logo, 0.05)
	
	im_save = Image.fromarray(im_)
	im_save.save(path_input_png)
