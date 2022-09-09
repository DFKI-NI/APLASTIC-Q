import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from matplotlib.colors import ListedColormap
from skimage import io
from imageslicer import cut_im_to_sections
from imageslicer import imageslicer_modelinput
from helperfunctions import matplotlib_params
from watermark import watermark_png

plt.rcParams.update(plt.rcParamsDefault)
pylab.rcParams.update(matplotlib_params)

def def_save_PARAMS(plastic_model_PLD, plastic_model_PLQ, labels_PLD, labels_PLQ,
						 image_name, path_to_save_figs,	 colors_PLD, colors_PLQ,
						 number_of_scence, save_dir_name, params_PLQ,
						 alt_correction, dense_few, litter_h_litter_l,
						 file_format, dpi, cut_im_sect):
	'''
	Saves Parameters in .txt file
	
	Parameters
	----------
	plastic_model_PLD : string
	plastic_model_PLQ : string
	labels_PLD : string
	labels_PLQ : string
	image_name : string
	path_to_save_figs : string
	colors_PLD : list
	colors_PLQ : list
	number_of_scence : int
	save_dir_name : string
	params_PLQ : list
	alt_correction : float or list
	dense_few : list
	litter_h_litter_l : list
	file_format : str
	dpi : int
	cut_im_sect : list
	
	Returns: .txt file 
	-------
	None.

	'''
	f=open(os.path.join(path_to_save_figs, 'PARAMETER_SETTINGS_'+save_dir_name+'.txt'),'w')
	f.write('PLD model name   =' + str(plastic_model_PLD.name) + '\n')
	f.write('PLQ model name   =' + str(plastic_model_PLQ.name) + '\n')
	f.write('Labels_PLD       =' + str(labels_PLD) + '\n')
	f.write('Labels_PLQ       =' + str(labels_PLQ) + '\n')
	f.write('image name       =' + str(image_name) + '\n')
	f.write('path_to_save_figs=' + str(path_to_save_figs) + '\n')
	f.write('colors_PLD       =' + str(colors_PLD) + '\n')
	f.write('colors_PLQ       =' + str(colors_PLQ) + '\n')
	f.write('number_of_scence =' + str(number_of_scence) + '\n')
	f.write('save_dir_name    =' + str(save_dir_name) + '\n')
	f.write('params_PLQ       =' + str(params_PLQ) + '\n')
	f.write('alt_correction   =' + str(alt_correction) + '\n')
	f.write('dense_few        =' + str(dense_few) + '\n')
	f.write('litter_h_litter_l=' + str(litter_h_litter_l) + '\n')
	f.write('file_format      =' + str(file_format) + '\n')
	f.write('dpi              =' + str(dpi) + '\n')
	f.write('cut_im_sect      =' + str(cut_im_sect) + '\n')
	f.close()

def scale_C_PLD(C, shape_of_grid_tuple):
	'''
	Scales Matrix C dimensions to 2*m x 2*n 
	each element will be represented 4-times in the rescaled C

	Parameters
	----------
	C : np.array
		Classificationsmatrix.
	shape_of_grid_tuple : tuple
		Classificationsmatrix-dimensions of PLQ.

	Returns
	-------
	C_scaled : np.array
		PLD-Classificationsmatrix rescaled

	'''
	C_scaled = np.zeros(shape_of_grid_tuple)
	for i in range(len(C_scaled)- len(C_scaled)%2):
		for j in range(len(C_scaled[0])- len(C_scaled[0])%2):
			C_scaled[i,j] = C[(i//2), (j//2)]
	return C_scaled


def prep_C_PLD(labels_PLD, colors_PLD, file_format):
	'''
	add new label for transparent sections in TIF files

	Parameters
	----------
	labels_PLD : list with str
	colors_PLD : list with str
	file_format : str.

	Returns
	-------
	labels_PLD_new : list with str
		labels with new label.
	colors_PLD : list with str
		color with new color.

	'''
	if file_format in ['TIF']:
		labels_PLD_new = labels_PLD + ['no data']
		colors_PLD.append('white')
		return labels_PLD_new, colors_PLD
	else: return labels_PLD, colors_PLD


def prep_C_PLQ(labels_PLQ, colors_PLQ, C_PLQ, C_PLQ_im, grid_shape_PLD):
	'''
	shrink PLQ grid to 2x PLD grid

	Parameters
	----------
	labels_PLQ : list with str
	colors_PLQ : list with str
	C_PLQ : np.array
		Classificationsmatrix.
	C_PLQ_im : TYPE
		DESCRIPTION.
	grid_shape_PLD : tuple
		Classificationsmatrix-dimensions of PLD.

	Returns
	-------
	labels_PLQ_new : list with str
	colors_PLQ : list with str
	C_PLQ : np.array
	C_PLQ_im : np.array

	'''
	labels_PLQ_new = labels_PLQ + ['    No pollution']
	C_PLQ = C_PLQ[:grid_shape_PLD[0]*2, :grid_shape_PLD[1]*2]
	C_PLQ_im = C_PLQ_im[:grid_shape_PLD[0]*2, :grid_shape_PLD[1]*2]
	return labels_PLQ_new, colors_PLQ, C_PLQ, C_PLQ_im


def cmatrix_transform(C, detected):
	'''
	transforms classificationmatrix C in a way that each color
	maps to the correct label	

	Parameters
	----------
	C : np.array
		Classificationsmatrix.
	detected : list of detected classes

	Returns
	-------
	C_new : np.array
		transformed classificationsmatrix.
	'''

	C_new = C.copy()
	for i in range(len(C)):
		for j in range(len(C[0])):
			C_new[i,j] = np.where(C[i,j]==detected)[0][0]
	return C_new


def polluted_area_helper(C_PLD, shape_of_grid):
	'''
	Outputs matrix with -1 = non polluted, and 1 = polluted entries'

	Parameters
	----------
	C_PLD : np.array
		Classificationsmatrix.
	shape_of_grid : tuple
		Classificationsmatrix-dimensions of PLQ.

	Returns
	-------
	C_scaled : np.array

	'''
	C_scaled = scale_C_PLD(C_PLD, shape_of_grid)
	C_scaled = C_scaled.flatten()
	C_scaled[np.where(C_scaled > 1)] = -1 # non polluted. If PLQ wants to be looked at: change to 0
	C_scaled[np.where(C_scaled == 0)] =  1 # polluted
	C_scaled[np.where(C_scaled == 1)] =  1 # polluted
	return C_scaled


def classify_im_postprocess(C_arr, shape_of_grid):
	'''
	classification of input image, returns classification matrix and others

	Parameters
	----------
	C_arr : np.arr of classifications results
		np.arr of classifications results.
	shape_of_grid : tuple
		Classificationsmatrix-dimensions.

	Returns
	-------
	C_matrix :  np.array
		Classificationsmatrix.
	C_matrix_new :  np.array
		Classificationsmatrix.
	shape_of_grid :  tuple
		Classificationsmatrix-dimensions.
	detected : list of detected classes

	'''
	# create matrix 
	C_matrix = C_arr.reshape(shape_of_grid)
	# identify detected classes
	detected = list(np.unique(C_matrix.astype('int64')))
	# make sure entries in C are in np.arange(max(C))
	C_matrix_new = cmatrix_transform(C_matrix, detected)
	return C_matrix, C_matrix_new, shape_of_grid, detected


# GSD to altcorrection
def get_altcorrection(GSD):
	'''
	GSD to altcorrection parameter

	Parameters
	----------
	GSD : float
		Ground spatial distance of imagery.

	Returns
	-------
	alt_correction : float
		parameter that will correct waste assessments in respect to the GSD.

	'''
	norm_GSD = 0.2 # cm/px 
	alt_correction = GSD**2 / norm_GSD**2
	return alt_correction


def classify_image_PLD(model_name, image_size, image_name, file_format, cut_im_sect = None):
	'''
	

	Parameters
	----------
	model_name : str
	image_size : int
	image_name : str
	file_format : str
		JPG, PNG or TIF.
	cut_im_sect : list with 4 floats 0<x<1, optional
		DESCRIPTION. The default is None.

	Returns
	-------
	C_matrix :  np.array
		Classificationsmatrix.
	C_matrix_new :  np.array
		Classificationsmatrix.
	shape_of_grid :  tuple
		Classificationsmatrix-dimensions.
	detected : list of detected classes

	'''
	
	# cut a image into a grid and converts it as input
	X, shape_of_grid = imageslicer_modelinput(image_name, image_size, file_format, cut_im_sect)	
	if type(X) == tuple:
		predictions = model_name.predict(X[0][(np.where(X[1] == 1))])
		classifications = np.zeros(shape_of_grid[0]*shape_of_grid[1])
		classifications[np.where(X[1] == 0)] = len(predictions[0])
		classifications[np.where(X[1] == 1)] = np.argmax(predictions, axis=1)
		
	if type(X) == np.ndarray:
		# probabilities for classes
		predictions = model_name.predict(X)
		# turn maxlikelihood of each row
		classifications = np.argmax(predictions, axis=1)
	
	return classify_im_postprocess(classifications, shape_of_grid)


def classify_image_PLQ(model_name, image_size, image_name, C_PLD, file_format, cut_im_sect = None, image_size_PLD = None):
	'''
	classification of polluted areas of input image of PLD CNN

	Parameters
	----------
	model_name : str
	image_size : int
	image_name : str

	C_PLD : np.arr
		Classificationsmatrix of PLD.
	file_format : str
		JPG, PNG or TIF.
	cut_im_sect : list with 4 floats 0<x<1, optional
		DESCRIPTION. The default is None.
	image_size_PLD : int, optional
		image size of PLD. The default is None.

	Returns
	-------
	C_matrix :  np.array
		Classificationsmatrix.
	C_matrix_new :  np.array
		Classificationsmatrix.
	shape_of_grid :  tuple
		Classificationsmatrix-dimensions.
	detected : list of detected classes

	'''
	# cut a image into a grid and converts it as input
	X, shape_of_grid = imageslicer_modelinput(image_name, image_size, file_format, cut_im_sect, image_size_PLD)
	# only compute polluted areas
	C_PLD_polluted = polluted_area_helper(C_PLD, shape_of_grid)#, X)

	predictions = []
	if np.where(C_PLD_polluted==1)[0] != []:
		if type(X) == tuple: # (X, alpha)
			predictions = model_name.predict(X[0][(np.where(C_PLD_polluted==1))])
	
		if type(X) == np.ndarray: # X
			predictions = model_name.predict(X[(np.where(C_PLD_polluted==1))])
	
	# polluted parts on last indice & insert classifications to Scaled array
	output_dim = model_name.compute_output_shape((1,image_size,image_size,3))
	C_PLD_polluted[np.where(C_PLD_polluted == -1)] = output_dim[1]
	if predictions != []:
		classifications = np.argmax(predictions, axis=1)
		C_PLD_polluted[np.where(C_PLD_polluted == 1)] = classifications
	elif predictions != []:
		C_PLD_polluted[np.where(C_PLD_polluted == 1)] = output_dim[1]
	return classify_im_postprocess(C_PLD_polluted, shape_of_grid)


# axis functions
def ax_show_image(ax, image_name, file_format, title = 'Input image', cut_im_sect = None):
	'''
	subplot of image

	Parameters
	----------
	ax : subplot
	image_name : str
		original image.
	file_format : str
		JPG, PNG or TIFF.
	title : str, optional
	cut_im_sect : list with 4 floats 0<x<1, optional
		DESCRIPTION. The default is None.

	Returns
	-------
	None.

	'''
	if file_format in ['JPG', 'PNG']:
		image = plt.imread(image_name)
	
	if file_format in ['TIF']:
		image = io.imread(image_name)
	
	image = cut_im_to_sections(image, cut_im_sect)
	
	ax.imshow(image)
	ax.axis('off')
	ax.set_title(title, fontweight='bold')
	
def ax_white_out(ax):
	'''
	ehite out subplot

	Parameters
	----------
	ax : subplot


	Returns
	-------
	None.

	'''
	ax.imshow(np.ones((1,1)), cmap='gray',vmin=0,vmax=1)
	ax.axis('off')

def ax_c_matrix(ax, ax_im, fig, C, colors, detected_classes, labels,
				add_ax_pos, title, axis_off = False, cbar_B = True):
	## ashow results of 100CNN
	colors = [colors[i] for i in detected_classes]
	cMap = ListedColormap(colors)
	heatmap = ax.pcolor(C, cmap=cMap)
		
	# colorbar stuff
	if cbar_B == True:
		cbar_ax = fig.add_axes(add_ax_pos)
		cbar = plt.colorbar(heatmap, cax=cbar_ax)
		cbar.ax.get_yaxis().set_ticks([])
		for j, lab in enumerate([labels[i] for i in detected_classes]):
			cbar.ax.text(len(detected_classes)+1,# j,
				     ( (len(detected_classes)-1) * j+len(detected_classes)/2) / (len(detected_classes)),
					 lab, rotation=0, fontsize = 9)
		cbar.ax.get_yaxis().labelpad = 15		
	ax.invert_yaxis()
	
	# that the classification plot isn't oversized...
	asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
	asp /= np.abs(np.diff(ax_im.get_xlim())[0] / np.diff(ax_im.get_ylim())[0])
	ax.set_aspect(abs(asp))
	ax.set_title(title, fontweight='bold')
	if axis_off == True: ax.axis('off')

def ax_table_PLD(ax, labels, C, GSD, table_pos, title,
				ind_litter_h_l, param_litter_h_l):
	
	collabel=tuple(labels+['Litter abundances            ',
						'Litter m²',
						'Litter m³'])
	table_data = np.zeros((len(labels)+5,1), dtype='int64')
	alt_correction = get_altcorrection(GSD)
	
	for i in range(len(labels)):
		table_data[i]  = np.sum(C==i)
	
	litter_h, litter_l = table_data[ind_litter_h_l[0]], table_data[ind_litter_h_l[1]]
	area_param = (128*GSD*0.01)**2  # convert to m²
	table_data[-5] = (litter_h*param_litter_h_l[0] + litter_l*param_litter_h_l[1])*alt_correction
	table_data[-4] = litter_h*area_param + litter_l*area_param*0.5 # area
	table_data[-3] = litter_h*area_param*0.30 + litter_l*area_param*0.07 # volume
	ax.table(table_data, rowLabels=collabel, loc='center', bbox = table_pos) # [left, bottom, width, height]
	ax.axis('off')
	ax.set_title('Assessed classifications, \n abundances, areas and volumes' + title, fontweight='bold')
	return table_data

def ax_table_PLD_org_debris(ax, labels, C, GSD, table_pos, title,
				ind_litter_h_l, param_litter_h_l):
	
	collabel=tuple(labels+['Litter abundances            ',
						'Litter m²',
						'Litter m³',
						'Org. Debris m²',
						'Org. Debris m³'])
	table_data = np.zeros((len(labels)+5,1), dtype='int64')
	alt_correction = get_altcorrection(GSD)
	
	for i in range(len(labels)):
		table_data[i]  = np.sum(C==i)
	
	litter_h, litter_l = table_data[ind_litter_h_l[0]], table_data[ind_litter_h_l[1]]
	area_param = (128*GSD*0.01)**2  # convert to m², "*0.01" because GSD is in cm
	table_data[-5] = (litter_h*param_litter_h_l[0] + litter_l*param_litter_h_l[1])*alt_correction
	table_data[-4] = litter_h*area_param + litter_l*area_param*0.5 # area
	table_data[-3] = litter_h*area_param*0.30 + litter_l*area_param*0.07 # volume
	table_data[-2] = table_data[2]*area_param # area
	table_data[-1] = table_data[2]*area_param*0.1 # volume

	ax.table(table_data, rowLabels=collabel, loc='center', bbox = table_pos) # [left, bottom, width, height]
	ax.axis('off')
	ax.set_title('Assessed classifications, \n abundances, areas and volumes' + title, fontweight='bold')
	return table_data

def ax_table_PLQ(ax, labels, C, GSD, params_PLQ, 
				 table_pos, title):
	
	alt_correction = get_altcorrection(GSD)
	labels_new = [label[4:] for label in labels]
	collabel=tuple(labels_new)
	table_data = np.zeros((len(labels),2), dtype='int64')

	for i in range(len(labels)):
		class_sum = int(np.sum(C==i)*alt_correction)
		table_data[i]  = class_sum, class_sum*params_PLQ[i]

	ax.table(table_data, rowLabels=collabel, colLabels=('Classifications', 'Assessed abundances'),
		  loc='center', bbox = table_pos) # [left, bottom, width, height]
	ax.axis('off')
	ax.set_title('Altitude corrected assessed areas\n and pollution type abundances ' + title, fontweight='bold')
	return table_data

def ax_pie_PLQ(ax, data_PLQ, labels_PLQ, colors_PLQ, NW_classes):

	treshhold = np.where(data_PLQ.flatten()[:-NW_classes]>(sum(data_PLQ.flatten())*0.01))
	list_labels_PLQ_prep = []
	list_colors_PLQ_prep = []
	for ind in treshhold[0]:
		list_labels_PLQ_prep.append(labels_PLQ[:-NW_classes][ind])
		list_colors_PLQ_prep.append(colors_PLQ[:-NW_classes][ind])

	ax.pie(data_PLQ.flatten()[:-4][treshhold], labels = [label[4:] for label in list_labels_PLQ_prep],
                                          autopct ='%.0f%%',
										  colors = list_colors_PLQ_prep,
                                          textprops = {'size': 'smaller'},
                                          shadow = False, radius= 0.9,
										  normalize = True,)
	ax.set_title('Share of waste types', fontweight='bold')
	
def fig_save(fig, path_to_save_figs, save_dir_name, scene_name, dpi):
	'''
	Parameters
	----------
	fig : matplotlib fig
	path_to_save_figs : str
		path to saving result report.
	save_dir_name : str
		directory where result report will be saved.
	scene_name : str
	dpi : int
		dpi of result report.

	Returns
	-------
	None.

	'''
	# saving figure
	fig_dir = os.path.join(path_to_save_figs, save_dir_name)
	if not os.path.exists(fig_dir):
		os.mkdir(fig_dir)
	fig.savefig(os.path.join(fig_dir, scene_name), dpi=dpi)
	plt.close(fig)


def aplasticq_waste_assessment(plastic_model_PLD,
						 plastic_model_PLQ,
						 labels_PLD,
						 labels_PLQ,
						 image_name,
						 path_to_save_figs,
						 colors_PLD,
						 colors_PLQ,
						 number_of_scence,
						 save_dir_name,
						 params_PLQ,
						 GSD = 0.2, # cm/px
						 dense_few = [3.5, 1,5],
						 litter_h_litter_l = [0,1],
						 file_format  = 'JPG',
						 dpi = 300,
						 cut_im_sect = None,
						 PARAMS_save_Bool = True,
						 watermark_path = False,
						 title_name = None,
						 ORG_DEBRIS_CONSIDER = False):
	
	# perform double classification with timing.
	t0 = time.time()
	print()
	C_PLD, C_PLD_im, grid_shape_PLD, detected_PLD = classify_image_PLD(plastic_model_PLD,
													128, image_name, file_format, cut_im_sect)
	t1 = time.time()
	print('Time elapsed to classify with PLD CNN :  ', t1-t0, ' secs')

	C_PLQ, C_PLQ_im, _, detected_PLQ = classify_image_PLQ(plastic_model_PLQ, 64,
													image_name, C_PLD, file_format,
													cut_im_sect, 128)
	t2 = time.time()	
	print('Time elapsed to classify with PLQ CNN :  ', t2-t1, ' secs')

	# prep PLD & PLQ C results
	labels_PLD_new, colors_PLD = prep_C_PLD(labels_PLD, colors_PLD, file_format)
	labels_PLQ_new, colors_PLQ, C_PLQ, C_PLQ_im = prep_C_PLQ(labels_PLQ, colors_PLQ, C_PLQ, C_PLQ_im, grid_shape_PLD)

	# fig init & titles
	fig, axs = plt.subplots(2,3, figsize=(16,9))#, constrained_layout=True)
	scene_name = os.path.basename(image_name)[:-4]	# When computing normal ims: [5:-4]
	if title_name == None:
		fig.suptitle('Scene: '+ scene_name, fontsize=20)
	if type(title_name) == str:
		fig.suptitle('Scene: '+ title_name, fontsize=20)
	fig.subplots_adjust(right=0.750)

	# axs[0,0] 
	ax_show_image(axs[0,0], image_name, file_format, cut_im_sect = cut_im_sect)

	# axs[0,1] C_PLD matrix
	ax_c_matrix(axs[0,1], axs[0,0], fig, C_PLD_im, colors_PLD, detected_PLD,
			   labels_PLD_new, [0.81,  0.60+0.075, 0.02, 0.20], 'PLD CNN')

	# axs[1,1]C_PLQ matrix
	labels_PLQ_no_code = [label[4:] for label in labels_PLQ_new]
	ax_c_matrix(axs[1,1], axs[0,0], fig, C_PLQ_im, colors_PLQ, detected_PLQ,
			   labels_PLQ_no_code, [0.81,  0.05+0.075, 0.02, 0.35], 'PLQ CNN')
	
	
	# axs[0,2] C_PLD table
	if ORG_DEBRIS_CONSIDER:
		data_PLD = ax_table_PLD_org_debris(axs[0,2], labels_PLD, C_PLD, GSD,
					   [0.7, 0.2, 0.6, 0.8], '', litter_h_litter_l, dense_few,)
	if not ORG_DEBRIS_CONSIDER:
		data_PLD = ax_table_PLD(axs[0,2], labels_PLD, C_PLD, GSD,
						   [0.7, 0.2, 0.6, 0.8], '', litter_h_litter_l, dense_few)
	
	# axs[1,2] C_PLQ table
	data_PLQ = ax_table_PLQ(axs[1,2], labels_PLQ, C_PLQ, GSD,
						 params_PLQ, [.70, -.2, .77,1.2], '')# [left, bottom, width, height]
	
	# axs[1,0] C_PLQ pie               
	ax_pie_PLQ(axs[1,0], data_PLQ[:,1], labels_PLQ, colors_PLQ, 4)
	
	# save fig
	fig_save(fig, path_to_save_figs, save_dir_name, scene_name, dpi)
	
	t3 = time.time()
	
	if watermark_path != False:
		watermark_png(os.path.join(path_to_save_figs, save_dir_name, scene_name + '.png'), watermark_path , 0.14)

	print('Time elapsed to create and save plot :  ', t3-t2, ' secs')

	if PARAMS_save_Bool:
		def_save_PARAMS(plastic_model_PLD, plastic_model_PLQ, labels_PLD, labels_PLQ,
				  image_name, path_to_save_figs, colors_PLD, colors_PLQ,
				  number_of_scence, save_dir_name, params_PLQ,
				  GSD, dense_few, litter_h_litter_l,
				  file_format, dpi, cut_im_sect)
	
	return data_PLD.flatten(), data_PLQ[:,0].flatten()
