import os
import numpy as np
from helperfunctions import load_model_labels
from helperfunctions import color_list_PLD, color_list_PLQ, params_PLQ, save_data_in_excel
from prepro_subplots import aplasticq_waste_assessment

wd = os.getcwd()
os.chdir(wd)

# Path definitions
dir_ims            = 'input_data'
dir_dir_out_sheets = 'output_sheets'
dir_ml_models      = 'ml_models'
models_names = [
				'PLD_CNN.h5',
				'PLQ_CNN.h5',
				]

model_ind_PLD = 0
model_ind_PLQ = 1

# load two plastic models along with label lists
plastic_model_PLD, labels_PLD = load_model_labels(os.path.join(wd, dir_ml_models, models_names[model_ind_PLD]))
plastic_model_PLQ, labels_PLQ = load_model_labels(os.path.join(wd, dir_ml_models, models_names[model_ind_PLQ]))

# performs aplastic-q waste assessment on imagefiles
image_names = os.listdir(os.path.join(wd, dir_ims))

# initializes result arrays
res_PLD = np.zeros((len(image_names), len(labels_PLD)+5), dtype='int64')
res_PLQ = np.zeros((len(image_names), len(labels_PLQ)), dtype='int64')

# run plastic waste detection on input_data
for i in range(len(image_names)):
	res_PLD[i], res_PLQ[i] = aplasticq_waste_assessment(plastic_model_PLD = plastic_model_PLD,
											  plastic_model_PLQ = plastic_model_PLQ,
						 labels_PLD = labels_PLD,
						 labels_PLQ = labels_PLQ,
						 image_name = os.path.join(wd, dir_ims, image_names[i]),
						 path_to_save_figs = os.path.join(wd, dir_dir_out_sheets),
						 colors_PLD = color_list_PLD,
						 colors_PLQ  = color_list_PLQ,
						 number_of_scence = i,
						 save_dir_name= dir_ims,
						 params_PLQ = params_PLQ,
						 GSD = 0.2, #in cm
						 dense_few = [5.0, 2,5],
						 litter_h_litter_l = [0,1],
						 file_format  = 'JPG',
						 dpi = 300,
						 cut_im_sect = None, 
						 PARAMS_save_Bool = True,
						 watermark_path = os.path.join(wd, 'DFKI_logo.png'),
						 ORG_DEBRIS_CONSIDER = True,
						 )
	
save_data_in_excel(os.path.join(wd,dir_dir_out_sheets, dir_ims, 'Result_sheets'),dir_ims+'.xlsx',
				   res_PLD, res_PLQ, labels_PLD, labels_PLQ, image_names)

