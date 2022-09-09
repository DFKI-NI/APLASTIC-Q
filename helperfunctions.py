import pandas as pd
import os
import pickle
from keras.models import load_model

# load plastic model
def load_model_labels(path_to_model):
	'''
	loads ML models with corresponding labels

	Parameters
	----------
	path_to_model : str

	Returns
	-------
	plastic_model : keras model
	labels : list with str
	'''
	plastic_model = load_model(path_to_model)
	with open(path_to_model+ 'labels.txt', "rb") as fp:
		labels = pickle.load(fp)
	return plastic_model, labels

# save data in Exccel
def save_data_in_excel(dir_exel_file,
					   excel_file_name,
					   res_PLD,
					   res_PLQ,
					   labels_PLD,
					   labels_PLQ,
					   image_names):
	'''
	saves results of PLD and PLQ in an excel file
	
	Parameters
	----------
	dir_exel_file : str
		dir for saving excel result file.
	excel_file_name : str
		name of excel file.
	res_PLD: np.array
		results of PLD CNN.
	res_PLQ : np.array
		results of PLQ CNN.
	labels_PLD : list with str
	labels_PLQ : list with str
	image_names : list with str 
		element i is the i-th image.

	Returns
	-------
	saves excel file

	'''
	# saving labels in the left column of the excel file
	dir_mk_if_absent(dir_exel_file)
	writer = pd.ExcelWriter(os.path.join(dir_exel_file,excel_file_name), engine='xlsxwriter')
	df1 = pd.DataFrame({'labels_PLD': labels_PLD + ['Est. litter abundances', 'Area est.', 'Volume est',
						'Org. Debris m² est.',
						'Org. Debris m³ est.']})
	df2 = pd.DataFrame({'labels_PLQ': labels_PLQ})
	df1.to_excel(writer, sheet_name='Classification results', startcol=0,
			  startrow = 0, index=False)
	df2.to_excel(writer, sheet_name='Classification results', startcol=0,
			  startrow = 17, index=False)
	
	# saving the results
	for i in range(len(image_names)):
		df3 = pd.DataFrame({image_names[i][:-4]: res_PLD[i]})
		df4 = pd.DataFrame({image_names[i][:-4]: res_PLQ[i]})
		df3.to_excel(writer, sheet_name='Classification results',startcol=i+1,
			   startrow = 0, index=False)
		df4.to_excel(writer, sheet_name='Classification results', startcol=i+1,
			   header=False, startrow = 18, index=False)
		
	writer.save()
	return print('Classifications saved in: ', os.path.join(dir_exel_file,excel_file_name))


# lists for visualization

# PLD colors
color_list_PLD = ['0.65', '0.85', 'saddlebrown', 'aquamarine', 'peru', 'slategrey', 'green', 'blue'] 

# PLQ colors
color_list_PLQ = ['lightsteelblue',
					 'lavender',
					 'mediumpurple',
					 'deeppink',
					 'red',
					 'thistle',      # 5
					 'yellow',
					 'oldlace',
					 'violet',
					 'mediumorchid',
					 'darkviolet',   # 10
					 'slategrey',
					 'cyan',
					 'darkslategrey',
					 'gainsboro',
					 'silver',       # 15
					 'grey',
					 'dimgrey',
					 'peru',
					 'green',
					 'saddlebrown',  # 20
					 'blue',
					 'aquamarine',
					 'white']


# labels for PLD / PLQ classes

labels_PLD = ['Litter - high', 'Litter - low', 'Organic debris', 'Other', 'Sand',
				 'Stones', 'Vegetation', 'Water']

labels_PLQ = ['P - bags LPDE thick',
			  'P - bags LPDE',
			  'P - bags robust PET',
			  'P - wrappers under 10cm',
			  'P - wrappers over 10cm', #5
			  'P - bottles PET',
			  'P - polystyrene under 20cm',
			  'P - polystyrene over 20cm',
			  'P - PPCP bottle',
			  'P - PPCP medical waste', #10
			  'P - PPCP other',
			  'P - fishing gear',
			  'P - cup lids, caps and small plastics',
			  'P - other plastics over 20cm',
			  'NP - rubber',            #15
			  'NP - metal',
			  'NP - glass',
			  'NP - other',
			  'NW - sand',
			  'NW - vegetation',        #20
			  'NW - wood',
			  'NW - water',
			  'NW - other']

# class parameters for PLQ for size correction 
params_PLQ = [0.5,#'P - bags LPDE thick'
				 1.0,#'P - bags LPDE'
				 0.5,#'P - bags robust PET'
				 2.0,#'P - wrappers under 10cm'
				 1.0,#'P - wrappers over 10cm'
				 1.0,#'P - bottles PET'
				 1.0,#'P - polystyrene under 20cm'
				 0.3,#'P - polystyrene over 20cm'
				 2.0,#'P - PPCP bottle'
				 1.0,#'P - PPCP medical waste'
				 1.0,#'P - PPCP other'
				 0.5,#'P - fishing gear'
				 2.0,#'P - cup lids, caps and small plastics'
				 0.7,#'P - other plastics over 20cm'
				 0.5,#'NP - rubber'
				 1.0,#'NP - metal'
				 1.0,#'NP - glass'
				 1.0,#'NP - other'
				 0.0,#'NW - sand'
				 0.0,#'NW - vegetation'
				 0.0,#'NW - wood'
				 0.0,#'NW - water'
				 0.0,#'NW - other'
				 ]

# matplotlib params like font, fontsizes 
matplotlib_params = {'legend.fontsize': '13',
          'figure.figsize': (9,5),
         'axes.labelsize': '10',
         'axes.titlesize':'14',
         'xtick.labelsize':'11',
         'ytick.labelsize':'11',
		 'font.sans-serif':'Arial',
		 'font.size': 9.0,
		 'mathtext.fontset': 'dejavusans'
		 }

matplotlib_params_PLQ_CM = {'legend.fontsize': '6',
          'figure.figsize': (9,5),
         'axes.labelsize': '10',
         'axes.titlesize':'14',
         'xtick.labelsize':'8',
         'ytick.labelsize':'8',
		 'font.sans-serif':'Arial',
		 'font.size': 6.0,
		 'mathtext.fontset': 'dejavusans'
		 }

# defs for datamerging
def intersection(lst1, lst2): 
    return [value for value in lst1 if value in lst2] 

def a_and_not_b(lsta, lstb):
	lstres = [value for value in lsta if value not in lstb]
	return lstres, len(lsta) - len(lstres)
	
def dir_mk_if_absent(path_dir):
	if not os.path.isdir(path_dir):
		os.makedirs(path_dir)
