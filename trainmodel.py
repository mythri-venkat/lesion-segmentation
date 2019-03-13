import os
from collections import OrderedDict
from base import *
from build_model import cascade_model
from config import *

options = {}

# --------------------------------------------------
# Experiment parameters
# --------------------------------------------------

# image modalities used (T1, FLAIR, PD, T2, ...) 
options['modalities'] = ['T1', 'FLAIR']

# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'test_CNN'

# In order to expose the classifier to more challeging samples, a threshold can be used to to select 
# candidate voxels for training. Note that images are internally normalized to 0 mean 1 standard deviation 
# before applying thresholding. So a value of t > 0.5 on FLAIR is reasonable in most cases to extract 
# all WM lesion candidates
options['min_th'] = 0.5

# randomize training features before fitting the model.  
options['randomize_train'] = True

# Select between pixel-wise or fully-convolutional training models. Although implemented, fully-convolutional
# models have been not tested with this cascaded model 
options['fully_convolutional'] = False


# --------------------------------------------------
# model parameters
# --------------------------------------------------

# 3D patch size. So, far only implemented for 3D CNN models. 
options['patch_size'] = (11,11,11)

# percentage of the training vector that is going to be used to validate the model during training
options['train_split'] = 0.25

# maximum number of epochs used to train the model
options['max_epochs'] = 200

# maximum number of epochs without improving validation before stopping training (early stopping) 
options['patience'] = 25

# Number of samples used to test at once. This parameter should be around 50000 for machines
# with less than 32GB of RAM
options['batch_size'] = 5000

# net print verbosity. Set to zero for this particular notebook, but setting this value to 11 is recommended
options['net_verbose'] = 11

# post-processing binary threshold. After segmentation, probabilistic masks are binarized using a defined threshold.
options['t_bin'] = 0.8

# The resulting binary mask is filtered by removing lesion regions with lesion size before a defined value
options['l_min'] = 20

options['load_weights']='False'

exp_folder = os.path.join(os.getcwd(), options['experiment'])
if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)
    os.mkdir(os.path.join(exp_folder,'nets'))
    os.mkdir(os.path.join(exp_folder,'.train'))

# set the output name 
options['test_name'] = 'cnn_' + options['experiment'] + '.nii.gz'

# load train folders in dictionary
train_folder = '/scratch/mythri.v/MIA_project/train_data'
train_x_data = {}
train_y_data = {}

filename1='CHB_train_Case0'
filename2='CHB_train_Case10'
filename3='UNC_train_Case0'
filename4='UNC_train_Case10'
for i in range(1,9):
    f1=filename1+str(i)
    f2=filename3+str(i)
    train_x_data[filename1+str(i)] = {'T1': os.path.join(train_folder,f1+ '_T1_brain.nii.gz'), 
                           'FLAIR': os.path.join(train_folder,f1+'_FLAIR_brain.nii.gz')}
    train_x_data[filename3+str(i)] = {'T1': os.path.join(train_folder,f2+ '_T1_brain.nii.gz'), 
                           'FLAIR': os.path.join(train_folder,f2+ '_FLAIR_brain.nii.gz')}

    # TRAIN LABELS 
    train_y_data[f1] = os.path.join(train_folder,f1+ '_lesion.nii.gz')
    train_y_data[f2] = os.path.join(train_folder,f2+ '_lesion.nii.gz')

train_x_data[filename2] = {'T1': os.path.join(train_folder,filename2+ '_T1_brain.nii.gz'), 
                           'FLAIR': os.path.join(train_folder,filename2+'_FLAIR_brain.nii.gz')}
train_x_data[filename4] = {'T1': os.path.join(train_folder,filename4+ '_T1_brain.nii.gz'), 
                        'FLAIR': os.path.join(train_folder,filename4+ '_FLAIR_brain.nii.gz')}

# TRAIN LABELS 
train_y_data[filename2] = os.path.join(train_folder,filename2+ '_lesion.nii.gz')
train_y_data[filename4] = os.path.join(train_folder,filename4+ '_lesion.nii.gz')

options['weight_paths'] = '/scratch/mythri.v/MIA_project/'

#Train the model
model = cascade_model(options)
model = train_cascaded_model(model, train_x_data, train_y_data,  options)
