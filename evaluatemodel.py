from nibabel import load as load_nii
from metrics import *
import sklearn.metrics as skmet
from scipy import ndimage as nd
import os
import matplotlib.pyplot as plt

test_folder = '/scratch/mythri.v/MIA_project/test_data/'

filename2='CHB_train_Case09'

filename4='UNC_train_Case09'
test_x_data = {}
test_x_data[filename2] = {'T1': os.path.join(test_folder,'CHB_train_Case09',filename2+ '_T1_brain.nii.gz'), 
                           'FLAIR': os.path.join(test_folder,'CHB_train_Case09',filename2+'_FLAIR_brain.nii.gz')}
test_x_data[filename4] = {'T1': os.path.join(test_folder,'UNC_train_Case09',filename4+ '_T1_brain.nii.gz'), 
                        'FLAIR': os.path.join(test_folder,'UNC_train_Case09',filename4+ '_FLAIR_brain.nii.gz')}

og2_t1  = load_nii(test_x_data[filename2]['T1']).get_data()
og2_flair = load_nii(test_x_data[filename2]['FLAIR']).get_data()
og4_t1  = load_nii(test_x_data[filename4]['T1']).get_data()
og4_flair = load_nii(test_x_data[filename4]['FLAIR']).get_data()

crop2_t1 = nd.zoom(og2_t1,[0.25,0.25,0.25])[24:104,24:104,24:104]
crop4_t1 = nd.zoom(og2_t1,[0.25,0.25,0.25])[24:104,24:104,24:104]

#prediction
obs2 = load_nii(os.path.join(test_folder,filename2,'test_CNN','test_CNN_out_CNN.nii.gz')).get_data()
obs4 = load_nii(os.path.join(test_folder,filename4,'test_CNN','test_CNN_out_CNN.nii.gz')).get_data()

#ground truth
gt2 = load_nii(os.path.join(test_folder,filename2,filename2+'_lesion.nii.gz'))
gt4 = load_nii(os.path.join(test_folder,filename4,filename4+'_lesion.nii.gz'))
gt2np = nd.zoom(gt2.get_data(),[0.25,0.25,0.25])[24:104,24:104,24:104]
gt4np = nd.zoom(gt4.get_data(),[0.25,0.25,0.25])[24:104,24:104,24:104]

plt.figure()
plt.subplot(2,2,1)
plt.imshow(crop2_t1,'gray')
plt.imshow(gt2np,'jet',alpha=0.5)
plt.title('CHB Ground Truth')
plt.subplot(2,2,2)
plt.imshow(crop2_t1,'gray')
plt.imshow(obs2,'jet',alpha=0.5)
plt.title('CHB Prediction')
plt.subplot(2,2,3)
plt.imshow(crop4_t1,'gray')
plt.imshow(gt4np,'jet',alpha=0.5)
plt.title('UNC Ground Truth')
plt.subplot(2,2,4)
plt.imshow(crop4_t1,'gray')
plt.imshow(obs4,'jet',alpha=0.5)
plt.title('UNC Prediction')
plt.show()


print 'CHB'
print 'vd:',vol_dif(obs2,gt2np)
print 'tpr:',skmet.precision_score(gt2np.flatten(),obs2.flatten())
tn, fp, fn, tp = skmet.confusion_matrix(gt2np.flatten(),obs2.flatten()).ravel()
fpr = fp /(fp + tp)
print 'fpr:',fpr

print 'UNC'
print 'vd:',vol_dif(obs4,gt4np)
print 'tpr:',skmet.precision_score(gt4np.flatten(),obs4.flatten())
tn, fp, fn, tp = skmet.confusion_matrix(gt4np.flatten(),obs4.flatten()).ravel()
fpr = fp /(fp + tp)
print 'fpr:',fpr

