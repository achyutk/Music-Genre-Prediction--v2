#Importing necessary libraries
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import model
import utils
import torch


#Hyperparameter for feature extraction
hop_length = 256 #the default spacing between frames
n_fft = 128 #number of samples 
mfcc= 20    #Change it to 128 for more number of features


# Feature Extraction of Augmented Audio
features_main,lable_main = utils.get_features('3sec',mfcc,n_fft = 128,hop_length = 256)
lable_main = utils.factorized_lable(lable_main)
features_main = utils.normalisation(features_main)


# Split twice to get the validation set
X_train, X_test, y_train, y_test = train_test_split(features_main, lable_main, test_size=0.1, random_state=123, stratify=lable_main)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

#Print the shapes of the dataset
print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))



model = model.lstmmodel() #Defining Model
model,history_model = model.training(X_train, y_train, X_val, y_val,model)  #Training the model


# Save the entire model
torch.save(model, 'model/model.pth')