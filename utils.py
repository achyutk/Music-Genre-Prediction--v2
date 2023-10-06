#Importing necessary libraries
import numpy as np
import pandas as pd
%matplotlib inline
import os
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')


#Data Exploration
def explore(folder):
  general_path = '/content/drive/MyDrive/archive/Data'
  i = list(os.listdir(f'{general_path}/{folder}/'))[0]
  j = list(os.listdir(f'{general_path}/{folder}/{i}'))[0]
  y, sr = librosa.load(f'{general_path}/{folder}/{i}/{j}')

  print('y:', y)
  print('y shape:', np.shape(y))
  print('Sample Rate (KHz):', sr)

  # Verify length of the audio
  print('Check Len of Audio:', len(y)/sr)

  zero_crossings = librosa.zero_crossings(y, pad=False)
  print('zero_crossings: ',sum(zero_crossings))

  y_harm, y_perc = librosa.effects.hpss(y)
  print('y_harm: ',y_harm.mean())
  print('y_perc: ',y_perc.mean())

  tempo, _ = librosa.beat.beat_track(y=y,sr=sr)
  print('tempo:', tempo)

  spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
  print('spectral_centroids mean: ',spectral_centroids.mean())
  print('spectral_centroids var: ',spectral_centroids.var())

  spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
  print('roll_off mean: ',spectral_rolloff.mean())
  print('roll_off var: ',spectral_rolloff.var())


#Function to extract feature from an audio file using librosa library
def get_features(folder,mfcc,n_fft = 128,hop_length = 256):
  labels=[] #List to store labels
  features=[]   #List to store features
  general_path = '/content/drive/MyDrive/archive/Data'
  music_type = list(os.listdir(f'{general_path}/{folder}/'))    #Creating list of genres
  
  #iterating over music genres
  for i in music_type:
    files= list(os.listdir(f'{general_path}/{folder}/{i}')) #Getting list of files in the directory
    #iterating over files
    for j in files:
      try:
        y, sr = librosa.load(f'{general_path}/{folder}/{i}/{j}')    #Loading audio file
        data = np.array(librosa.feature.mfcc(y=y, n_fft=n_fft,hop_length=hop_length,n_mfcc=mfcc))   #Extracting mfcc features and converting it to numpy array
        
        # print(i," ",j," ",data.shape)
        if data.shape[1]==259:
          
          #Transposing and reshaping data to form time series
          data=np.transpose(data)   
          data= np.reshape(data,(1,data.shape[0],data.shape[1]))    
          features.append(data) #Appending data to the lists
          labels.append(i)
        # else:
        #   print("Didn't fill: ",j, " ",data.shape)
      except:
        print("Error: ",j)        
  output=np.concatenate(features,axis=0)  
  return(np.array(output), labels)

#Function to normaise the array my mean and SD
def normalisation(features):
  features = np.array((features-np.mean(features)))
  features = features/np.std(features)
  return features

#Factorizing labels
def factorized_lable(labels):
  a = pd.DataFrame(labels,columns=['Hello'])
  a.Hello = pd.factorize(a.Hello)[0]
  y2=list(a.Hello)
  y = np.array(y2)
  return y

#Function to visualise results
def graph(history):

  history_dict=history.history
  loss_values=history_dict['loss']
  acc_values=history_dict['acc']
  val_loss_values = history_dict['val_loss']
  val_acc_values=history_dict['val_acc']
  epochs=range(1,51)
  fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
  ax1.plot(epochs,loss_values,'co',label='Training Loss')
  ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
  ax1.set_title('Training and validation loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.legend()
  ax2.plot(epochs,acc_values,'co', label='Training accuracy')
  ax2.plot(epochs,val_acc_values,'m',label='Validation accuracy')
  ax2.set_title('Training and validation accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.legend()
  plt.show()