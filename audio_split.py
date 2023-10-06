#Importing necessary libraries
%matplotlib inline
import os
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf

def divide(folder,output):
  general_path = '/content/drive/MyDrive/archive/Data'
  music_type = list(os.listdir(f'{general_path}/{folder}/')) 

  
  for i in music_type:
    files= list(os.listdir(f'{general_path}/{folder}/{i}'))
    for j in files:
      try: 
        y, sr = librosa.load(f'{general_path}/{folder}/{i}/{j}')
        y, _ = librosa.effects.trim(y)

        for k in range(0,len(y),(sr*3)):
          l= k+(sr*3)
          temp_audio = y[k:l]
          sf.write('/content/drive/MyDrive/archive/Data/'+output +'/'+i+ '/' +j.rstrip('.wav')+ "." +str(k)+ '.wav', temp_audio, sr)
          print('file_written: ',i,j)
      except:
        print('Skipped',i,j)

divide('genres_original','3sec')
# divide('Augmented','Aug3sec')