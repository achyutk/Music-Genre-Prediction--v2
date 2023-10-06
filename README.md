# Music Genre Prediciton
![img](https://github.com/achyutk/Music-Genre-Prediction/assets/73283117/24ad3028-2d2f-4b9c-bfd7-5063980a9528)


The following repository consists of code to predicting music genre using audio files. The dataset chosen for this project is sourced from this link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

This project is an extended version of the project [Music-Genre-Prediction](https://github.com/achyutk/Music-Genre-Prediction)

This method uses audio files from the sourced website to predict the genre of the music. An LSTM model architecture was tested on two different audio set. 
- One original clips, divided into 3 sec clips. 
- dataset formed by implementing a GAN. 

The accuracy of LSTM model was test on the two dataset.

# Installations and working

Clone repo and install the following libraries:

> pip install torch torchvision torchaudio <br>
> pip install librosa

Download the [genres_original dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and paste in the *data* folder of this repository

- run audio_split.py file, this will create a folder in *data* named *3sec* which will contatin audio-clips split into 3 seconds. 
- train.py file will execute and save the best model achieved which is the LSTM model with 128mfcc features. <br>

To implement the GAN-LSTM model: 
- generate the augmented audio files using the scripts in *augmented_audio_generation* folder. Copy the audio files from the last epoch and save it in the same folder structure as original folder in the *data* file. Keep the name of the parent folder as Augmented or
<br> 
- Download the augented generated files by auther from the following [link](https://drive.google.com/file/d/1Mb23DJJYu7ENUSwH9Up3tlBD1F2bc-Lv/view?usp=sharing) and paste it in the data folder with the name of parent folder as *Augmented*.
- run the audio_split.py file after uncommenting the last line to generate 3sec clip of generated audio files.
- run the demo.ipynb file. This file demonstrates exectution of all the models that were tested vanila LSTm and GAN-LSTM.


# Results

Accuracy achieved on the validation dataset formed from the dataset:

| Model  | Val accuracy    |
 :---: | :---: |
| vanila LSTM :20 mfcc features | 62.26%   |
| vanila LSTM :128 mfcc features | 72.67%   |
| GAN-LSTM :20 mfcc features | 26.80%   |
| GAN-LSTM :128 mfcc features | 13.01%   |

# Further Reading

The report of this project can be found on the following link:
https://drive.google.com/file/d/18xyLXuRlXh7rVW_m7scLAA7ArKv2O-24/view?usp=sharing

