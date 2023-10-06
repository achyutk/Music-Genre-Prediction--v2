# GAN implementation for audio generation

The scripts in this folder are used to generate augmented audio files using GAN( Genrative Adversarial Networks).

# Installations

Clone repo and install the following libraries:

> pip install torch torchvision torchaudio <br>
> pip install sklearn


Download the [genres_original dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and paste in the *data* folder of this repository

Run implemeentation.ipynb file. This file will generate audio for the selected genre. It will fetch the clips from main data set. This file will generate two folders, 
 - Model: This folder stores the GAN model that were trained and then used to generate audio files <br>
 - Ouputs: this folder stores audio files generated after every epochs


