# Import modules
import tensorflow as tf
import numpy as np
import librosa
import random
import os
import soundfile as sf
import model

#Code to load google drive
# from google.colab import drive
# drive.mount('/content/drive')


#Hyperparameters
INSTRUMENT = "rock"   # Change this according to the genre you want to develop
DATA_DIR = "/content/drive/MyDrive/archive/Data/genres_original/"+INSTRUMENT  #Path for directory. Change this according to the genre you want to develop
MODEL_DIMS = 64   #Set the d values in Critic and generator
NUM_SAMPLES = 65536 
Fs = 16000  
NOISE_LEN = 100
# Defining Hyperparameters for Loss Function
GRADIENT_PENALTY_WEIGHT = 10.0
# Defining hyperparameters for training
D_UPDATES_PER_G_UPDATE = 5 # Decides how many time a GAN will update the generator for a particular batch is used for training a 
EPOCHS = 50
EPOCHS_PER_SAMPLE = 2  # To determing when to generate the audio file and save the model. It is generated at every even number of epochs
BATCH_SIZE = 64


# Creating directories
paths = ["/content/drive/MyDrive/Logs/train", 
         f"/content/drive/MyDrive/Model/{INSTRUMENT}/js",   #List for the path
         f"/content/drive/MyDrive/Outputs/{INSTRUMENT}",]
for path in paths:
    if not os.path.exists(os.path.join(os.getcwd(), path)):
        os.makedirs(path)   #Creating directory

# Instantiate model
gan = model.GAN(model_dims=MODEL_DIMS, num_samples=NUM_SAMPLES, 
              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT, instrument=INSTRUMENT,
              noise_len=NOISE_LEN, batch_size=BATCH_SIZE, sr=Fs)


# Create training data
X_train = []
for file in os.listdir(DATA_DIR): ### Modify for your data directory
    with open(DATA_DIR + fr"/{file}", "rb") as f:
        samples, _ = librosa.load(f, sr = Fs)
        # Pad short audio files to NUM_SAMPLES duration
        if len(samples) < NUM_SAMPLES:
            audio = np.array([np.array([sample]) for sample in samples])
            padding = np.zeros(shape=(NUM_SAMPLES - len(samples), 1), dtype='float32')
            X_train.append(np.append(audio, padding, axis=0))
        # Create slices of length NUM_SAMPLES from long audio
        else:
            p = len(samples) // (NUM_SAMPLES)
            for i in range(p - 1):
                sample = np.expand_dims(samples[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES], axis=1)
                X_train.append(sample)

#Visualising training set
print(f"X_train shape = {(len(X_train),) + X_train[0].shape}")

# Save some random training data slices and create baseline generated data for comparison
for i in range(50):  
    sf.write(f"/content/drive/MyDrive/Outputs/{INSTRUMENT}/real-{i}.wav", X_train[random.randint(0, len(X_train) - 1)],samplerate=Fs)

# Save some random data slices as fake for comparison
gan.sample("fake")
train_summary_writer = tf.summary.create_file_writer("logs/train")



# Train GAN
with train_summary_writer.as_default():
    steps_per_epoch = len(X_train) // BATCH_SIZE 

    for e in range(EPOCHS):
        for i in range(steps_per_epoch):
            D_loss_sum = 0
        
            # Update dcritic a set number of times for each update of the generator
            for n in range(D_UPDATES_PER_G_UPDATE):
                gan.D.reset_states()
                D_loss_dict = gan.train_D(np.array(random.sample(X_train, BATCH_SIZE)))
                D_loss_sum += D_loss_dict['d_loss']
            
            # Calculate average loss of critic for current step
            D_loss = D_loss_sum / D_UPDATES_PER_G_UPDATE
            
            G_loss_dict = gan.train_G()
            G_loss = G_loss_dict['g_loss']
        
            # Write logs
            tf.summary.scalar('d_loss', D_loss, step=(e*steps_per_epoch)+i)
            tf.summary.scalar('g_loss', G_loss, step=(e*steps_per_epoch)+i)
        
            print(f"step {(e*steps_per_epoch)+i}: d_loss = {D_loss} g_loss = {G_loss}")
        
        # Periodically sample generator
        if e % EPOCHS_PER_SAMPLE == 0:
            gan.sample(e)