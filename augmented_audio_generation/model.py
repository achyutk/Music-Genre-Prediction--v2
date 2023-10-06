# Import modules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Activation, Input
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv1D, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import soundfile as sf


# Defining hyperparameters of GAN mode
MODEL_DIMS = 64   #Set the d values in Critic and generator
NUM_SAMPLES = 65536 
Fs = 16000  
NOISE_LEN = 100

GRADIENT_PENALTY_WEIGHT = 10.0  # Defining Hyperparameters for Loss Function

# Defining hyperparameters for training
D_UPDATES_PER_G_UPDATE = 5 # Decides how many time a GAN will update the generator for a particular batch is used for training a 
EPOCHS = 50
EPOCHS_PER_SAMPLE = 2  # To determing when to generate the audio file and save the model. It is generated at every even number of epochs
BATCH_SIZE = 64

INSTRUMENT = "rock"   # Change this according to the genre you want to develop

# Define class that contains GAN infrastructure
class GAN:
    def __init__(self, model_dims=MODEL_DIMS, num_samples=NUM_SAMPLES, 
                 gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT, instrument=INSTRUMENT,
                 noise_len=NOISE_LEN, batch_size=BATCH_SIZE, sr=Fs):
        self.model_dims = model_dims
        self.num_samples = num_samples
        self.noise_dims = (noise_len,)
        self.batch_size = batch_size
        
        # self.G = GANModels.Generator(self.model_dims, num_samples)
        self.G = Generator(self.model_dims, num_samples)
        print(self.G.summary())

        # self.D = GANModels.Critic(self.model_dims, num_samples)
        self.D = Critic(self.model_dims, num_samples)
        print(self.D.summary())
        
        self.G_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        
        self.gradient_penalty_weight = gradient_penalty_weight
        
        self.sr = sr

        self.instrument = INSTRUMENT

    # Loss function for discriminator
    def _d_loss_fn(self, rlog, flog):
        fl = tf.reduce_mean(flog)
        rl = - tf.reduce_mean(rlog)
        return rl, fl
    
    # Loss function for generator
    def _g_loss_fn(self, flog):
        fl = - tf.reduce_meanflog
        return fl

    # Calculates gradient penalty
    def _gradient_penalty(self, real, fake):
        # performs intrapolation
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter
            
        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = self.D(x, training=True)
            
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp
        
    # Trains generator by keeping critic constant and returns the loss after traning
    @tf.function
    def train_G(self):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batch_size,) + self.noise_dims)
            x_fake = self.G(z, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)
            G_loss = self._g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))

        return {'g_loss': G_loss}

    # Trains critic by keeping generator constant
    @tf.function
    def train_D(self, x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(x_real.shape[0],) + self.noise_dims)
            x_fake = self.G(z, training=True)

            x_real_d_logit = self.D(x_real, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = self._d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = self._gradient_penalty(x_real, x_fake)

            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.gradient_penalty_weight

        D_grad = t.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}
        
    # Creates music samples and saves current generator model
    def sample(self, epoch, num_samples=50):
        self.G.save(f"models/{epoch}.h5")
        z = tf.random.normal(shape=(num_samples,) + self.noise_dims)
        result = self.G(z, training=False)
        for i in range(num_samples):
            audio = result[i, :, :]
            audio = np.reshape(audio, (self.num_samples,))
            sf.write(f"/content/drive/MyDrive/Outputs/{self.instrument}/{epoch}-{i}.wav",audio,samplerate=self.sr)



# Defining the Generator of GAN
def Generator(d, num_samples, c=16):

    input_layer = Input(shape=(100,))

    # Upsampling

    # output shape = (None, 16, 16d)
    dense0 = Dense(16*c*d)(input_layer)
    reshape0 = Reshape((c, c*d))(dense0)
    relu0 = ReLU()(reshape0)
 
    # output shape = (None, 64, 8d)
    c //= 2
    expanded0 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu0)
    conv0 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded0)
    slice0 = Lambda(lambda x: x[:, 0])(conv0)
    relu1 = ReLU()(slice0)

    # output shape = (None, 256, 4d)
    c //= 2
    expanded1 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu1)
    conv1 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded1)
    slice1 = Lambda(lambda x: x[:, 0])(conv1)
    relu2 = ReLU()(slice1)

    # output shape = (None, 1024, 2d)
    c //= 2
    expanded2 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu2)
    conv2 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded2)
    slice2 = Lambda(lambda x: x[:, 0])(conv2)
    relu3 = ReLU()(slice2)

    # output shape = (None, 4096, d)
    c //= 2
    expanded3 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu3)
    conv3 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded3)
    slice3 = Lambda(lambda x: x[:, 0])(conv3)
    relu4 = ReLU()(slice3)

    # output shape = (None, 16384, d)
    expanded4 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu4)
    conv4 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded4)
    slice4 = Lambda(lambda x: x[:, 0])(conv4)
    relu5 = ReLU()(slice4)
 

    # output shape = (None, 65536, 1)
    expanded5 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu5)
    conv5 = Conv2DTranspose(1, (1, 25), strides=(1, 4), padding='same')(expanded5)
    slice5 = Lambda(lambda x: x[:, 0])(conv5)

    #### num_samples == 65536

    # Squeeze values between (-1, 1)
    tanh0 = Activation('tanh')(slice5)

    model = Model(inputs=input_layer, outputs=tanh0)

    return model


# Implementation of PHase Shuffle
def _apply_phaseshuffle(x, rad=2, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

# Discriminator of the GAN
def Critic(d, num_samples, c=1):

    input_layer = Input(shape=(num_samples, 1))

    # Downsampling

    # output shape = (None, 4096, d)
    conv0 = Conv1D(c*d, 25, strides=4, padding='same')(input)
    LReLU0 = LeakyReLU(alpha=0.2)(conv0)
    phaseshuffle0 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU0)

    # output shape = (None, 1024, 2d)
    c *= 2
    conv1 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle0)
    LReLU1 = LeakyReLU(alpha=0.2)(conv1)
    phaseshuffle1 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU1)

    # output shape = (None, 256, 4d)
    c *= 2
    conv2 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle1)
    LReLU2 = LeakyReLU(alpha=0.2)(conv2)
    phaseshuffle2 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU2)

    # output shape = (None, 64, 8d)
    c *= 2
    conv3 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle2)
    LReLU3 = LeakyReLU(alpha=0.2)(conv3)
    phaseshuffle3 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU3)

    # output shape = (None, 16, 16d)
    c *= 2
    conv4 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle3)
    LReLU4 = LeakyReLU(alpha=0.2)(conv4)

    #### num_samples == 65536

    # output shape = (None, 256d)
    reshape0 = Reshape((64*c*d,))(LReLU4)#

    # Output a critic score
    dense1 = Dense(1)(reshape0)

    model = Model(inputs=input_layer, outputs=dense1)

    return model

