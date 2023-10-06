#Importing necessary libraries
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import tensorflow
from tensorflow.keras.layers import LSTM, Dense


#Function to create an LSTM model
def lstmmodel():
  input_shape=(259,128) # Change the second dimension according to n_mfcc
  model = tensorflow.keras.Sequential()
  model.add(LSTM(128,input_shape=input_shape,return_sequences=True))
  model.add(tensorflow.keras.layers.Dropout(0.3))
  model.add(LSTM(128,input_shape=input_shape))
  model.add(tensorflow.keras.layers.Dropout(0.3))
  model.add(Dense(64, activation='relu'))
  model.add(tensorflow.keras.layers.Dropout(0.3))
  model.add(Dense(10, activation='softmax'))
  model.summary()

  return model

def training(X_train,y_train,X_val,y_val,model):
  model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['acc'])
  history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_val, y_val), shuffle=False)
  return model,history