# It showed the test_acc of 0.986 in comparison to that of 0.965 in the textbook on page 90.

import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split( cancer.data , cancer.target , random_state= 42 )
X_test, X_val , y_test, y_val = train_test_split( X_test, y_test , random_state=42, train_size=0.5  )

######## KERAS ########

import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
import numpy as np

# Data Pre-processing - Normalizing
max_val = np.amax(X_train)
X_train = X_train / max_val
X_val = X_val / max_val
X_test = X_test / max_val

# One_hot encoding
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)


# Model
model = Sequential()

model.add(Dense(units=30, input_dim=X_train.shape[1], kernel_initializer='he_normal'))
model.add(BatchNormalization()) # Batch Normalization
model.add(Activation('relu'))

model.add(Dense(units=30, kernel_initializer='he_normal'))
model.add(BatchNormalization()) # Batch Normalization
model.add(Activation('relu'))

model.add(Dense(units=30 , kernel_initializer='he_normal', activation=None)) # We don't put "Batch Normalization" and "activation func." b/w the last hidden layer and the output layer
model.add(Dense(units=y_test.shape[1], activation='softmax')) # multi-classification : from a number 0 to 9.


# Loss func. , Optimizer
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam , metrics=['accuracy']) # "categorical_crossentropy" must go with "softmax"


# Train
batch_size = 2**5  # Usually, batch_size is a multiple of 2. (2의 배수)  |  ref: http://3months.tistory.com/73
tb_his = keras.callbacks.TensorBoard(log_dir='C:\\python64\\envs\\venv\\python64\\logs', histogram_freq=0, write_graph=True, write_images=True) # To use Tensorboard
early_stopping = keras.callbacks.EarlyStopping(patience=3) # No patience will lead to "hasty stopping"
hist = model.fit(X_train, y_train, epochs=1000, batch_size=batch_size , validation_data=[X_val, y_val] ,  callbacks=[tb_his, early_stopping ])


# Training process history
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.title('Batch_Normalization Applied')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])

plt.subplot(2,1,2)
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])
plt.ylim(0,1)
plt.show()


# Validation for the entire test set
test_acc =  np.mean( np.argmax( model.predict(X_test), axis=1) == np.argmax(y_test, axis=1) )
print('test_acc : ', test_acc)


# Predict
pred = model.predict( X_test[0].reshape(1,len(X_test[0])) )
pred = np.argmax(pred)
t = np.argmax(y_test[0])
print('\n # Prediction')
print( ' pred:',pred, ' test:',t )


# Save the trained model
model.save('C:\\python64\\python64\\mnist_dnn_model.h5')

'''
# Load the trained model
from keras.models import load_model
model = load_model('C:\\python64\\python64\\mnist_dnn_model.h5')
'''

