from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
characters = string.digits + string.ascii_uppercase
width,height,n_len,n_class = 170,80,4,len(characters)
generator = ImageCaptcha(width=width,height=height)



def gen(batch_size = 32):
    X = np.zeros((batch_size, height,width, 3),dtype = np.uint8)
    y = [np.zeros((batch_size,n_class),dtype = np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width,height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j,ch in enumerate(random_str):
                y[j][i,:] = 0
                y[j][i,characters.find(ch)] = 1
        yield X,y
def decode(y):
    y = np.argmax(np.array(y),axis=2)[:,0]
    return ''.join([characters[x] for x in y])

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

input_tensor = tf.keras.Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(32*2**i, 3, 3, activation='relu',padding='same')(x)
    x = Conv2D(32*2**i, 3, 3, activation='relu',padding='same')(x)
    x = MaxPool2D((2, 2),padding='same')(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x)
 for i in range(4)]
model = Model(input_tensor, x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(gen(), validation_data=gen(),steps_per_epoch=51200, epochs=5
                ,verbose=1,workers=1,
                     validation_steps=1280)
keras_model_path = "./tmp/keras_save"
model.save(keras_model_path)
