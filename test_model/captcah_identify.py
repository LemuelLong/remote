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

class Captcha(Model):
    def __init__(self):
        super(Captcha,self).__init__()
        self.c11 = Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu')
        self.c12 = Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu')
        self.p1 = MaxPool2D(pool_size=(2, 2))

        self.c21 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.c22 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.p2 = MaxPool2D(pool_size=(2, 2))

        self.c31 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.c32 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.p3 = MaxPool2D(pool_size=(2, 2))

        self.c41 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')
        self.c42 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')
        self.p4 = MaxPool2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.d = Dropout(0.25)
        self.f1 = Dense(36, activation='softmax')
        self.f2 = Dense(36, activation='softmax')
        self.f3 = Dense(36, activation='softmax')
        self.f4 = Dense(36, activation='softmax')

    def call(self, x):
        x = self.c11(x)
        x = self.c12(x)
        x = self.p1(x)

        x = self.c21(x)
        x = self.c22(x)
        x = self.p2(x)

        x = self.c31(x)
        x = self.c32(x)
        x = self.p3(x)

        x = self.c41(x)
        x = self.c42(x)
        x = self.p4(x)

        x = self.flatten(x)
        x1 = self.f1(x)
        x2 = self.f2(x)
        x3 = self.f3(x)
        x4 = self.f4(x)

        return x1,x2,x3,x4
model = Captcha()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(gen(), validation_data=gen(),steps_per_epoch=51200, epochs=5
                ,verbose=1,workers=0,
                     validation_steps=1280)


