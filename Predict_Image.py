#Import Libraries
from keras.models import load_model
import cv2
import numpy as np
#Load Model
model = load_model('fruits.h5')
#Execute Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Image Processing
img = cv2.imread('apple.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)
#Print Output Prediction
print (classes)
