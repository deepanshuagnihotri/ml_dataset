#Call Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
#Model Parameters 
model=Sequential()
model.add(Convolution2D(32,3,3, input_shape=(64,64,3) ,activation='relu'))
model.add(Flatten())
model.add(Dense(activation='relu',output_dim=128))
model.add(Dense(output_dim=1,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Train Data Model
train_datagen=ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

#Train Directory
training_set=train_datagen.flow_from_directory('dataset/train',target_size=(64,64),batch_size=32,class_mode='binary')
#Test Directory
training_set=train_datagen.flow_from_directory('dataset/test',target_size=(64,64),batch_size=32,class_mode='binary')

#Model Creation
model.fit_generator(training_set,steps_per_epoch=1000,epochs=10,validation_data=training_set,validation_steps=8000)
model.save('fruits.h5')