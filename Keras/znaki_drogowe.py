import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)

"""# Otwieranie danych

"""

with open('german-traffic-signs-1/train.p','rb') as file:
  train_data = pickle.load(file)
with open('german-traffic-signs-1/valid.p','rb') as file:
  valid_data = pickle.load(file)
with open('german-traffic-signs-1/test.p','rb') as file:
  test_data = pickle.load(file)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']


data = pd.read_csv('german-traffic-signs-1/signnames.csv')

num_of_samples=[]
 
cols = 5
num_classes = 43
 
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
# fig.tight_layout()
 
# for i in range(cols):
#     for j, row in data.iterrows():
#       x_selected = X_train[y_train == j]
#       axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
#       axs[j][i].axis("off")
#       if i == 2:
#         axs[j][i].set_title(str(j) + "-"+ row['SignName'])
#         num_of_samples.append(len(x_selected))

# print(num_of_samples)
# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the train dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()


"""# Image Preprocesing

"""

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# img = grayscale(X_train[1000])
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# print(img.shape)

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# img = equalize(img)
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# print(img.shape)

def preprocesing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocesing, X_train)))
X_val = np.array(list(map(preprocesing, X_val)))
X_test = np.array(list(map(preprocesing, X_test)))

# plt.imshow(X_train[random.randint(0, len(X_train)-1)], cmap='gray')
# plt.axis('off')
# print(X_train.shape)

X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)

"""# Fit Generator


"""

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range= 0.1, rotation_range=10)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# fig, axis = plt.subplots(1, 15, figsize=(20,5))
# fig.tight_layout()

# for i in range(15):
#   axis[i].imshow(X_batch[i].reshape(32,32), cmap=plt.get_cmap('gray'))
#   axis[i].axis('off')

# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

"""# leNet Implementation"""

def leNet_model():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape = (32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def modified_model():
    model = Sequential()
    model.add(Conv2D(60,(5,5), input_shape = (32,32,1), activation='relu'))
    model.add(Conv2D(60,(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = modified_model()
print(model.summary())

# history = model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_val, y_val), batch_size=400, verbose=1, shuffle=1)
# training_set = datagen.flow(X_train, y_train, batch_size=50)
history = model.fit(datagen.flow(X_train, y_train, batch_size=50), epochs=10,validation_data=(X_val, y_val), verbose=1, shuffle=1)

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel('Epoch')
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel('Epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose = 0 )
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

"""# Save pickle model """
pickle_out = open("myTraffic_sign_model.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

"""# Testing Image"""

# import requests
# from PIL import Image
# url = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
# r = requests.get(url, stream=True)
# img = Image.open(r.raw)
# plt.imshow(img, cmap=plt.get_cmap('gray'))

# #Preprocess image
# img = np.asarray(img)
# img = cv2.resize(img, (32, 32))
# img = preprocesing(img)
# plt.imshow(img, cmap = plt.get_cmap('gray'))
# print(img.shape)

# #Reshape reshape
# img = img.reshape(1, 32, 32, 1)

# #Test image
# print("predicted sign: "+ str(np.argmax(model.predict(img), axis=1)))


# img = cv2.imread('priority_road.png')
# plt.imshow(img)

# #Preprocess image
# img = np.asarray(img)
# img = cv2.resize(img, (32, 32))
# img = preprocesing(img)
# plt.imshow(img, cmap = plt.get_cmap('gray'))
# print(img.shape)

# #Reshape reshape
# img = img.reshape(1, 32, 32, 1)

# #Test image
# print("predicted sign: "+ str(np.argmax(model.predict(img), axis=1)))