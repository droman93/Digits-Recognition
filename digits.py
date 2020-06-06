%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as KU
import  seaborn as sns
test_set = pd.read_csv('test.csv')
train_set = pd.read_csv('train.csv')
# read first row -----------------------------------------q
X = []
for i in range(0,train_set.shape[0]):
    X.append(train_set.iloc[i,1:].values.reshape(28,28,1))
X = np.array(X) #----------------------------------------------------d

y = train_set.iloc[:,0].values
y = KU.to_categorical(y, num_classes=10)

# test data ----------------------------------------q
X_test = []
for i in range(0,test_set.shape[0]):
    X_test.append(test_set.iloc[i,0:].values.reshape(28,28,1))
X_test = np.array(X_test) #---------------------------------------------------d


# splitting
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=1)


# image  -  just for testing
plt.imshow(X_test[3].reshape(28,28))


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))) # pooling

# Adding a second convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))) # pooling

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# data preparation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size,shuffle=True)
val_generator = train_datagen.flow(X_val,y_val,batch_size=batch_size,shuffle=True)


#   #       model trainingclassifier.fit_generator(train_generator,
#         steps_per_epoch= X_train.shape[0] // batch_size,
#         epochs=50,
#         validation_data=val_generator,
#         validation_steps=X_val.shape[0] // batch_size)



from keras.models import model_from_json,load_model
#   #   Save the model
classifier.save_weights('model_wieghts.h5')
classifier.save('model_keras.h5')

loaded_model = load_model('model_keras.h5')
loaded_model.load_weights('model_wieghts.h5')

#   #   #   #   #   #   #   #   #

classifier.predict(X_test[0])
# # #  Prediciton
probabilities = []
k=0
for batch in test_datagen.flow(X_test,batch_size=1,shuffle=False):
    pred = classifier.predict(batch)
    probabilities.append((np.argmax(pred)))
    # plt.figure(k)
    # plt.imshow(batch[0].reshape(28, 28))
    # plt.title(str(np.argmax(pred)))
    print(k)
    if k == 28000-1:
        break
    k += 1


len(probabilities)
pred = loaded_model.predict(test_datagen.flow(X_test[0],batch_size=1))

plt.imshow(batch[0].reshape(28, 28))





# sending to Kaggle
sub = {'ImageId': range(1,X_test.shape[0]+1),'Label':probabilities}
file = pd.DataFrame(sub)
file.to_csv('MySubDig.csv',index = False)

##### bull shiting around -------------------------------------------------------------------------------------------------
import cv2
img = cv2.imread('2.png')
plt.imshow(img)
img2 = []
img2.append(cv2.resize(cv2.imread('2.png',cv2.IMREAD_GRAYSCALE), (28,28)).reshape(28,28,1))
# img2.append(X_test[0])
img2 = np.array(img2)
img2.shape
pred = loaded_model.predict(test_datagen.flow(img2,batch_size=1,shuffle=False))
np.argmax(pred)
