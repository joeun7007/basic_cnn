import tensorflow as tf
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255., x_test / 255.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

learning_rate = 0.01
N_EPOCHS = 10
N_BATCH = 100
N_CLASS = 10
IMG_SIZE = 227

def img_resize(images, labels):
    return tf.image.resize(images, (IMG_SIZE, IMG_SIZE)), labels

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(50000).batch(N_BATCH).map(img_resize)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.repeat().map(img_resize).batch(N_BATCH)

def AlexNet():
    model = keras.Sequential()
    # layer1
    model.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid',
                                  activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(keras.layers.BatchNormalization())
    # layer2
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same',
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(keras.layers.BatchNormalization())
    # layer3
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                                  activation='relu'))
    # layer4
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same',
                                  activation='relu'))
    # layer 5
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same',
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(keras.layers.BatchNormalization())
    # layer 6
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # layer 7
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # layer 8
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model

model = AlexNet()
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = 500
validation_steps = 100

history=model.fit(train_data, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch,
                  validation_data=test_data, validation_steps=validation_steps)






















