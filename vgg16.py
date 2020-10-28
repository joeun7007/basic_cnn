import tensorflow as tf
from tensorflow import keras


# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# label을 onehot-encoding
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

IMG_SIZE = 224

def img_resize(images, labels):
    return tf.image.resize(images, (IMG_SIZE, IMG_SIZE)), labels

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128).map(img_resize)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000).map(img_resize)
test_data_iter = iter(test_data)

class VGG(tf.keras.Model):
    def __init__(self):
        super(VGG, self).__init__()
        # 1
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')

        # 2
        self.conv3 = tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                strides=2,
                                                padding='same')

        # 3
        self.conv5 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv7 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')
        
        # 3
        self.conv8 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv9 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv10 = tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')

        # 4
        self.conv11 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv12 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv13 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')

        # 5
        self.conv14 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv15 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.conv16 = tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation='relu')
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same')

        self.conv_flat = tf.keras.layers.Flatten()
        self.fc_layer1 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc_layer2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(10)

    def call(self, x, is_training):
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.pool1(x)
       x = self.conv3(x)
       x = self.conv4(x)
       x = self.pool2(x)
       x = self.conv5(x)
       x = self.conv6(x)
       x = self.conv7(x)
       x = self.pool2(x)
       x = self.conv8(x)
       x = self.conv9(x)
       x = self.conv10(x)
       x = self.pool3(x)
       x = self.conv11(x)
       x = self.conv12(x)
       x = self.conv13(x)
       x = self.pool4(x)
       x = self.conv14(x)
       x = self.conv15(x)
       x = self.conv16(x)
       x = self.pool5(x)
       x = self.fc_layer1(x)
       x = self.fc_layer2(x)
       x = self.dropout(x, training=is_training)
       logits = self.output_layer(x)
       y_pred = tf.nn.softmax(logits)

       return y_pred, logits

# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 최적화를 위한 RMSprop 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y, is_training):
  with tf.GradientTape() as tape:
    y_pred, logits = model(x, is_training)
    loss = cross_entropy_loss(logits, y)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
  correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy

# Convolutional Neural Networks(CNN) 모델을 선언합니다.
CNN_model = VGG()

# 10000 Step만큼 최적화를 수행합니다.
for i in range(10000):
  batch_x, batch_y = next(train_data_iter)

  # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
  if i % 100 == 0:
    train_accuracy = compute_accuracy(CNN_model(batch_x, False)[0], batch_y)
    loss_print = cross_entropy_loss(CNN_model(batch_x, False)[1], batch_y)

    print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
  # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
  train_step(CNN_model, batch_x, batch_y, True)

# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy = 0.0
for i in range(10):
  test_batch_x, test_batch_y = next(test_data_iter)
  test_accuracy = test_accuracy + compute_accuracy(CNN_model(test_batch_x, False)[0], test_batch_y).numpy()
test_accuracy = test_accuracy / 10
print("테스트 데이터 정확도: %f" % test_accuracy)