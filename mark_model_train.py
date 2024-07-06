import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
import numpy as np

arr = np.load('./markdataset/datanpz/imageData.npz')
img_list = arr['arr_0']
label_list =arr['arr_1']
print(img_list.shape,label_list.shape)
da = label_list.reshape(-1, 1)
# 编码
y_onehot = onehot.fit_transform(label_list.reshape(-1, 1))
y_onehot_arr = y_onehot.toarray()
x_train, x_test, y_train, y_test = train_test_split(img_list, y_onehot_arr, test_size=0.2, random_state=123)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")

model = models.Sequential([
    layers.Conv2D(16, 3, padding='same', input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(166, activation='relu'),
    layers.Dense(22, activation='relu'),
    layers.Dense(3, activation='softmax') #sigmoid
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=(x_test,y_test),
                    batch_size=30,
                    epochs=15)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(loss))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save('./model weights/mark_1.h5')