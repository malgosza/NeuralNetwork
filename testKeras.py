import tensorflow as tf
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist #28x28 images of hand-written digit 0-9
(x_train, y_train),(x_test,Y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
#
# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=3)
#
# val_loss, val_acc=model.evaluate(x_test,y_train)
# # print(val_loss,val_acc)
#
# model.save('epic_num_reader.model')

plt.imshow(x_train[0])
plt.show()