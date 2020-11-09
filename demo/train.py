import tensorflow as tf 
import os
#MODEL_PATH=
model_save_path ="model"
if __name__ == "__main__":
#获取mnist数据集 并归一化
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#构建模型
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test,  y_test, verbose=2)
    model.save(model_save_path)
#删除之前保存的
    #if os.path.isdir(model_save_path):
        #shutil.rmtree(model_save_path)
    #os.mkdir(model_save_path)
    
