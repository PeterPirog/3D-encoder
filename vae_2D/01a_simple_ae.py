import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def plot_image(image):
    plt.imshow(image, cmap='binary')
    plt.axis("off")


def show_reconstructions(model, x_valid, n_images=5):
    reconstructions = model.predict(x_valid[:n_images])
    reconstructions = reconstructions * 255  #
    x_valid = x_valid * 255
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image((x_valid[image_index]))
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    plt.show()

if __name__ == "__main__":
    units_layer1=100
    units_latenr_layer=30

    (x_train, y_train),(x_valid,y_valid) = keras.datasets.mnist.load_data()

    # printing the shapes of the vectors
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_valid:  ' + str(x_valid.shape))
    print('Y_valid:  ' + str(y_valid.shape))



    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.show()


    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_valid = np.expand_dims(x_valid, -1).astype("float32") / 255

    #convert to binary values 0 and 1
    x_train=tf.round(x_train)
    x_valid = tf.round(x_valid)

    encoder=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(units_layer1,activation='selu'),
        keras.layers.Dense(units_latenr_layer,activation='selu'),
    ])

    decoder=keras.models.Sequential([
        keras.layers.Dense(units_layer1,activation='selu',input_shape=[units_latenr_layer]),
        keras.layers.Dense(28*28,activation='sigmoid'),
        keras.layers.Reshape([28,28]),

    ])
    ae=keras.models.Sequential([encoder,decoder])
    ae.summary()
    ae.compile(loss="binary_crossentropy",
               optimizer=keras.optimizers.Adam(learning_rate=0.001))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                tf.keras.callbacks.TensorBoard(log_dir="./logs")]

    history=ae.fit(x_train,x_train,epochs=500,
           validation_data=[x_valid,x_valid],
                   callbacks=callbacks)



    show_reconstructions(ae,x_valid)