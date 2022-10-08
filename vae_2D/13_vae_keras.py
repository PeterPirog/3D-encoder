"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

# added binary crossentropy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

"""
## Create a sampling layer
"""
latent_dim = 8


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        sigma = tf.exp(0.5 * z_log_var)
        return z_mean + sigma * epsilon


"""
## Build the encoder
"""

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim=5, beta_cov=150, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta_cov = beta_cov
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cov_loss_tracker = keras.metrics.Mean(name="cov_loss")
        self.l1_loss_tracker = keras.metrics.Mean(name="L1_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.cov_loss_tracker,
            self.l1_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            # sparse experiments

            # corr matrix
            cov_matrix = tfp.stats.correlation(z)
            cov_matrix = tf.linalg.band_part(cov_matrix, num_lower=0, num_upper=-1)  # get upper part
            # cov_matrix = tf.linalg.set_diag(cov_matrix, tf.zeros([m]))  # zeroing diagonal
            cov_matrix = tf.linalg.set_diag(cov_matrix, tf.zeros([self.latent_dim]))  # zeroing diagonal
            cov_matrix = tf.math.abs(cov_matrix)  # get abs values of covariance

            n_nonzero = tf.math.count_nonzero(cov_matrix, dtype=tf.dtypes.float32)
            sum = tf.reduce_sum(cov_matrix)
            cov_loss = self.beta_cov * sum / n_nonzero

            # total_loss = reconstruction_loss + kl_loss +l1_loss
            total_loss = reconstruction_loss + cov_loss
            # total_loss =l1_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.cov_loss_tracker.update_state(cov_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cov_loss": self.cov_loss_tracker.result(),
        }

    def predict(self, data):
        latent_vector = self.encoder.predict(data)
        reconstructed = vae.decoder.predict(latent_vector[2])
        return reconstructed


"""
## Display a grid of sampled digits
"""

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

def create_activations_plot(z):
    values = np.abs(z[2])

    values = tf.reduce_mean(values, axis=0)
    values=np.sort(values)
    values=np.flip(values)

    latent_inputs=np.shape(values)[0]
    bars = [str(i) for i in range(0, latent_inputs)]
    x_pos = np.arange(len(bars))

    # Create bars
    plt.bar(x_pos, values)

    # Create names on the x-axis
    plt.xticks(x_pos, bars)

    # Show graphic
    plt.show()


if __name__ == "__main__":
    """
    ## Train the VAE
    """


    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    # data preprocessing
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    x_train = tf.math.round(x_train)
    x_test = tf.math.round(x_test)

    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                         patience=3, min_lr=0.001)
                    ]
    beta_cov = 500
    vae = VAE(encoder, decoder, latent_dim=latent_dim, beta_cov=beta_cov)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
    vae.fit(x_train, epochs=500, batch_size=128, callbacks=my_callbacks)

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255

    z = vae.encoder.predict(x_train)

    # Create visualization
    show_reconstructions(model=vae, x_valid=x_train, n_images=5)
    create_activations_plot(z)

