"""
The file is based on the file from https://keras.io/examples/generative/vae/
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
Features:
    - covariance loss
    - latent vector distribution Gaussioan
    - KL divergece used: No
    - L1 regularization used: No
    - Hyberparameter tuning: No
    -tf callbacks:
        -
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import seaborn as sns

"""
## Create a sampling layer
"""


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


def create_encoder_net(latent_dim):
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
    return encoder


"""
## Build the decoder
"""


def create_decoder_net(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

"""
## Define the VAE as a `Model` with a custom `train_step`
"""
class VAE(keras.Model):
    def __init__(self, latent_dim=5,beta_kl=0, beta_cov=150, beta_l1=0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        # loss parameters
        self.beta_kl=beta_kl
        self.beta_cov = beta_cov
        self.beta_l1 = beta_l1
        #self.mu0 = 0.0
        #self.sigma0 = 1.0

        # neural nets
        self.encoder = create_encoder_net(self.latent_dim)
        self.decoder = create_decoder_net(self.latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cov_loss_tracker = keras.metrics.Mean(name="cov_loss")
        self.l1_loss_tracker = keras.metrics.Mean(name="L1_loss")

    @property
    def metrics(self):
        metric = [self.total_loss_tracker,
                  self.reconstruction_loss_tracker,
                  self.kl_loss_tracker,
                  self.l1_loss_tracker,
                  self.cov_loss_tracker]
        return metric

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Calculate reconstruction loss for the batch
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            # Calculate KL divergence loss for the batch
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta_kl*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Calculate l1 loss for the batch
            l1_loss = tf.abs(z)
            l1_loss = tf.reduce_mean(tf.reduce_sum(l1_loss, axis=1))
            l1_loss = self.beta_l1 * l1_loss
            # sparse experiments

            # Calculate covariance loss for the batch
            cov_matrix = tfp.stats.correlation(z)
            cov_matrix = tf.linalg.band_part(cov_matrix, num_lower=0, num_upper=-1)  # get upper part
            cov_matrix = tf.linalg.set_diag(cov_matrix, tf.zeros([self.latent_dim]))  # zeroing diagonal
            cov_matrix = tf.math.abs(cov_matrix)  # get abs values of covariance

            n_nonzero = tf.math.count_nonzero(cov_matrix, dtype=tf.dtypes.float32)
            sum = tf.reduce_sum(cov_matrix)
            cov_loss = self.beta_cov * sum / n_nonzero

            # Calculate total loss for the batch
            # total_loss = reconstruction_loss + kl_loss +l1_loss
            # total_loss = reconstruction_loss + cov_loss
            #total_loss = reconstruction_loss + kl_loss + cov_loss + l1_loss
            total_loss = reconstruction_loss + kl_loss+cov_loss + l1_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.l1_loss_tracker.update_state(l1_loss)
        self.cov_loss_tracker.update_state(cov_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cov_loss": self.cov_loss_tracker.result(),
            "L1_loss": self.l1_loss_tracker.result(),
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


def show_reconstructions(model, x_valid, n_images=5,beta_kl=None, beta_cov=None, beta_l1=None):
    reconstructions = model.predict(x_valid[:n_images])
    reconstructions = reconstructions * 255  #
    x_valid = x_valid * 255
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image((x_valid[image_index]))
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])

    # set title
    plt.title(f"beta kl={beta_kl},beta cov={beta_cov}, beta_l1={beta_l1}")
    plt.show()


def create_activations_plot(z,beta_kl=None, beta_cov=None, beta_l1=None):
    values = np.abs(z[2])  # get values with reconstructed z only

    values = tf.reduce_mean(values, axis=0)
    # sort values and set descending direction
    values = np.sort(values)
    values = np.flip(values)

    latent_inputs = np.shape(values)[0]
    bars = [str(i) for i in range(0, latent_inputs)]
    x_pos = np.arange(len(bars))

    # Create bars
    plt.bar(x_pos, values)

    # Create names on the x-axis
    plt.xticks(x_pos, bars)

    # set title
    plt.title(f"beta kl={beta_kl},beta cov={beta_cov}, beta_l1={beta_l1}")

    # Show graphic
    plt.show()


if __name__ == "__main__":
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    # data preprocessing
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    # x_train = tf.math.round(x_train) # transform to zeros or ones
    # x_test = tf.math.round(x_test) # transform to zeros or ones

    # Define tf callbacks
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                         patience=3, min_lr=0.001),
                    tf.keras.callbacks.TensorBoard(log_dir="./logs"),  # tensorboard --logdir=./examples/logs
                    ]

    latent_dim = 5
    beta_kl=0
    beta_cov = 150
    beta_l1 = 0

    vae = VAE(latent_dim=latent_dim,
              beta_kl=beta_kl,
              beta_cov=beta_cov,
              beta_l1=beta_l1)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
    vae.fit(x_train, epochs=500, batch_size=128, callbacks=my_callbacks)

    # Predict reconstructed values and create visualization
    z = vae.encoder.predict(x_train)

    show_reconstructions(model=vae, x_valid=x_train, n_images=5, beta_kl=beta_kl,beta_cov=beta_cov, beta_l1=beta_l1)
    create_activations_plot(z,beta_kl=beta_kl, beta_cov=beta_cov, beta_l1=beta_l1)

    # make violin plot
    print(np.shape(z[2]))
    ax=sns.violinplot(data=z[2])
    ax.set_title("Output neuron activations")
    ax.set_ylabel("Activation value")
    ax.set_xlabel("Outputs")
    plt.show()

