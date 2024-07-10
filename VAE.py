import tensorflow as tf
import keras
from keras import layers

# If astroNN package is unvailable, please install it using:
# !pip install astroNN -q
from astroNN.datasets import load_galaxy10sdss

# Data Augmentation Routine


def augment(img):
    img = tf.image.resize(img, [64, 64], method='gaussian', antialias=True)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

# Normalization to [0,1] range


def rescale(img):
    img = tf.cast(img, dtype=tf.float32)
    img = img / 255.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img

# tf.Data mapping funcion


def train_preprocessing(x):
    img = augment(x)
    img = rescale(img)
    return img

# Convert ot tf.Data dataset object


def get_data(images):
    dataset = tf.convert_to_tensor(images, dtype=tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_ds = (
        dataset.map(train_preprocessing)
        .repeat(r)
        .batch(128, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds

# Latent Layer Implementation


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder

def get_encoder(latent_dim=64):
    encoder_inputs = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(
        32,
        3,
        activation="relu",
        strides=2,
        padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs, [
            z_mean, z_log_var, z], name="encoder")
    return encoder


# Decoder

def get_decoder(latent_dim=64):

    latent_inputs = keras.Input(shape=(latent_dim,))
    inp_dim = 2
    x = layers.Dense(inp_dim * inp_dim * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((inp_dim, inp_dim, 64))(x)
    x = layers.Conv2DTranspose(
        512,
        3,
        activation="relu",
        strides=2,
        padding="same")(x)
    x = layers.Conv2DTranspose(
        256,
        3,
        activation="relu",
        strides=2,
        padding="same")(x)
    x = layers.Conv2DTranspose(
        128,
        3,
        activation="relu",
        strides=2,
        padding="same")(x)
    x = layers.Conv2DTranspose(
        64,
        3,
        activation="relu",
        strides=2,
        padding="same")(x)
    x = layers.Conv2DTranspose(
        32,
        3,
        activation="relu",
        strides=2,
        padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        3, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

# Define the VAE model with a custom training step


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == "__main__":

    latent_dim = 64

    # Min and max of data
    clip_min = 0.0
    clip_max = 1.0

    # Data is repeated 3 times for each epoch
    # different augmentation for each repetition
    r = 3

    # Get images from AstroNN...might take a few minutes
    images, labels = load_galaxy10sdss()

    train_ds = get_data(images)
    encoder = get_encoder(latent_dim)

    decoder = get_decoder(latent_dim)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(train_ds, epochs=64)
