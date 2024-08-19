import jax
from jax import random, numpy as jnp, tree
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence
from dataclasses import dataclass
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import os


@dataclass
class EncoderConfig:
    conv_filters: Sequence[int] = (32, 64, 64, 64)
    conv_strides: Sequence[int] = (1, 2, 2, 1)
    latent_dim: int = 100

class Encoder(nn.Module):
    config: EncoderConfig

    @nn.compact
    def __call__(self, x, noise_rng):
        for i in range(len(self.config.conv_filters)):
            x = nn.Conv(
                features=self.config.conv_filters[i],
                kernel_size=(3, 3),
                strides=self.config.conv_strides[i],
                padding='same')(x)
            x = nn.leaky_relu(x)
        x = jax.vmap(jnp.ravel)(x)
        mean = nn.Dense(features=self.config.latent_dim)(x)
        logvar = nn.Dense(features=self.config.latent_dim)(x)
        epsilon = random.normal(key=noise_rng, shape=mean.shape)
        x = mean + jnp.exp(logvar / 2) * epsilon
        return x, mean, logvar


@dataclass
class DecoderConfig:
    conv_filters: Sequence[int] = (64, 64, 32, 1)
    conv_strides: Sequence[int] = (1, 2, 2, 1)
    latent_dim: int = 100


class Decoder(nn.Module):
    config: DecoderConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=7 * 7 * 64)(x) # shape of image before flatenning in encoder
        x = jnp.reshape(x, (-1, 7, 7, 64))
        for i in range(len(self.config.conv_filters)):
            x = nn.ConvTranspose(
                features=self.config.conv_filters[i],
                kernel_size=(3, 3),
                strides=self.config.conv_strides[i],
                padding='SAME')(x)
            if (i+1) != len(self.config.conv_filters):
                x = nn.leaky_relu(x)
            else:
                x = nn.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig

    @nn.compact
    def __call__(self, x, noise_rng):
        x, mean, logvar = Encoder(self.encoder_config)(x, noise_rng)
        x = Decoder(self.decoder_config)(x)
        return x, mean, logvar


key, subkey = random.split(random.key(1337))
encoder_config = EncoderConfig()
decoder_config = DecoderConfig()

autoencoder = Autoencoder(encoder_config, decoder_config)
params = autoencoder.init(
    key,
    jnp.ones(shape=(1, 28, 28, 1), dtype=jnp.float32),
    noise_rng=subkey
)
tx = optax.adam(learning_rate=3e-5)
state = train_state.TrainState.create(
    apply_fn=autoencoder.apply,
    params=params,
    tx=tx)



@jax.jit
def train_step(state, x, y, noise_key):
    encoder_noise_key = random.fold_in(key=noise_key, data=state.step)
    def forward_and_loss(params, x, y):
        y_pred, mean, logvar = state.apply_fn(
            params,
            x,
            encoder_noise_key)
        mse_loss = optax.squared_error(y, y_pred).mean()
        kl_loss = -0.5 * jnp.sum(
            1 + logvar - jnp.square(mean) - jnp.exp(logvar),
            axis=1).mean()
        loss = mse_loss + kl_loss
        return loss
    grad_fn = jax.value_and_grad(forward_and_loss)
    loss, grads = grad_fn(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ======================
# LOAD DATASET
# ======================
batch_size = 128
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)

mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) - 127.5) / 127.5)


decoder = Decoder(decoder_config)
def generate_and_save_image(decoder_params, latent_dim, epoch, key):
    noise = random.normal(key, (1, latent_dim))
    fake_image = decoder.apply(decoder_params, noise)
    fake_image = np.asarray(fake_image).reshape((28, 28)) * 127.5 + 127.5
    img = Image.fromarray(fake_image).convert("L")
    if not os.path.exists("./images/vae"):
        os.makedirs("./images/vae")
    img.save(f"./images/vae/epoch_{epoch}_decoded.png")

latent_dim = decoder_config.latent_dim
n_epochs = 1000
n_batches = 0
per_batch_losses = []
key = random.key(99999)
for epoch in range(1, n_epochs+1):
    for batch, x in enumerate(mnist_ds, start=1):
        batch_str = "if you're seeing this, something's wrong"
        x = x.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?"
        else:
            batch_str = f"{batch}/{n_batches}"
        state, loss = train_step(state, x, x, key)   # targets are the real images
        key, subkey = random.split(subkey)
        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Loss: {loss:.6f}", end="")
        per_batch_losses.append(loss)

    loss = jnp.mean(jnp.asarray(per_batch_losses))
    print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
          f"Loss: {loss:.6f}", end="")
    key, subkey = random.split(subkey)
    if epoch % 10 == 0:
        generate_and_save_image(state.params['params']['Decoder_0'],
                                decoder_config.latent_dim, epoch, key)
    print()