import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Any
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from functools import partial
import os


class Generator(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x, train):
        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.5,
                axis_name='batch')(x)
        x = nn.Dense(features=self.out_dim)(x)
        x = nn.tanh(x)
        return x


class Discriminator(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int = 1

    @nn.compact
    def __call__(self, x, train):
        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class TrainState(train_state.TrainState):
    batch_stats: Any


latent_dim: int = 100
image_dim: int = 28 * 28 * 1
generator = Generator((256, 512, 1024), 784)
discriminator = Discriminator((512, 256, 128), 1)

key = random.key(1337)
generator_key, discriminator_key = random.split(key, 2)

generator_variables = generator.init(
    generator_key,
    jnp.ones((1, latent_dim)),
    train=False)
generator_state = TrainState.create(
    apply_fn=generator.apply,
    params=generator_variables['params'],
    batch_stats=generator_variables['batch_stats'],
    tx=optax.adam(0.0002, b1=0.5),
)
del generator_variables

discriminator_variables = discriminator.init(
    discriminator_key,
    jnp.ones((1, image_dim)),
    train=False)
discriminator_state = TrainState.create(
    apply_fn=discriminator.apply,
    params=discriminator_variables['params'],
    batch_stats=None,
    tx=optax.adam(0.0002, b1=0.5),
)


@jax.jit
def train_step(
        generator_state: TrainState,
        discriminator_state: TrainState,
        x_batched,
        noise_batched_for_disc,
        noise_batched_for_gen):
    metrics = dict()

    def discriminator_forward_and_loss(params, x_batched, y):
        def for_a_single_instance(x):
            y_pred = discriminator_state.apply_fn(
                {'params': params},
                x=x,
                train=True)
            # loss = y * jnp.log(y_pred) + (1 - y) * jnp.log(1. - y_pred)
            # loss = optax.sigmoid_binary_cross_entropy(y_pred, y)
            loss = -y * jax.nn.log_sigmoid(y_pred) - (1 - y) * jax.nn.log_sigmoid(-y_pred)
            return loss

        loss = jax.vmap(
            for_a_single_instance,
            in_axes=0,
            out_axes=0)(x_batched)
        loss = jnp.mean(loss)
        return loss

    discriminator_grad_fn = jax.value_and_grad(discriminator_forward_and_loss,
                                               has_aux=False)
    disc_loss_real, grads_real = discriminator_grad_fn(
        discriminator_state.params,
        x_batched,
        1.0)

    fake_images = generator_state.apply_fn(
        {'params': generator_state.params,
         'batch_stats': generator_state.batch_stats},
        noise_batched_for_disc,
        train=False,
    )
    disc_loss_fake, grads_fake = discriminator_grad_fn(
        discriminator_state.params,
        fake_images,
        0.0)
    grads = jax.tree.map(lambda g1, g2: g1 + g2, grads_real, grads_fake)
    discriminator_state = discriminator_state.apply_gradients(grads=grads)
    disc_loss = disc_loss_real + disc_loss_fake
    metrics.update({'disc_loss': disc_loss})

    def generator_forward_and_loss(
            gen_params,
            disc_params,
            noise_batched
    ):
        def for_single_instance(x):
            fake_image, gen_updates = generator_state.apply_fn(
                {'params': gen_params,
                 'batch_stats': generator_state.batch_stats},
                x=x,
                train=True,
                mutable=["batch_stats"],
            )
            y_pred = discriminator_state.apply_fn(
                {'params': disc_params},
                x=fake_image,
                train=False,
            )
            loss = -1.0 * jax.nn.log_sigmoid(y_pred)
            return loss, gen_updates

        loss, gen_updates = jax.vmap(
            for_single_instance,
            in_axes=0,
            out_axes=(0, None),
            axis_name='batch')(noise_batched)
        return jnp.mean(loss), gen_updates

    generator_grad_fn = jax.value_and_grad(generator_forward_and_loss,
                                           has_aux=True)
    (gen_loss, gen_updates), grads = generator_grad_fn(
        generator_state.params,
        discriminator_state.params,
        noise_batched_for_gen)
    generator_state = generator_state.apply_gradients(grads=grads)
    generator_state = generator_state.replace(
        batch_stats=gen_updates['batch_stats'])
    metrics.update({'gen_loss': gen_loss})

    return generator_state, discriminator_state, metrics


batch_size = 2048
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)

mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(tf.reshape(x, (batch_size, -1)),
                          tf.float32) - 127.5) / 127.5)


def generate_and_save_image(generator_state, epoch, key):
    noise = random.normal(key, (1, latent_dim))
    fake_image = generator.apply(
        {'params': generator_state.params,
         'batch_stats': generator_state.batch_stats},
        noise,
        train=False, )
    fake_image = np.asarray(fake_image).reshape((28, 28)) * 127.5 + 127.5
    img = Image.fromarray(fake_image).convert("L")
    if not os.path.exists("./images/"):
        os.makedirs("./images/")
    img.save(f"./images/epoch_{epoch}.png")


n_epochs = 1000
n_batches = 0
gen_per_batch_losses = []
disc_per_batch_losses = []
key = random.key(4444)
for epoch in range(1, n_epochs + 1):
    batch_str = "if you're seeing this, something's wrong"
    for batch, x in enumerate(mnist_ds, start=1):
        x = x.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?"
        else:
            batch_str = f"{batch}/{n_batches}"

        key, key_2 = random.split(key, 2)
        noise_batched_for_disc = random.normal(
            key,
            shape=(x.shape[0], latent_dim))
        noise_batched_for_gen = random.normal(
            key_2,
            shape=(x.shape[0], latent_dim))
        generator_state, discriminator_state, metrics = train_step(
            generator_state=generator_state,
            discriminator_state=discriminator_state,
            x_batched=x,
            noise_batched_for_disc=noise_batched_for_disc,
            noise_batched_for_gen=noise_batched_for_gen,
        )

        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Gen Loss: {metrics['gen_loss']:.6f} | "
              f"Disc Loss: {metrics['disc_loss']:.6f}", end="")
        gen_per_batch_losses.append(metrics['gen_loss'])
        disc_per_batch_losses.append(metrics['disc_loss'])

    gen_loss = jnp.mean(jnp.asarray(gen_per_batch_losses))
    disc_loss = jnp.mean(jnp.asarray(disc_per_batch_losses))
    print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
          f"Gen Loss: {gen_loss:.6f} | "
          f"Disc Loss: {disc_loss:.6f}", end="")
    key = random.split(key, 1)[0]
    if epoch % 10 == 0:
        generate_and_save_image(generator_state, epoch, key)
    print()
