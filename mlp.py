import jax
from jax import random, numpy as jnp, tree
import flax
from flax import linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Sequence


class MLP(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(features=512)(x)
            x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.softmax(x, axis=-1)
        return x


model = MLP((256, 128, 64))
key1, key2 = random.split(random.key(1337), 2)
sample_image = random.normal(key1, (1, 784))
output, params = model.init_with_output(key2, sample_image)


batch_size = 512
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)
mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(lambda x, y: (tf.reshape(x, (batch_size, -1)),
                                      tf.one_hot(y, depth=10)))


@jax.jit
def forward_and_loss(params, x_batched, y_batched, eps=1e-15):
    def for_single_instance(x, y):
        y_pred = model.apply(params, x)
        y_pred = jnp.clip(y_pred, eps, 1. - eps)
        loss = jnp.sum(y * jnp.log(y_pred))
        return loss
    return -jnp.mean(jax.vmap(
        for_single_instance, in_axes=(0, 0))(x_batched, y_batched))


@jax.jit
def update_weights(params, grad, lr):
    return tree.map(lambda p, g: p - lr * g, params, grad)


grad_fn = jax.value_and_grad(forward_and_loss)
n_epochs = 100
n_batches = 0
lr = 0.001
per_epoch_loss = []
for epoch in range(1, n_epochs + 1):
    for batch, (x, y) in enumerate(mnist_ds, start=1):
        x = x.numpy()
        y = y.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?"
        else:
            batch_str = f"{batch}/{n_batches}"
        loss, grad = grad_fn(params, x, y)
        params = update_weights(params, grad, lr)
        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Loss: {loss:.5f}", end="")
    print()
