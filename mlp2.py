import jax
from jax import random, numpy as jnp, tree
import flax
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Sequence, Any
from functools import partial


class MLP(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x, train: bool):
        for dim in self.hidden_dims:
            x = nn.Dense(features=512)(x)
            x = nn.BatchNorm(
                use_running_average=not train,
                axis_name='batch')(x)
            x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class TrainState(train_state.TrainState):
    batch_stats: Any


model = MLP((256, 128, 64))
key1, key2 = random.split(random.key(1337), 2)
sample_image = random.normal(key1, (1, 784))
variables = model.init(key2, sample_image, train=False)


batch_size = 512
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)
mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(lambda x, y: (tf.reshape(x, (batch_size, -1)),
                                      tf.one_hot(y, depth=10)))


@partial(jax.jit, static_argnums=3)
def train_step(state: TrainState, x_batched, y_batched, epsilon=1e-15):
    def forward_and_loss(params):
        def for_single_instance(x, y):
            y_pred, non_trainable_updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, x,
                train=True, mutable=['batch_stats'])
            y_pred = jnp.clip(y_pred, epsilon, 1. - epsilon)
            loss = optax.softmax_cross_entropy(y_pred, y)
            return loss, non_trainable_updates
        losses, non_trainable_updates = jax.vmap(
            for_single_instance,
            in_axes=(0, 0),
            out_axes=(0, None),
            axis_name='batch')(x_batched, y_batched)
        return jnp.mean(losses), non_trainable_updates
    grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
    (loss, non_trainable_updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        batch_stats=non_trainable_updates['batch_stats'])
    metrics = {'loss': loss}
    return state, metrics


lr = 0.001
tx = optax.sgd(learning_rate=lr)
state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx,
    batch_stats=variables['batch_stats']
)

n_epochs = 100
n_batches = 0
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
        state, metrics = train_step(state, x, y)
        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Loss: {metrics['loss']:.5f}", end="")
    print()
