import jax
from jax import random, numpy as jnp, tree
import flax
from flax import linen as nn


# Generate data
n_samples = 16
x_dim = 5
y_dim = 3
key = random.key(1337)
key1, key2 = random.split(key, 2)
W = random.normal(key1, (x_dim, y_dim))
b = random.normal(key2, (y_dim,))
true_params = flax.core.freeze({'params': {'bias': b, 'kernel': W}})
key_samples, key_noise = random.split(key)
x_samples = random.normal(key_samples, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(
    key_noise, (n_samples, y_dim))


params_key = random.split(key2, 1)[0]
model = nn.Dense(features=3)
params = model.init(params_key, x_samples[0])


@jax.jit
def forward_and_loss(params, x_batched, y_batched):
    def for_single_instance(x, y):
        y_pred = model.apply(params, x)
        loss = jnp.inner(y - y_pred, y - y_pred) / 2.0
        return loss
    return jnp.mean(
        jax.vmap(for_single_instance, in_axes=(0, 0))(x_batched, y_batched))


lr = 0.01
grad_fn = jax.value_and_grad(forward_and_loss)
for epoch in range(100):
    loss, grad = grad_fn(params, x_samples, y_samples)
    params = tree.map(lambda p, g: p - lr * g, params, grad)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}")
