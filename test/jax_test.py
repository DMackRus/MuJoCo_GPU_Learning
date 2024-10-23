import jax
import jax.numpy as jnp


# Simple test script to ensure that jax devices cuda is listed and that we can perform a simple gpu computation
print(jax.devices())
x_gpu = jax.device_put(jnp.array([1.0, 2.0, 3.0]), device=jax.devices()[0])
y_gpu = x_gpu * 2
print(y_gpu)