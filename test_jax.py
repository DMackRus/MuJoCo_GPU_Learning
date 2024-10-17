# import jax
# import jax.numpy as jnp
# print(jax.devices())
# x_gpu = jax.device_put(jnp.array([1.0, 2.0, 3.0]), device=jax.devices()[0])
# # y_gpu = x_gpu * 2
# # print(y_gpu)

import jax
import jax.numpy as jnp

x = jnp.ones((1000, 1000))
x_gpu = jax.device_put(x, device=jax.devices()[0])

print(x_gpu.device_buffer.device())  # This should print "gpu:0" if the GPU is used
