import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
from jax import numpy as jp
import numpy as np

import time

import concurrent.futures
import multiprocessing

def main():
    # mj_model = mujoco.MjModel.from_xml_path("mujoco_models/Acrobot/acrobot.xml")
    
    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 4096)
    batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

    # Creates a JIT (Just in time) compilation of the function using JAX.
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    # batch = jit_step(mjx_model, batch)

    print(batch.qpos)
    print(batch.qpos.size)

def simulate_step(model, mj_data, num_steps):
    for _ in range(num_steps):
        mujoco.mj_step(model, mj_data)

    return mj_data.qpos

def time_model(mj_model, total_steps):

    time_cpu_serial = 0
    time_cpu_parallel = 0
    time_gpu = 0

    # ------------------------------- General setup -----------------------------------------
    num_cores = multiprocessing.cpu_count()

    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(num_cores)]

    batch_size = 4096
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    print(f'Num total steps {total_steps}')
    print(f'num cores: {num_cores}')

    num_cpu_steps = int(total_steps / num_cores)
    print(f'Num CPU Steps: {num_cpu_steps}')

    num_batch_steps = int(total_steps / batch_size)
    print(f'num batch steps: {num_batch_steps}')
    
    # -------------------------------- CPU serial test ---------------------------------------
    print("Start of CPU serial test")
    time_start = time.time()
    for i in range(total_steps):
        mujoco.mj_step(mj_model, mj_data)
    time_cpu_serial = time.time() - time_start

    # ------------------------------- CPU parallel test --------------------------------------
    print("Start of CPU parallel test")

    def parallel_cpu_simulation(mj_data):
        return simulate_step(mj_model, mj_data, num_cpu_steps)

    # Execute our CPU parallelised mj_steps
    time_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(parallel_cpu_simulation, mj_data_instances))
    time_cpu_parallel = time.time() - time_start

    # ------------------------------- GPU parallel test --------------------------------------

    time_start = time.time()
    for i in range(num_batch_steps):
        batch = jit_step(mjx_model, batch)
    time_gpu = time.time() - time_start

    return time_cpu_serial, time_cpu_parallel, time_gpu

if __name__ == "__main__":

    xml = """
    <mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
    </worldbody>
    </mujoco>
    """

    mj_model = mujoco.MjModel.from_xml_string(xml)

    time_cpu_serial, time_cpu_parallel, time_gpu = time_model(mj_model, 4096*1000)

    print(f'time cpu serial: {time_cpu_serial}, time cpu parallel {time_cpu_parallel} and time_gpu: {time_gpu}')
