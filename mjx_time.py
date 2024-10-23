import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
from jax import numpy as jp
import numpy as np

import time

import concurrent.futures
import multiprocessing

import matplotlib.pyplot as plt

import math

def simulate_step(model, data, num_steps):
    #mj_data = mujoco.MjData(model)  # Create mjData inside the function for each process
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    return data.qpos 

def simulate_GPU(mj_model, mj_data, total_steps, batch_size):

    print(f"Start of GPU parallel test for batch size {batch_size}")

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    num_batch_steps = math.ceil(total_steps / batch_size)
    print(f'num batch steps: {num_batch_steps}')

    time_start = time.time()
    for i in range(num_batch_steps):
        batch = jit_step(mjx_model, batch)
    time_gpu = time.time() - time_start

    return time_gpu

def time_model(mj_model, total_steps, gpu_batch_sizes):

    time_cpu_serial = 0
    time_cpu_parallel = 0
    time_gpu = []

    # ------------------------------- General setup -----------------------------------------
    num_cores = multiprocessing.cpu_count()

    mj_data = mujoco.MjData(mj_model)

    mj_data_instances = [mujoco.MjData(mj_model) for _ in range(num_cores)]

    print(f'Num total steps {total_steps}')
    print(f'num cores: {num_cores}')

    num_cpu_steps = math.ceil(total_steps / num_cores)
    print(f'Num CPU Steps: {num_cpu_steps}')

    # -------------------------------- CPU serial test ---------------------------------------
    print("Start of CPU serial test")
    time_start = time.time()
    for i in range(total_steps):
        mujoco.mj_step(mj_model, mj_data)
    time_cpu_serial = time.time() - time_start

    # ------------------------------- CPU parallel test --------------------------------------
    print("Start of CPU parallel test")
    multiprocessing.set_start_method('spawn', force=True)

    time_start = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(simulate_step, [(mj_model, mj_data_instances[_], num_cpu_steps) for _ in range(num_cores)])
    time_cpu_parallel = time.time() - time_start

    # ------------------------------- GPU parallel test --------------------------------------

    # For GPU we loop through differen batch sizes and test
    for batch_size in gpu_batch_sizes:
        time_gpu.append(simulate_GPU(mj_model, mj_data, total_steps, batch_size))

    return time_cpu_serial, time_cpu_parallel, time_gpu

def create_plot(inputs, times_cpu_serial, times_cpu_parallel, times_gpu, batch_sizes):
    print("begin plotting script")
    plt.figure(figsize=(10, 6))

    print(times_gpu)

    plt.plot(inputs, times_cpu_serial, label='Time CPU (s)', marker = 'o')
    plt.plot(inputs, times_cpu_parallel, label='Time CPU Parallel (s)', marker='o')

    for i, timing_list in enumerate(times_gpu):
        plt.plot(inputs, timing_list, label=f'Time GPU - Batch size: {batch_sizes[i]} (s)', marker='x')

    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.xlabel('Number mj_steps')
    plt.ylabel('Time (s)')
    plt.title('Timing results')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('timing_results.png')

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

    num_cores = multiprocessing.cpu_count()

    inputs = np.logspace(5, 9, num = 8)
    inputs = [int(x) for x in inputs]

    mj_model = mujoco.MjModel.from_xml_string(xml)

    batch_sizes = [1024, 2048, 4096]

    time_cpu_serial = []
    time_cpu_parallel = []
    time_gpu_parallel = [[] for _ in range(len(batch_sizes))] 

    for size in inputs:
        t_cpu_s, t_cpu_p, t_gpu = time_model(mj_model, size, batch_sizes)
        time_cpu_serial.append(t_cpu_s)
        time_cpu_parallel.append(t_cpu_p)

        for i, t in enumerate(t_gpu):
            time_gpu_parallel[i].append(t)

    create_plot(inputs, time_cpu_serial, time_cpu_parallel, time_gpu_parallel, batch_sizes)


    # print(f'time cpu serial: {time_cpu_serial}, time cpu parallel {time_cpu_parallel} and time_gpu: {time_gpu}')
