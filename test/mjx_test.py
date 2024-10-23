import mujoco
from mujoco import mjx
import mujoco_viewer

import jax
from jax import numpy as jp
import numpy as np

# %env MUJOCO_GL=egl

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path("mujoco_models/Acrobot/acrobot.xml")
mj_data = mujoco.MjData(mj_model)
# renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

for _ in range(1000):
    if viewer.is_alive:
        mujoco.mj_step(mj_model, mj_data)
        viewer.render()
    else:
        break

viewer.close()