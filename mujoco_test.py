import mujoco
import mujoco_viewer

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

# %env MUJOCO_GL=egl

# # Make model, data, and renderer
# mj_model = mujoco.MjModel.from_xml_string(xml)
# mj_data = mujoco.MjData(mj_model)
# renderer = mujoco.Renderer(mj_model)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
# with mujoco.Renderer(model) as renderer:
#   media.show_image(renderer.render())

viewer = mujoco_viewer.MujocoViewer(model, data)

for _ in range(1000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()


# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         # viewer.sync()
#         viewer.render()


# with mujoco.viewer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data)

#     media.show_image(renderer.render())