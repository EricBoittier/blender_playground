import blender_plots as bplt
import MolecularNodes as mn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bpy
from pathlib import Path

data_path = Path("/Users/ericboittier/Documents/github/pythonProject/psi4/water0")

#  read data
# grid = np.genfromtxt(data_path / "ingrid.dat")
# grid = grid.astype(np.float32)
#
# #  read esp
# esp = np.genfromtxt(data_path / "ingridesp.dat")
#
# print(esp.shape)
# print(grid.shape)
# print(esp.min(), esp.max())
# print(grid.min(), grid.max())
# print(esp)
# print(grid)
#
# stride: int = 10
# esp = esp[::stride]
# grid = grid[::stride]

# convert esp to colors using red to blue colormap
# cmap = cm.get_cmap('bwr')
# norm = plt.Normalize(vmin=esp.min(), vmax=esp.max())
# norm = plt.Normalize(vmin=-0.01, vmax=0.01)
# colors = cmap(norm(esp))
# colors = colors[:, :3]

def remove_all():
#  remove old objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

remove_all()

# #  add points
# bplt.Scatter(
#         grid,
#         color=colors,
#         marker_type="uv_spheres",
#         radius=0.1235,
#         name="points",
#     )
#
# style = "ball_and_stick"
# pdb_path = "/Users/ericboittier/Documents/github/pythonProject/cubes/water_pbe0/water0.pdb"
# mn.load.molecule_local(pdb_path, default_style=0)
#
#
# #  scale molecule
# bpy.ops.object.select_all(action="DESELECT")
# bpy.data.objects["Name"].select_set(True)
# bpy.ops.transform.resize(value=(10, 10, 10))


def add_camera(orthoscale=2):
    #  add camera
    bpy.ops.object.camera_add(
            enter_editmode=False,
            align="VIEW",
            location=(0, 0, 0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )
    # set camera to orthographic
    bpy.data.cameras["Camera"].type = "ORTHO"
    #  set camera to orthographic scale
    bpy.data.cameras["Camera"].ortho_scale = orthoscale
    #  make active camera
    bpy.context.scene.camera = bpy.data.objects["Camera"]


# add_camera()

#  set white background
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
        1,
        1,
        1,
        1,
    )


bpy.context.scene.render.resolution_x = 1256  # perhaps set resolution in code
bpy.context.scene.render.resolution_y = 1256
bpy.context.scene.render.engine = "CYCLES"

def camera_to_points(name=None):
    #  set camera to fit all points
    bpy.ops.object.select_all(action="DESELECT")
    if name is None:
        # select all objects
        bpy.ops.object.select_all(action="SELECT")
    else:
        # select specific object
        bpy.data.objects[name].select_set(True)
    bpy.ops.view3d.camera_to_view_selected()
    #  set camera to orthographic
    bpy.data.cameras["Camera"].type = "ORTHO"

# camera_to_points()


def render(frame):
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(f"/Users/ericboittier/Downloads/test-{frame}.png")

def rot():
    # create a rotation animation for 360 degrees
    bpy.context.scene.frame_end = 360
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["points"].select_set(True)
    bpy.data.objects["Camera"].select_set(True)
    bpy.data.objects["Name"].select_set(True)
    bpy.ops.anim.keyframe_insert_menu(type="Rotation")
    bpy.ops.anim.keyframe_insert_menu(type="Location")
    # loop to perform 360 degree rotation
    for i in range(0, 360, 30):
        bpy.context.scene.frame_set(i)
        bpy.data.objects["points"].rotation_euler[2] = np.radians(i)
        bpy.data.objects["Camera"].rotation_euler[2] = np.radians(i)
        bpy.data.objects["Name"].rotation_euler[2] = np.radians(i)
        bpy.ops.anim.keyframe_insert_menu(type="Rotation")
        bpy.ops.anim.keyframe_insert_menu(type="Location")

