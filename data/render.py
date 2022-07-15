"""
This script is borrowed from https://github.com/mkocabas/VIBE
Adhere to their licence to use this script
It has been modified
"""

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
import os


os.environ['PYOPENGL_PLATFORM'] = 'egl'

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(
        self, 
        scale,
        translation,
        znear=pyrender.camera.DEFAULT_Z_NEAR,
        zfar=None,
        name=None
        ):
        
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, faces=None, resolution=(224, 224), bg_color=[0, 0, 0, 0.5], orig_img=False, wireframe=False):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose.copy())

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], transparency=False):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        if not transparency:
            valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        else:
            output_img = rgb
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def get_renderer(width, height, faces):
    renderer = Renderer(resolution=(width, height),
        bg_color=[1, 1, 1, 0.5],
        orig_img=False,
        faces=faces,
        wireframe=False)
    return renderer