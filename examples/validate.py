# validation for the created dataset

import json
import os
from dex_ycb_toolkit.factory import get_dataset
osp = os.path
from tqdm import tqdm
import numpy as np
import torch
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from manopth.manolayer import ManoLayer
import pyrender
import trimesh
import cv2
import matplotlib.pyplot as plt

#read the dataset
datadir = osp.join(osp.dirname(osp.abspath(__file__)),'..','data','dexycb','test')
#datadir = osp.join(osp.dirname(osp.abspath(__file__)),'..','..','Downloads','mesh_data')

obj_mesh_dir = osp.join(datadir, 'obj_mesh')
hand_mesh_dir = osp.join(datadir, 'hand_mesh')

#idx = '3_20200820_135508_836212060125_9'
idx = 30
#load the mesh
mesh_obj = trimesh.load(osp.join(obj_mesh_dir,'{}.obj'.format(idx)))
mesh_hand = trimesh.load(osp.join(hand_mesh_dir,'{}.obj'.format(idx)))
print(len(mesh_hand.vertices))
# pyrender
scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                ambient_light=np.array([1.0, 1.0, 1.0]))

mesh_obj = pyrender.Mesh.from_trimesh(mesh_obj)
node = scene.add(mesh_obj)
mesh1 = pyrender.Mesh.from_trimesh(mesh_hand)
mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
mesh2 = pyrender.Mesh.from_trimesh(mesh_hand, wireframe=True)
mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
node1 = scene.add(mesh1)
node2 = scene.add(mesh2)

pyrender.Viewer(scene)
