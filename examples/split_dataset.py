# Create and save a dataset
# Based on create_dataset.py
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



class dexycb():
    def __init__(self,setup,split):
        self.getdata = DexYCBDataset(setup,split)
        self.filter_no_hand = True
        self.filter_no_contact = True
        self.filter_threshold = 50
        self.filter_hand_side = 'right'


    def __len__(self):
        return len(self.getdata)
    
    def get_camera(self,idx,scene):
        fx = self.getdata[idx]['intrinsics']['fx']
        fy = self.getdata[idx]['intrinsics']['fy']
        cx = self.getdata[idx]['intrinsics']['ppx']
        cy = self.getdata[idx]['intrinsics']['ppy']
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        scene.add(cam, pose=np.eye(4))
        return None
    
    def get_label(self,idx):
        label = np.load(self.getdata[idx]['label_file'])
        pose_y = label['pose_y']
        pose_m = label['pose_m']

        return pose_y,pose_m
    
    def get_meta(self,idx):
        meta_info = self.getdata[idx]
        return meta_info

    # deprected
    def _get_mesh(self,idx):
        #get pose label
        pose_y,pose_m = self.get_label(idx)
        
        # grasping object mesh
        obj_id = self.getdata[idx]['ycb_ids'][self.getdata[idx]["ycb_grasp_ind"]]
        pose_y = pose_y[self.getdata[idx]["ycb_grasp_ind"]]
        mesh_obj = trimesh.load(self.getdata.obj_file[obj_id])
        pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        pose[1] *= -1
        pose[2] *= -1


        # transform to camera coordinate
        #transformed_vertices = np.dot(mesh_obj.vertices, pose[:3, :3].T) + pose[:3, 3]
        #homogeneous_coords = np.hstack((mesh_obj.visual.uv, np.ones((mesh_obj.visual.uv.shape[0], 1))))
        #transformed_texture_coords = np.dot(homogeneous_coords, pose[:3, :3].T)
        #transformed_mesh_obj = trimesh.Trimesh(transformed_vertices, mesh_obj.faces,visual = trimesh.visual.TextureVisuals(uv=transformed_texture_coords[:,:2]))
        transformed_mesh_obj = mesh_obj.apply_transform(pose)


        #hand mesh
        mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=self.getdata[idx]['mano_side'],
                         mano_root='manopth/mano/models',
                         use_pca=True)
        faces = mano_layer.th_faces.numpy()
        betas = torch.tensor(self.getdata[idx]['mano_betas'], dtype=torch.float32).unsqueeze(0)
        
        # Add MANO meshes.
        if not np.all(pose_m == 0.0):
            pose = torch.from_numpy(pose_m)
            vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
            vert /= 1000
            vert = vert.view(778, 3)
            vert = vert.numpy()
            vert[:, 1] *= -1
            vert[:, 2] *= -1
            mesh_hand = trimesh.Trimesh(vertices=vert, faces=faces)


        return transformed_mesh_obj,mesh_hand
    
    def add_scene(self,idx):
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                ambient_light=np.array([1.0, 1.0, 1.0]))
        # get camera
        self.get_camera(idx,scene)
        # get mesh
        mesh_obj,mesh_hand = self.get_mesh(idx)
        # add mesh to scene
        mesh_obj = pyrender.Mesh.from_trimesh(mesh_obj)
        node = scene.add(mesh_obj)
        mesh1 = pyrender.Mesh.from_trimesh(mesh_hand)
        mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
        mesh2 = pyrender.Mesh.from_trimesh(mesh_hand, wireframe=True)
        mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
        node1 = scene.add(mesh1)
        node2 = scene.add(mesh2)
        return scene

    def plot(self,idx):
        render = pyrender.OffscreenRenderer(self.getdata.w, self.getdata.h) 

        scene_im = self.add_scene(idx)
        scene_obj = self.add_scene(idx)

        # render image
        im_render, _ = render.render(scene_im)
        im_real = cv2.imread(self.getdata[idx]['color_file'])
        im_real = im_real[:, :, ::-1]
        im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
        im = im.astype(np.uint8)

        print('Close the window to continue.')

        plt.imshow(im)
        plt.tight_layout()
        plt.show()

        print('Visualizing pose using pyrender 3D viewer')

        # render object
        pyrender.Viewer(scene_obj)

def creat_dataset():
    # train data
    dex = dexycb('s1', 'train')

    print('Dataset length: ',dex.__len__())
    #os.makedirs('data/dexycb/train',exist_ok = True)
    os.makedirs('data/dexycb/train/obj_mesh/',exist_ok = True)
    os.makedirs('data/dexycb/train/hand_mesh/',exist_ok = True)
    os.makedirs('data/dexycb/train/label/',exist_ok = True)
    count = 0
    for idx,_ in tqdm(enumerate(range(dex.__len__())),total = dex.__len__()):
        try:
            #pose_y,pose_m = dex.get_label(idx)
            meta = dex.get_meta(idx)
            mesh_obj,mesh_hand = dex.get_mesh(idx)
            # print('data/dexycb/test/obj_mesh/{}.obj'.format(count))
            mesh_obj.visual = trimesh.visual.ColorVisuals()
            mesh_hand.export('data/dexycb/train/hand_mesh/{}.obj'.format(idx))

            mesh_obj.export('data/dexycb/train/obj_mesh/{}.obj'.format(idx))
            json.dump(meta,open('data/dexycb/train/label/{}.json'.format(idx),'w'))
            #np.save('data/dexycb/train/label/{}.npy'.format(count),np.vstack((pose_y,pose_m)))
            
            count = count + 1
        except Exception as e:
            print(e)
            continue


    # test data
    # dex = dexycb('s1', 'test')
    # print('Dataset length: ',dex.__len__())
    # os.makedirs('data/dexycb/test',exist_ok = True)
    # os.makedirs('data/dexycb/test/obj_mesh',exist_ok = True)
    # os.makedirs('data/dexycb/test/hand_mesh',exist_ok = True)
    # os.makedirs('data/dexycb/test/label',exist_ok = True)
    # count = 0
    # for idx,_ in tqdm(enumerate(range(dex.__len__())),total = dex.__len__()):
    #     try:
    #         pose_y,pose_m = dex.get_label(idx)
    #         mesh_obj,mesh_hand = dex.get_mesh(idx)
    #         # mesh_obj.export('data/dexycb/test/obj_mesh/{}.obj'.format(count))
    #         # mesh_hand.export('data/dexycb/test/hand_mesh/{}.obj'.format(count))
    #         # np.save('data/dexycb/test/label/{}.npy'.format(count),np.vstack((pose_y,pose_m)))
    #         count = count + 1
    #     except:
    #         continue

if __name__ == '__main__':
    creat_dataset()
    #dex = dexycb('s1', 'train')
    #dex.plot(50)
    

