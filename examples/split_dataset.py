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
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R



def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None, scale=False):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if scale:
        if bbox_scale is None:
            bbox_scale = np.linalg.norm(vertices, 2, 1).max()
        vertices = vertices / bbox_scale
    else:
        bbox_scale = 1
    return vertices, bbox_center, bbox_scale



def seal(mesh_to_seal):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal

def seal2(mesh_to_seal):
    close_face = np.array([
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ])
    mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, close_face])
    return mesh_to_seal

class dexycb():
    def __init__(self,setup,split):
        self.getdata = DexYCBDataset(setup,split)
        self.filter_no_hand = True
        self.filter_no_contact = True
        self.filter_threshold = 15
        self.use_right_hand = True
        self.split = split

    def __len__(self):
        return len(self.getdata)

    def mano_process(self,sample):
        #init mano layer
        mano_layer = ManoLayer(flat_hand_mean=False,
                    ncomps=45,
                    side=sample['mano_side'],
                    mano_root='manopth/mano/models',
                    use_pca=True)
              
        label = self.load_label(sample)
        pose_params = label['pose_m']
        obj_pose = label['pose_y'][sample['ycb_grasp_ind']]

        #TODO: jt vt from mano layer, nomalize jt and obj center

        pass


    def iterate(self):
        #sample = self.getdata[idx]
        objdir = 'data/dexycb/{}/obj_mesh/'.format(self.split)
        handdir = 'data/dexycb/{}/hand_mesh/'.format(self.split)
        metadir = 'data/dexycb/{}/meta/'.format(self.split)
        os.makedirs(objdir,exist_ok = True)
        os.makedirs(handdir,exist_ok = True)
        os.makedirs(metadir,exist_ok = True)
        count = 0
        if self.filter_no_hand and self.filter_no_contact and self.use_right_hand:
            for i,sample in tqdm(enumerate(self.getdata),total = self.__len__()):
                if sample["mano_side"] == 'left':
                    continue
                if np.all(self.get_joint2d(sample) == -1.0):
                    continue
                if cdist(self.get_obj_verts_transmed(sample),self.get_joint3d(sample)).min()*1000 > self.filter_threshold:
                    continue
                hand_mesh,obj_mesh = self.get_mesh(sample)
                
                hand_mesh.export(handdir + '{}.obj'.format(count))
                obj_mesh.export(objdir + '{}.obj'.format(count))
                json.dump(sample,open(metadir + '{}.json'.format(count),'w'))
                
                count += 1

    def iterate_shadow(self):
        #hand joints & obj center
        pass

    def viz(self):
        if self.filter_no_hand and self.filter_no_contact and self.use_right_hand:
            for i,sample in tqdm(enumerate(self.getdata),total = self.__len__()):
                if sample["mano_side"] == 'left':
                    continue
                if np.all(self.get_joint2d(sample) == -1.0):
                    continue
                if cdist(self.get_obj_verts_transmed(sample),self.get_joint3d(sample)).min()*1000 > self.filter_threshold:
                    continue
                joint,center = self.mano_process(sample)
                print(center)
                print(joint)
                self.plot(sample)

    def load_label(self,sample):
        label = np.load(sample['label_file'])
        return label

    def get_joint2d(self,sample):
        label = self.load_label(sample)
        return label['joint_2d'].squeeze(0)

    def get_joint3d(self,sample):
        label = self.load_label(sample)
        return label['joint_3d'].squeeze(0)

    def get_obj_mesh(self,obj_index):
        return trimesh.load(self.getdata.obj_file[obj_index])

    def get_obj_transm(self,sample):
        label = self.load_label(sample)
        transm = label['pose_y'][sample['ycb_grasp_ind']]
        obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
        obj_mesh = self.get_obj_mesh(obj_index)
        verts,offset,bbox_scale = center_vert_bbox(obj_mesh.vertices)
        R,t = transm[:3,:3],transm[:,3:]
        _t = R @ offset.reshape(3,1) + t
        new_transm = np.concatenate([np.concatenate([R,_t],axis=1),
                                    np.array([[0.,0.,0.,1.]],dtype=np.float32)])
        return new_transm.astype(np.float32),verts,offset

    def get_obj_verts_transmed(self,sample):
        transm,verts,_ = self.get_obj_transm(sample)
        R,t = transm[:3,:3],transm[:3,[3]]
        verts = (R @ verts.T + t).T
        return verts

    def get_mesh(self,sample):
        label = self.load_label(sample)

        obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
        obj_mesh = self.get_obj_mesh(obj_index)
        pose_obj = label['pose_y'][sample['ycb_grasp_ind']]
        pose_obj = np.vstack((pose_obj, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        pose_obj[1] *= -1
        pose_obj[2] *= -1
        transformed_mesh_obj = obj_mesh.apply_transform(pose_obj)
        transformed_mesh_obj.visual = trimesh.visual.ColorVisuals()
        transformed_mesh_obj = self.mesh_template(transformed_mesh_obj)

        mano_layer = ManoLayer(flat_hand_mean=False,
                    ncomps=45,
                    side=sample['mano_side'],
                    mano_root='manopth/mano/models',
                    use_pca=True)
        faces = mano_layer.th_faces.numpy()
        betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        pose_m = label['pose_m']
        pose = torch.from_numpy(pose_m)
        vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
        vert /= 1000
        vert = vert.view(778, 3)
        vert = vert.numpy()
        vert[:, 1] *= -1
        vert[:, 2] *= -1
        mesh_hand = trimesh.Trimesh(vertices=vert, faces=faces)
        seal_mesh_hand = seal(mesh_hand)
        return seal_mesh_hand,transformed_mesh_obj

    def mesh_template(self,mesh):
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return new_mesh

    def get_camera(self,sample,scene):
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        scene.add(cam, pose=np.eye(4))
        return None
    
    def opencv_camera(self,sample):
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0,  1 ]])
        rotation = np.eye(3)
        translation = np.zeros(3)
        points = self.load_label(sample)['pose_y'][sample['ycb_grasp_ind']][:3,3]
        points = points.reshape(3,1)
        points_2d = cv2.projectPoints(points,rotation,translation,intrinsics,None)[0]
        print(points_2d)
        return points_2d
    
    def _test_mano_process(self,sample):
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0,  1 ]])
        rotation = np.eye(3)
        translation = np.zeros(3)
        joint,points = self.mano_process(sample)





        joint_2d = cv2.projectPoints(joint,rotation,translation,intrinsics,None)[0]
        points_2d = cv2.projectPoints(points,rotation,translation,intrinsics,None)[0]
        return joint_2d,points_2d      
    
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
        im_real = cv2.imread(idx['color_file'])
        im_real = im_real[:, :, ::-1]
        im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
        im = im.astype(np.uint8)

        print('Close the window to continue.')
        point = self.opencv_camera(idx)
        cv2.circle(im,(int(point[0,0,0]),int(point[0,0,1])),5,(0,0,255),-1)
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
            mesh_hand.export('data/dexycb/train/hand_mesh/{}.obj'.format(count))

            mesh_obj.export('data/dexycb/train/obj_mesh/{}.obj'.format(count))
            json.dump(meta,open('data/dexycb/train/label/{}.json'.format(count),'w'))
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
    #creat_dataset()
    # dex_train = dexycb('s1', 'train')
    # dex_train.iterate()
    
    # dex_test = dexycb('s1', 'test')
    # dex_test.iterate()
    
    dex_train = dexycb('s1', 'train')
    label = dex_train.load_label(dex_train.getdata[500])
    print(dex_train.getdata[500])
    print(label["pose_m"])
    print(label["joint_3d"])

    #dex_train.viz()
    #dex_train.opencv_camera(dex_train.getdata[0])


