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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from manopth.manolayer import ManoLayer
import pyrender
import trimesh
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def display(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for m in mesh:
        mh = Poly3DCollection(m.vertices[m.faces], alpha=0.1)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mh.set_edgecolor(edge_color)
        mh.set_facecolor(face_color)
        ax.add_collection3d(mh)
    cam_equal_aspect_3d(ax = ax, verts = np.array(mesh[0].vertices))

    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Plot the x, y, and z axes
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r', label='X')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g', label='Y')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b', label='Z')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()   

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
        hand_translation = pose_params[:,48:51]
        obj_pose = label['pose_y'][sample['ycb_grasp_ind']]
        obj_center = obj_pose[:3,3]

        #get mano verts
        betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        #set tranlation to zero
        tranlation = torch.zeros_like(hand_translation)
        vert, joint, root_trans = mano_layer(pose_params[:, 0:48], betas, tranlation)

        #transform joints to align with robot
        ones = torch.ones(1,21,1)
        rot_mat = torch.tensor([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0],[0, 0, 0, 1]],dtype=torch.float32)

        joint = torch.cat([joint,ones],dim = 2)
        jt = torch.transpose(torch.matmul(torch.linalg.inv(root_trans), torch.transpose(joint,1,2)),1,2)
        jt = torch.transpose(torch.matmul(rot_mat, torch.transpose(jt,1,2)),1,2)
        jt = jt[:,:,:3]

        #transform obj center 
        obj_center = obj_center - hand_translation
        obj_center = torch.cat([obj_center,torch.ones(1,1)],dim = 1)
        oc = torch.transpose(torch.matmul(torch.linalg.inv(root_trans), torch.transpose(obj_center,1,2)),1,2)
        oc = torch.transpose(torch.matmul(rot_mat, torch.transpose(oc,1,2)),1,2)

        return jt.squeeze(0).numpy(),oc.squeeze(0).numpy()

    def iterate(self):
        #sample = self.getdata[idx]
        hand_dir = 'data/dexycb/{}/hand_obj/'.format(self.split)

        os.makedirs(hand_dir,exist_ok = True)

        count = 0
        if self.filter_no_hand and self.filter_no_contact and self.use_right_hand:
            for i,sample in tqdm(enumerate(self.getdata),total = self.__len__()):
                if sample["mano_side"] == 'left':
                    continue
                if np.all(self.get_joint2d(sample) == -1.0):
                    continue
                if cdist(self.get_obj_verts_transmed(sample),self.get_joint3d(sample)).min()*1000 > self.filter_threshold:
                    continue
                joints,obj_center = self.mano_process(sample)
                data_dict = {"joints":joints,"obj_center":obj_center}
                np.save(osp.join(hand_dir,'{}.npz'.format(count)),data_dict)
                count += 1

    def viz(self,sample):
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                ambient_light=np.array([1.0, 1.0, 1.0]))
        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        scene.add(cam, pose=np.eye(4))   


        label = self.load_label(sample)
        obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
        obj_mesh = self.get_obj_mesh(obj_index)
        
        
        obj_pose = label['pose_y'][sample['ycb_grasp_ind']]
        obj_pose = np.vstack([obj_pose,np.array([0,0,0,1])])
        # obj_pose[1] *= -1
        # obj_pose[2] *= -1
        obj_mesh.apply_transform(obj_pose)

        # obj_mesh_transed = obj_mesh.apply_transform(obj_pose)

        # obj_mesh_transed.visual = trimesh.visual.ColorVisuals()
        # obj_mesh_transed = trimesh.Trimesh(vertices=obj_mesh_transed.vertices, faces=obj_mesh_transed.faces)

        mano_layer = ManoLayer(flat_hand_mean=False,
            ncomps=45,
            side=sample['mano_side'],
            mano_root='manopth/mano/models',
            use_pca=True)
        faces = mano_layer.th_faces.numpy()
        betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        pose_params = torch.from_numpy(label['pose_m'])
        hand_translation = pose_params[:,48:51]
        tranlation = torch.zeros_like(hand_translation)
        verts,joints,trans = mano_layer(pose_params[:, 0:48], betas, hand_translation)
        verts /= 1000
        verts = verts.view(778, 3)
        print(hand_translation.size())
        verts = verts - hand_translation[0]
        obj_mesh.apply_translation(hand_translation[0]*-1)
        
        verts = verts.numpy()
        # verts[:, 1] *= -1
        # verts[:, 2] *= -1


        hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        hand_mesh_sealed = seal(hand_mesh)

        

        #trans[0,:3,3] += hand_translation[0]        
   
        trans_inv = torch.linalg.inv(trans.squeeze(0))
        trans_inv[1] *= -1
        trans_inv[2] *= -1
        print(trans_inv)

        hand_mesh_sealed.apply_transform(trans_inv.numpy())
        obj_mesh.apply_transform(trans_inv.numpy())
        obj_mesh.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],[0, 0, 0, 1]]))
        hand_mesh_sealed.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],[0, 0, 0, 1]]))
        verts_hand = hand_mesh_sealed.vertices
        verts_obj = obj_mesh.vertices

        # verts_hand[:, 1] *= -1
        # verts_hand[:, 2] *= -1
        # verts_obj[:, 1] *= -1
        # verts_obj[:, 2] *= -1

        display([hand_mesh_sealed,obj_mesh])

        mesh1 = pyrender.Mesh.from_trimesh(hand_mesh_sealed)
        mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
        mesh2 = pyrender.Mesh.from_trimesh(hand_mesh_sealed, wireframe=True)
        mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
        obj_mesh = pyrender.Mesh.from_trimesh(obj_mesh)

        print(hand_mesh_sealed.vertices.min(0))
        scene.add(obj_mesh)

        node1 = scene.add(mesh1)
        node2 = scene.add(mesh2)
        # obj_mesh_transed = obj_mesh_transed.apply_translation((hand_translation.squeeze(0)))
        # obj_mesh_transed = obj_mesh_transed.apply_transform(np.linalg.inv(trans.squeeze(0)))
        # obj_mesh_transed = obj_mesh_transed.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0],[0, 0, 0, 1]]))

        # hand_mesh_sealed = hand_mesh_sealed.apply_transform(np.linalg.inv(trans.squeeze(0)))
        # hand_mesh_sealed = hand_mesh_sealed.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0],[0, 0, 0, 1]]))

        #print(hand_mesh_sealed.vertices.max(0))

        # print(obj_mesh_transed.vertices.max(0))

     
        #obj_mesh_pr.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]

        #hand_mesh_pr.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
        
        #scene.add(obj_mesh_pr)
        #scene.add(hand_mesh_pr)


        # render color image
        # r = pyrender.OffscreenRenderer(self.getdata.w, self.getdata.h)
        # im_render, _ = r.render(scene)
        # im_real = cv2.imread(sample['color_file'])
        # im_real = im_real[:, :, ::-1]
        # im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
        # im = im.astype(np.uint8)
        # print('Close the window to continue.')
        # plt.imshow(im)
        # plt.tight_layout()
        # plt.show()       




        #pyrender.Viewer(scene)








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
    
    def get_label(self,idx):
        label = np.load(self.getdata[idx]['label_file'])
        pose_y = label['pose_y']
        pose_m = label['pose_m']

        return pose_y,pose_m
    
    def get_meta(self,idx):
        meta_info = self.getdata[idx]
        return meta_info


if __name__ == '__main__':
    #creat_dataset()
    # dex_train = dexycb('s1', 'train')
    # dex_train.iterate()
    
    # dex_test = dexycb('s1', 'test')
    # dex_test.iterate()
    
    dex_train = dexycb('s1', 'train')
    dex_train.viz(dex_train.getdata[1000])
    #dex_train.viz()
    #dex_train.opencv_camera(dex_train.getdata[0])


