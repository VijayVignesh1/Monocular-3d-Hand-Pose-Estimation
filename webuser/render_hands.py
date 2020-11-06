import pickle as pkl
import pandas as pd
import torch
import torch.nn.functional as F
from collections import namedtuple
import trimesh
import numpy as np
import glob
from PIL import Image
from io import BytesIO
import cv2
import pyrender
from webuser.smpl_handpca_wrapper_HAND_only import load_model

class Render():
    def blend_shapes(self, betas, shape_disps):
        ''' Calculates the per vertex displacement due to the blend shapes


        Parameters
        ----------
        betas : torch.tensor Bx(num_betas)
            Blend shape coefficients
        shape_disps: torch.tensor Vx3x(num_betas)
            Blend shapes

        Returns
        -------
        torch.tensor BxVx3
            The per-vertex displacement due to shape deformation
        '''

        # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
        # i.e. Multiply each shape displacement by its corresponding beta and
        # then sum them.
        blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
        return blend_shape

    def vertices2joints(self, J_regressor, vertices):
        ''' Calculates the 3D joint locations from the vertices

        Parameters
        ----------
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from the
            position of the vertices
        vertices : torch.tensor BxVx3
            The tensor of mesh vertices

        Returns
        -------
        torch.tensor BxJx3
            The location of the joints
        '''

        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

    def batch_rodrigues(self, rot_vecs, epsilon=1e-8, dtype=torch.float32):
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''

        batch_size = rot_vecs.shape[0]
        device = rot_vecs.device

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rot_mat

    def transform_mat(self, R, t):
        ''' Creates a batch of transformation matrices
            Args:
                - R: Bx3x3 array of a batch of rotation matrices
                - t: Bx3x1 array of a batch of translation vectors
            Returns:
                - T: Bx4x4 Transformation matrix
        '''
        # No padding left or right, only add an extra row
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                        F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


    def batch_rigid_transform(self, rot_mats, joints, parents, dtype=torch.float32):
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """

        joints = torch.unsqueeze(joints, dim=-1)
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = self.transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms

    def lbs(self, betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
            lbs_weights, pose2rot=True, dtype=torch.float32):
        ''' Performs Linear Blend Skinning with the given shape and pose parameters

            Parameters
            ----------
            betas : torch.tensor BxNB
                The tensor of shape parameters
            pose : torch.tensor Bx(J + 1) * 3
                The pose parameters in axis-angle format
            v_template torch.tensor BxVx3
                The template mesh that will be deformed
            shapedirs : torch.tensor 1xNB
                The tensor of PCA shape displacements
            posedirs : torch.tensor Px(V * 3)
                The pose PCA coefficients
            J_regressor : torch.tensor JxV
                The regressor array that is used to calculate the joints from
                the position of the vertices
            parents: torch.tensor J
                The array that describes the kinematic tree for the model
            lbs_weights: torch.tensor N x V x (J + 1)
                The linear blend skinning weights that represent how much the
                rotation matrix of each part affects each vertex
            pose2rot: bool, optional
                Flag on whether to convert the input pose tensor to rotation
                matrices. The default value is True. If False, then the pose tensor
                should already contain rotation matrices and have a size of
                Bx(J + 1)x9
            dtype: torch.dtype, optional

            Returns
            -------
            verts: torch.tensor BxVx3
                The vertices of the mesh after applying the shape and pose
                displacements.
            joints: torch.tensor BxJx3
                The joints of the model
        '''

        batch_size = max(betas.shape[0], pose.shape[0])
        device = betas.device

        # Add shape contribution
        v_shaped = v_template + self.blend_shapes(betas, shapedirs)

        # Get the joints
        # NxJx3 array
        J = self.vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)

        if pose2rot:
            rot_mats = self.batch_rodrigues(
                pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(pose_feature, posedirs) \
                .view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)


        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        J_transformed, A = self.batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        return verts, J_transformed

    def renderer(self,hand_pose):

        m = load_model('models/MANO_RIGHT.pkl', ncomps=45, flat_hand_mean=False)


        betas=torch.zeros((1,10)).cuda()
        pose=torch.zeros((1,16*3)).cuda()


        # pose[0][:]=torch.FloatTensor([[-0.9076, -2.1418,  1.3428, -0.0209, -0.0530, -0.0746, -0.0030, -0.0256,
        #         0.2309,  0.0129, -0.0453,  0.0233,  0.0684,  0.0022,  0.1944, -0.1247,
        #         -0.0996, -0.1648,  0.0471,  0.0249, -0.1274, -0.0260, -0.0078,  0.2745,
        #         -0.0534, -0.0938, -0.0986,  0.0842,  0.1186, -0.1396,  0.1119,  0.0065,
        #         0.1549, -0.1469, -0.0863, -0.0340, -0.0132,  0.1406, -0.0166, -0.0618,
        #         0.0674, -0.2518, -0.2383, -0.0024,  0.1387,  0.0155, -0.0763,  0.0194]]).cuda()
        pose[0][:]=hand_pose

        pose[0][3:]+=torch.FloatTensor([ 0.11167872, -0.04289217,  0.41644184,  0.10881133,  0.06598568,
                0.75622001, -0.09639297,  0.09091566,  0.18845929, -0.11809504,
            -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.70485714,
            -0.01918292,  0.09233685,  0.33791352, -0.45703298,  0.19628395,
                0.62545753, -0.21465238,  0.06599829,  0.50689421, -0.36972436,
                0.06034463,  0.07949023, -0.14186969,  0.08585263,  0.63552826,
            -0.30334159,  0.05788098,  0.63138921, -0.17612089,  0.13209308,
                0.37335458,  0.85096428, -0.27692274,  0.09154807, -0.49983944,
            -0.02655647, -0.05288088,  0.53555915, -0.04596104,  0.27735802]).cuda()

        v_template=torch.from_numpy(np.array(m.v_template,dtype=np.float32)).cuda()

        shapedirs=torch.from_numpy(np.array(m.shapedirs,dtype=np.float32)).cuda()
        posedirs=torch.from_numpy(np.array(m.posedirs,dtype=np.float32)).cuda()
        posedirs=posedirs.permute(2,0,1)
        posedirs=posedirs.contiguous().view(135,-1)
        J_regressor=torch.from_numpy(np.array(m.J_regressor.toarray(),dtype=np.float32)).cuda()
        lbs_weights=torch.from_numpy(np.array(m.weights,dtype=np.float32)).cuda()

        parents=torch.LongTensor([-1,          0,          1,          2,          0,
                        4,          5,          0,          7,          8,
                        0,         10,         11,          0,         13,
                        14]).cuda()

        faces=torch.from_numpy(np.array(m.f,dtype=np.float32))
        scene=trimesh.scene.Scene()
        v,j=self.lbs(betas,pose,v_template,shapedirs,posedirs,J_regressor,parents,lbs_weights)
        out_mesh = trimesh.Trimesh(v.detach().cpu().numpy().reshape((778,3)),faces=faces,prcess=True)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1.0, 0.0,  0.0])
        out_mesh.apply_transform(rot)

        scene.add_geometry(out_mesh)
        file_name="../evaluation_output/output.jpg"
        img_scene = scene.save_image(resolution=(224,224))
        rendered = Image.open(BytesIO(img_scene)).convert('RGB')
        scene.delete_geometry(["geometry_0"])
        return rendered, out_mesh

"""
ren=Render()
pose=torch.FloatTensor([[-0.9076, -2.1418,  1.3428, -0.0209, -0.0530, -0.0746, -0.0030, -0.0256,
                0.2309,  0.0129, -0.0453,  0.0233,  0.0684,  0.0022,  0.1944, -0.1247,
                -0.0996, -0.1648,  0.0471,  0.0249, -0.1274, -0.0260, -0.0078,  0.2745,
                -0.0534, -0.0938, -0.0986,  0.0842,  0.1186, -0.1396,  0.1119,  0.0065,
                0.1549, -0.1469, -0.0863, -0.0340, -0.0132,  0.1406, -0.0166, -0.0618,
                0.0674, -0.2518, -0.2383, -0.0024,  0.1387,  0.0155, -0.0763,  0.0194]]).cuda()
_=ren.renderer(pose)
"""