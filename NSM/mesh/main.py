"""
https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/mesh.py
"""

from skimage.measure import marching_cubes
import pyvista as pv
import os
import torch
import numpy as np
import pymskt as mskt
import vtk

def scale_mesh_(mesh, scale=1.0, offset=(0., 0., 0.), icp_transform=None, verbose=False):
    if not issubclass(type(mesh), mskt.mesh.Mesh):
        mesh = mskt.mesh.Mesh(mesh)
    
    if verbose is True:
        print('scale_mesh_. scale:', scale)

    pts = mesh.point_coords * scale
    pts += offset

    mesh.point_coords = pts

    if icp_transform is not None:
        transform = vtk.vtkTransform()
        transform.SetMatrix(icp_transform.GetMatrix())
        transform.Inverse()
        if verbose is True:
            print(icp_transform)
            print('INVERSE')
            print(transform)
        mesh.apply_transform_to_mesh(transform)
    
    return mesh

def scale_mesh(
    new_mesh, 
    old_mesh=None, 
    scale=1.0,
    offset=(0., 0., 0.),
    scale_method='max_rad',
    icp_transform=None,
    verbose=False
    ):

    if old_mesh is not None:
        old_mesh = mskt.mesh.Mesh(old_mesh) # should handle vtk, pyvista, or string path to file
        old_pts = old_mesh.point_coords

        if not issubclass(type(new_mesh), mskt.mesh.Mesh):
            new_mesh = mskt.mesh.Mesh(new_mesh) # should handle vtk, pyvista, or string path to file
        new_pts = new_mesh.point_coords

        offset = np.mean(old_pts, axis=0)
        old_pts -= offset

        if scale_method == 'max_rad':
            scale = np.max(np.linalg.norm(old_pts, axis=-1), axis=-1)
        else:
            raise NotImplementedError
    
    mesh = scale_mesh_(new_mesh, scale=scale, offset=offset, icp_transform=icp_transform, verbose=verbose)
    return mesh

def create_mesh(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    batch_size=32**3,
    scale=1.0,
    offset=(0., 0., 0.),
    path_save=None,
    filename='mesh_{mesh_idx}.vtk',
    path_original_mesh=None,
    scale_to_original_mesh=True,
    icp_transform=None,
    objects=1,
    verbose=False,
    device='cuda'
):

    if voxel_size is None:
        voxel_size = 2.0 / (n_pts_per_axis - 1)

    decoder.eval()

    samples = create_grid_samples(n_pts_per_axis, voxel_origin, voxel_size)
    sdf_values_ = get_sdfs(decoder, samples, latent_vector, batch_size, objects=objects, device=device)

    # resample SDFs into a grid:
    sdf_values = torch.zeros((n_pts_per_axis, n_pts_per_axis, n_pts_per_axis, objects))
    for i in range(objects):
        sdf_values[..., i] = sdf_values_[..., i].reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)
    # sdf_values = sdf_values.reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)
    
    # create mesh from gridded SDFs
    meshes = []
    for mesh_idx in range(objects):
        # iterate over all the meshes
        sdf_values_ = sdf_values[..., mesh_idx]

        # check if there is a surface
        if 0 < sdf_values_.min() or 0 > sdf_values_.max():
            if verbose is True:
                print('WARNING: SDF values do not span 0 - there is no surface')
                print('\tSDF min: ', sdf_values_.min())
                print('\tSDF max: ', sdf_values_.max())
                print('\tSDF mean: ', sdf_values_.mean())
            meshes.append(None)
        else:
            # if there is a surface, then extract it & post-process
            # for mesh_idx in range(objects):
            mesh = sdf_grid_to_mesh(sdf_values_, voxel_origin, voxel_size)
            meshes.append(mesh)

            if scale_to_original_mesh:
                if verbose is True:
                    print('Scaling mesh to original mesh... ')
                    print(icp_transform)
                # for mesh_idx, mesh in enumerate(meshes):
                mesh = scale_mesh(meshes[mesh_idx], old_mesh=path_original_mesh, scale=scale, offset=offset, icp_transform=icp_transform, verbose=verbose)
                meshes[mesh_idx] = mesh

            # save the mesh (if desired)
            if path_save is not None:
            # for mesh_idx, mesh in enumerate(meshes):
                meshes[mesh_idx].save_mesh(os.path.join(path_save, filename.format(mesh_idx=mesh_idx)))
    return meshes[0] if objects == 1 else meshes


def sdf_grid_to_mesh(
    sdf_values,
    voxel_origin,
    voxel_size,
    scale=None,
    offset=None,
    verbose=False,
):
    sdf_values = sdf_values.cpu().numpy()

    if verbose is True:
        print('Starting marching cubes... ')

    verts, faces, normals, values = marching_cubes(
        sdf_values, 
        level=0, 
        spacing=(voxel_size, voxel_size, voxel_size)
    )

    if verbose is True:
        print('Starting vert/face conversion...')

    verts += voxel_origin

    # additional scale / offset: 
    # this could be stored info from normalizing? 
    # e.g., the center/normalizing to be in r = 1 unit sphere
    if scale is not None:
        verts /= scale
    if offset is not None:
        verts -= offset
    
    faces_ = []
    for face_idx in range(faces.shape[0]):
        face = np.insert(faces[face_idx, :], 0, faces.shape[1])
        faces_.append(face)
    
    faces = np.hstack(faces_)

    if verbose is True:
        print('Creating mesh... ')
        
    mesh = mskt.mesh.Mesh(mesh=pv.PolyData(verts, faces))

    return mesh

def create_grid_samples(
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
):    
    n_pts_total = n_pts_per_axis ** 3

    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)
    
    # generate samples on a grid... 
    samples[:, 2] = indices % n_pts_per_axis
    samples[:, 1] = (indices // n_pts_per_axis) % n_pts_per_axis
    samples[:, 0] = ((indices // n_pts_per_axis) // n_pts_per_axis) % n_pts_per_axis

    # scale & transform the grid as appropriate
    samples[:, :3] = samples[:, :3] * voxel_size
    for axis in range(3): 
        samples[:, axis] = samples[:, axis] + voxel_origin[axis]

    return samples

def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device='cuda'):

    n_pts_total = samples.shape[0]

    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects)
    
    if batch_size > n_pts_total:
        print('WARNING: batch_size is greater than the number of samples, setting batch_size to the number of samples')
        batch_size = n_pts_total
    

    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)
        sdf_values[current_idx : current_idx + current_batch_size, :] = decode_sdf(
            decoder, latent_vector, sampled_pts
        ).detach().cpu()

        current_idx += current_batch_size
        print(f"Processed {current_idx} / {n_pts_total} points")
    # sdf_values.squeeze(1)
    return sdf_values

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], dim=1)
    
    return decoder(inputs)


