import pymskt as mskt
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import time
import numpy as np
import point_cloud_utils as pcu


def vtk_sdf(mesh, pts):
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh)

    # sdfs = np.zeros(pts.shape[0])

    # for pt_idx in range(pts.shape[0]):
    #     sdfs[pt_idx] = implicit_distance.EvaluateFunction(pts[pt_idx,:])

    vtk_pts = numpy_to_vtk(pts)
    sdfs = numpy_to_vtk(np.zeros(pts.shape[0]))
    implicit_distance.FunctionValue(vtk_pts, sdfs)

    sdfs = vtk_to_numpy(sdfs)
    return sdfs


def pcu_sdf(mesh, pts):
    faces = vtk_to_numpy(mesh.GetPolys().GetData())
    faces = faces.reshape(-1, 4)
    faces = np.delete(faces, 0, 1)
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(pts, points, faces)

    return sdfs


MEAN = (0, 0, 0)
# SIGMA = 1/45
# NORM = True
SIGMA = 1
NORM = False
N_REPEATS = 4

rand_gen = np.random.default_rng().multivariate_normal
COV = np.identity(len(MEAN)) * SIGMA**2  # **2

mesh = mskt.mesh.Mesh(
    "/dataNAS/people/aagatti/projects/OAI_Segmentation/oai_predictions_april_2023/2d_models/00_month/meshes_July_5_2023/9569243/RIGHT_femur__fixed_July_5_2023.vtk"
)

pts = mesh.point_coords

if NORM:
    pts -= np.mean(pts, axis=0)
    pts = pts / np.max(np.linalg.norm(pts, axis=1))
    mesh.point_coords = pts

pts = np.concatenate(
    [
        pts,
    ]
    * N_REPEATS,
    axis=0,
)

print(np.min(pts, axis=0))
print(np.max(pts, axis=0))

noise = rand_gen(MEAN, COV, pts.shape[0])
# noise = (np.random.rand(pts.shape[0], pts.shape[1]) - 0.5) * 1 / 1000
pts += noise

# pts = pts.astype(np.float64)
print(pts.shape)
print(pts.dtype)

tic = time.time()
vtk_sdfs = vtk_sdf(mesh.mesh, pts)
toc = time.time()
print(toc - tic)

tic = time.time()
pcu_sdfs = pcu_sdf(mesh.mesh, pts)
toc = time.time()
print(toc - tic)

print(np.mean(vtk_sdfs), np.std(vtk_sdfs), np.min(vtk_sdfs), np.max(vtk_sdfs))
print(np.mean(pcu_sdfs), np.std(pcu_sdfs), np.min(pcu_sdfs), np.max(pcu_sdfs))
diff = vtk_sdfs - pcu_sdfs
print(np.mean(diff), np.std(diff), np.min(diff), np.max(diff))

np.save("vtk_sdfs.npy", vtk_sdfs)
np.save("pcu_sdfs.npy", pcu_sdfs)
