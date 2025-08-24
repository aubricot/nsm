import pymskt as mskt
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import time
import numpy as np

MEAN = (0, 0, 0)
# SIGMA = 0.00025 ** (1/2) * 45
# SIGMA = 0.00025 ** (1/2) * 45
# NORM = False
# SIGMA = 0.0025 ** 0.5
# SIGMA = 0.00025
# SIGMA = 1.0 #0.0025 ** 0.5 * 45
SIGMA = 1 / 45
NORM = True
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
mesh = mesh.mesh
implicit_distance = vtk.vtkImplicitPolyDataDistance()
implicit_distance.SetInput(mesh)

# sdfs = np.zeros(pts.shape[0])

# for pt_idx in range(pts.shape[0]):
#     sdfs[pt_idx] = implicit_distance.EvaluateFunction(pts[pt_idx,:])

vtk_pts = numpy_to_vtk(pts)
sdfs = numpy_to_vtk(np.zeros(pts.shape[0]))
implicit_distance.FunctionValue(vtk_pts, sdfs)

sdfs = vtk_to_numpy(sdfs)

toc = time.time()
print(toc - tic)

print(np.mean(sdfs), np.std(sdfs), np.min(sdfs), np.max(sdfs))
print(np.mean(noise), np.std(noise), np.min(noise), np.max(noise))
