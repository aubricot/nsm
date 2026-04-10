import numpy as np
from scipy.spatial import cKDTree

def _uniform_surface_sample(poly, n):
    # Triangulate the mesh
    poly = poly.triangulate().extract_geometry()
    verts = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:] # (T,3)
    # Calculate areas of each triangle
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    # Select triangles to sample from
    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(faces), size=n, p=probs)
    a = v0[tri_idx]; b = v1[tri_idx]; c = v2[tri_idx]
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    # Barycentric sampling to find random points inside each triangle
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return pts

def chamfer_distance(sg, sp):
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0].mean()
    d2 = t2.query(sp, k=1)[0].mean()
    return np.float64(0.5*(d1 + d2)) # Return average distance (symmetric penalty)

def f_score(sg, sp, d=0.005):
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    tree_pred = cKDTree(sp); tree_gt = cKDTree(sg)
    nn_pred = tree_pred.query(sg, k=1)[0]
    recall = np.mean(nn_pred < d)
    nn_gt = tree_gt.query(sp, k=1)[0]
    precision = np.mean(nn_gt < d)
    if precision + recall == 0:
        return 0.0, precision, recall
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall

def ave_sym_surface_distance(sg, sp):
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0]
    d2 = t2.query(sp, k=1)[0]
    return (np.sum(d1) + np.sum(d2)) / (d1.shape[0] + d2.shape[0])