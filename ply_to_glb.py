# First run this to convert plys to glbs
# Then run compress_draco.sh to convert glbs to draco glbs

import os
import open3d as o3d
import numpy as np
import pyvista as pv

input_dir = "run_v72/fossils/models_smooth_hollow/aligned/"
output_dir = "glbs_v72_fossils/"

os.makedirs(output_dir, exist_ok=True)
def vtk_to_open3d_mesh(vtk_path):
    mesh = pv.read(vtk_path)

    if mesh.n_points == 0:
        return None

    surface = mesh.extract_surface(algorithm=None).triangulate()

    if surface.n_points == 0 or surface.n_cells == 0:
        return None

    points = np.asarray(surface.points)

    faces = surface.faces.reshape(-1, 4)[:, 1:4]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".ply", ".vtk")):
        continue

    input_file = os.path.join(input_dir, filename)
    ext = os.path.splitext(filename)[1].lower()

    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".glb")

    print(f"\nConverting:\n  {input_file}\n  -> {output_file}")

    try:
        if ext == ".ply":
            mesh = o3d.io.read_triangle_mesh(input_file)
            if mesh.is_empty():
                print("ERROR: empty PLY mesh, skipping")
                continue
        elif ext == ".vtk":
            mesh = vtk_to_open3d_mesh(input_file)
            if mesh is None or mesh.is_empty():
                print("ERROR: empty VTK mesh, skipping")
                continue
        else:
            print("ERROR: unsupported extension, skipping")
            continue

        mesh.compute_vertex_normals()

        success = o3d.io.write_triangle_mesh(output_file, mesh, write_triangle_uvs=False)

        if success:
            print("done")
        else:
            print("FAILED")

    except Exception as e:
        print(f"ERROR: {e}")