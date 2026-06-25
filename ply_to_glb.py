import os
import open3d as o3d

input_dir = "PLYs_kw_renamed_v4"
output_dir = "glbs"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):

    if not filename.lower().endswith(".ply"):
        continue

    input_file = os.path.join(input_dir, filename)

    # change extension
    output_file = os.path.join(
        output_dir,
        os.path.splitext(filename)[0] + ".glb"
    )

    print(f"Converting:\n  {input_file}\n  -> {output_file}")

    mesh = o3d.io.read_triangle_mesh(input_file)

    if mesh.is_empty():
        print("  ERROR: empty mesh, skipping")
        continue

    mesh.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(
        output_file,
        mesh,
        write_triangle_uvs=False
    )

    if success:
        print("  done")
    else:
        print("  FAILED")