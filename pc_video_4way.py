import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image

# Define PC index and model checkpoint to use for video generation
TRAIN_DIR = "run_v30" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
PC_idx = 0   # TO DO: Choose PC index for PC of interest (ex: For PC1, choose 0)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

class NumpyTransform:
    def __init__(self, matrix):
        self.matrix = matrix
    def GetMatrix(self):
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, self.matrix[i, j])
        return vtk_mat

def pv_to_o3d(mesh_pv):
    pts = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces)
    tris = faces.reshape(-1,4)[:,1:4]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tris)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def generate_latent_path_plot(projections, proj_val, min_proj, max_proj, width=200, height=80):
    print(f"projections shape: {projections.shape}")
    print(f"proj_val: {proj_val}")

    # Ensure projections is a 1D array
    projections = projections.flatten()

    # Plot latent points
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=200)

    # Set the background color to black
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')  # Black background for the axes

    # Plot latent points
    #ax.scatter(projections, np.zeros_like(projections), color='paleturquoise', alpha=0.3, s=4)
    ax.set_xlim(min_proj, max_proj)
    ax.plot([min_proj, max_proj], [0, 0], color='paleturquoise', alpha=0.2, linewidth=1)

    # Plot the current latent point (use proj_val directly and ensure it's scalar)
    ax.scatter(proj_val, 0, color='deeppink', s=10)

    # Customize the plot
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_xticks([0])
    ax.set_xticklabels(['0'])
    ax.grid(False)
    ax.legend().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Latent Path (PC{str((PC_idx+1))})", fontsize=10, color='white')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)[..., :3]  # Drop alpha if any
    return img_np

# Load config
config_path = 'model_params_config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\033[92mLoaded config from {config_path}\033[0m")
except FileNotFoundError:
    raise FileNotFoundError(f"Error: model_params_config.json not found at {config_path}")

device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

latent_ckpt = torch.load(LC_PATH, map_location=device)
latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()
triplane_args = {
    'latent_dim': config['latent_size'],
    'n_objects': config['objects_per_decoder'],
    'conv_hidden_dims': config['conv_hidden_dims'],
    'conv_deep_image_size': config['conv_deep_image_size'],
    'conv_norm': config['conv_norm'], 
    'conv_norm_type': config['conv_norm_type'],
    'conv_start_with_mlp': config['conv_start_with_mlp'],
    'sdf_latent_size': config['sdf_latent_size'],
    'sdf_hidden_dims': config['sdf_hidden_dims'],
    'sdf_weight_norm': config['weight_norm'],
    'sdf_final_activation': config['final_activation'],
    'sdf_activation': config['activation'],
    'sdf_dropout_prob': config['dropout_prob'],
    'sum_sdf_features': config['sum_conv_output_features'],
    'conv_pred_sdf': config['conv_pred_sdf'],
}
model = TriplanarDecoder(**triplane_args)
model_ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(model_ckpt['model'])
model.to(device)
model.eval()

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 128
voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset = np.array([0.0, 0.0, 0.0])
scale = 1.0
icp_transform = NumpyTransform(np.eye(4))
objects = 1

# PCA on latent codes
latents_np = latent_codes.numpy()
latent_mean = np.mean(latents_np, axis=0)
centered = latents_np - latent_mean
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
pc1 = Vt[PC_idx]
projections = centered.dot(pc1)
dists = np.linalg.norm(latents_np - latent_mean, axis=1)
center_idx = np.argmin(dists)
center_sample = latents_np[center_idx]
center_proj = np.dot(center_sample - latent_mean, pc1)
max_proj = np.max(projections)
min_proj = np.min(projections)
high_delta = max_proj - center_proj
low_delta = min_proj - center_proj

# Alpha path
n_rotations = 3
amplify = 1.5
n_seg = 30 * n_rotations
alpha_vals = np.concatenate([
    np.linspace(+low_delta, 0, n_seg),
    np.linspace(0, +high_delta, n_seg),
    np.linspace(+high_delta, 0, n_seg),
    np.linspace(0, +low_delta, n_seg)
])

total_frames = len(alpha_vals)

# Setup Offscreen Renderers (4)
width, height = 640, 480
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height) for _ in range(4)]
for r in renderers:
    r.scene.set_background([0.0, 0.0, 0.0, 1.0])

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1.0, 1.0, 1.0, 1.0]

# Video Writer (2x2 grid → 1280x960)
video_path = f"pc{PC_idx+1}_4way_{amplify}xpc.mp4"
fps = 15
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

for i, alpha in enumerate(alpha_vals):
    try:
        new_latent_np = center_sample + amplify * alpha * pc1
        new_latent = torch.tensor(new_latent_np, dtype=torch.float32).unsqueeze(0).to(device)
        # Compute PC1 projection for the latent space plot
        proj_val = np.dot(new_latent_np - latent_mean, pc1)

        mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device
        )
        mesh_out = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
        mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_geometry()
        mesh_pv = mesh_pv.compute_normals(cell_normals=False, point_normals=True, inplace=False)
        mesh_o3d = pv_to_o3d(mesh_pv)

        for r in renderers:
            r.scene.clear_geometry()
            r.scene.add_geometry("mesh", mesh_o3d, material)

        # Camera setup
        pts = np.asarray(mesh_o3d.vertices)
        center = pts.mean(axis=0)
        r = np.linalg.norm(pts - center, axis=1).max()
        distance = 2.5 * r
        elevation = np.deg2rad(30)

        # Define 4 camera positions
        angle_deg = (i / total_frames) * 360 * n_rotations
        angle_rad = np.deg2rad(angle_deg)
        cam_positions = [
            center + np.array([  # Top Left: rotating
                distance * np.cos(angle_rad) * np.cos(elevation),
                distance * np.sin(angle_rad) * np.cos(elevation),
                distance * np.sin(elevation)
            ]),
            center + np.array([0, -distance, 0]),  # Top Right: front
            center + np.array([-distance, 0, 0]),  # Bottom Left: side (90° CCW from front)
            center + np.array([0, 0, distance])    # Bottom Right: top-down (90° CCW from side)
        ]
        ups = [
            [0, 0, 1],  # rotating
            [0, 0, 1],  # front
            [0, 0, 1],  # side
            [0, 1, 0],  # top-down
        ]

        for idx, (rdr, pos, up) in enumerate(zip(renderers, cam_positions, ups)):
            rdr.setup_camera(60, center, pos, up)

        # Render images
        imgs = [np.asarray(r.render_to_image()) for r in renderers]
        imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

        # Generate latent path plot image
        latent_path_img = generate_latent_path_plot(projections, proj_val, min_proj, max_proj, width=200, height=80)
        latent_path_img_bgr = cv2.cvtColor(latent_path_img, cv2.COLOR_RGB2BGR)

        # Compose 4 views into 2x2 grid (width=640, height=480)
        top = np.hstack([imgs_bgr[0], imgs_bgr[1]])
        bottom = np.hstack([imgs_bgr[2], imgs_bgr[3]])
        combined = np.vstack([top, bottom])

        # Overlay latent_path_img in the middle of the 2x2 grid
        # Calculate overlay position (centered horizontally & vertically)
        center_x = combined.shape[1] // 2
        center_y = combined.shape[0] // 2

        h, w, _ = latent_path_img_bgr.shape
        x_start = center_x - w // 2
        y_start = center_y - h // 2

        # Add transparency by blending (optional, here full opaque)
        combined[y_start:y_start+h, x_start:x_start+w] = latent_path_img_bgr
        out_video.write(combined)

        print(f"Captured frame {i + 1}/{total_frames}")

    except Exception as e:
        print(f"Error at frame {i}: {e}")
    finally:
        for var in ['mesh_out', 'mesh_pv', 'mesh_o3d', 'new_latent']:
            if var in locals():
                del locals()[var]
        gc.collect()

out_video.release()
print("Video saved as", video_path)
