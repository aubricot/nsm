# Make a video walking down the spine of 4 species

import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.manifold import Isomap
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import re
from scipy.signal import savgol_filter
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents

# Define PC index and model checkpoint to use for video generation
TRAIN_DIR = "run_v41" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '1000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Pick 4 species to compare
species_ids =   ["varanidae_varanus-prasinus-uf71411",
                "varanidae_varanus_komodoensis_tnhc113000",
                "teiidae_tupinambis_tex_lsumz47686",
                "tropiduridae_tropidurus_torquatus_uf60944"]  # TO DO: Update this to the desired species ID
incl_spec = True # TO DO: Include species label or only to genus level?

# Number of total vertebrae to generate along the spine
n_vertebrae = 50 # TO DO: Adjust this number

# Define functions

# Sort the files by vertebrae label (C1, C2, ..., L40)
def sort_by_numerical_prefix_and_vertebra(file_label):
    # Extract the numerical prefix (e.g., "01", "03") and the vertebra label (e.g., C3, L1)
    match = re.search(r'(\d{2})-([CTL]\d+)', file_label)
    if match:
        prefix = int(match.group(1))  # Convert prefix to integer for proper sorting
        vertebra_label = match.group(2)  # e.g., "C3", "L10"
        return (prefix, vertebra_label)
    return (0, "")  # Default case if matching fails

# More accurate control over number of interpolated latent codes
def interpolate_latent_codes_exact(latent_codes, total_output):
    segments = len(latent_codes) - 1
    points_per_segment = total_output // segments
    extra = total_output % segments
    interpolated = []
    for i in range(segments):
        n_points = points_per_segment + (1 if i < extra else 0)
        start = latent_codes[i]
        end = latent_codes[i + 1]
        for t in np.linspace(0, 1, n_points, endpoint=False):
            interpolated.append((1 - t) * start + t * end)
    interpolated.append(latent_codes[-1])  # Ensure we include the last one
    return np.array(interpolated)

def smooth_latent_codes(latent_codes, window_size=5):
    smoothed_codes = []
    half_window = window_size // 2
    n = len(latent_codes)
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = latent_codes[start:end]
        avg = np.mean(window, axis=0)
        smoothed_codes.append(avg)
    return np.array(smoothed_codes)

def plot_spinal_path(proj_val, n_vertebrae, width=200, height=80):
    # Plot latent points
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=200)
    # Set the background color to black
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')  # Black background for the axes
    # Plot latent points
    ax.set_xlim(0, n_vertebrae)
    ax.plot([0, n_vertebrae], [0, 0], color='paleturquoise', alpha=0.2, linewidth=1)
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
    ax.text(0.5, 1, "Spinal Position", fontsize=8, color='white', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.5, -0.05, f"({proj_val} / {n_vertebrae} vertebrae)", fontsize=8, color='white', ha='center', va='top', transform=ax.transAxes)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)[..., :3]  # Drop alpha if any
    return img_np

# Get marker based on life history of species
def get_marker(species):
    species = species.lower()
    if "ouroborus" in species:  # bites tail to front
        return 'v' # triangle (down)
    elif any(k in species for k in ["chalcides", "tetradactylus", "chamaesaura"]):  # swims in grass
        return 'P' # plus filled
    elif any(k in species for k in ["skoog", "eremiascincus", "_scincus"]): # swims in sand
        return '+'  # plus regula
    elif any(k in species for k in ["acontias", "mochlus", "rhineura", "dibamus", "lanthonotus", "bipes", "diplometopon", "pseudopus"]):  # burrowing
        return 's'  # square
    elif any(k in species for k in ["jonesi", "corucia", "gecko", "chamaeleo", "iguana", "brookesia", "dracaena", "anolis", "basiliscus", 
                                    "dracaena", "aristelliger", "sceloporus", "lialis", "phyllurus"]):  # arboreal
        return 'd' # thin diamond
    elif any(k in species for k in ["elgaria", "smaug_giganteus", "broadleysaurus", "ateuchosaurus", "alopoglossus", "heloderma", "tupinambis",
                                    "carlia", "lipinia", "tiliqua", "tribolonotus", "leiolepis", "eublepharis", "oreosaurus", "baranus",
                                    "callopistes", "cricosaura", "lepidophyma", "sphenodon", "lacerta", "enyaloides", "crocodilurus"]):  # terrestrial
        return 'X' # x filled
    elif any(k in species for k in ["eryx", "homalopsis", "aniolios"]):   # snake
        return '2' #
    else:                      # saxicolous/rock dwelling
        return 'o'  # circle (default)

# Define markers and their corresponding life history strategy labels
legend_items = [
    ('v', 'Bites tail to side'),
    ('P', 'Grass swimmer'),
    ('+', 'Sand swimmer'),
    ('s', 'Burrowers'),
    ('d', 'Arboreal'),
    ('X', 'Terrestrial'),
    ('o', 'Saxicolous'),
    ('2', 'Snake')]

# Get the corresponding label for the marker
def get_life_history_label(marker):
    for m, label in legend_items:
        if m == marker:
            return label
    return None 

# Load config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

# Define vertebral regions
vertebral_regions = ['C', 'T', 'L']
latent_codes_subs = []
all_vtk_files_subs = []
# Match "_C1" to "_C40" or "-C1" to "-C40"
for vert_region in vertebral_regions:
    r_p = r'[_-]' + vert_region + r'([1-9]|[1-3][0-9]|40)(?!\d)'
    pattern = re.compile(r_p, re.IGNORECASE)
    # Subset indices from all paths
    matches = [
                (i, int(pattern.search(fname).group(1)))
                for i, fname in enumerate(all_vtk_files)
                if pattern.search(fname)
                ]
    indices = [i for i, _ in matches]

    # Filter latent codes and corresponding mesh paths
    vert_region_codes = latent_codes[indices]
    vert_region_files = [all_vtk_files[i] for i in indices]
    # Add to dictionary
    latent_codes_subs.extend(vert_region_codes)
    all_vtk_files_subs.extend(vert_region_files)

latent_codes_tensor = torch.stack([torch.tensor(latent) for latent in latent_codes_subs])
latent_codes_subs = latent_codes_tensor
print(f"Found: {len(all_vtk_files_subs)} latent codes for vertebral regions: {vertebral_regions}.")

# Filter latent codes and file paths that match the species
species_latents_dict = {}
for species_id in species_ids:
    species_latent_codes = []
    species_vtk_files = []
    for fname, latent in zip(all_vtk_files_subs, latent_codes_subs):
        # Assume that the species ID is part of the file name
        if species_id.lower() in fname.lower():
            species_latent_codes.append(latent.numpy())  # Assuming `latent` is a tensor
            species_vtk_files.append(fname)
    sorted_files = sorted(zip(species_vtk_files, species_latent_codes), key=lambda x: sort_by_numerical_prefix_and_vertebra(x[0]))
    sorted_latents = [x[1] for x in sorted_files]
    # Interpolate between each adjacent latent code
    sorted_latents = [latent.numpy() if isinstance(latent, torch.Tensor) else latent for latent in sorted_latents]
    smoothed_latents = smooth_latent_codes(sorted_latents, window_size=5)
    interpolated_latents = interpolate_latent_codes_exact(smoothed_latents, n_vertebrae)
    species_latents_dict[species_id] = interpolated_latents
    print(f"Found {len(interpolated_latents)} latent codes for species '{species_id}'.")

# Map interpolated latent codes back to filenames (assuming interpolation is uniform)
# You can optionally create a function to map filenames for each interpolated latent code
interpolated_filenames = []
for i in range(len(interpolated_latents)):
    # Calculate the closest original latent code by distance (or use an averaging approach)
    filename_idx = min(i, len(sorted_files) - 1)  # Wrap around if needed
    interpolated_filenames.append(sorted_files[filename_idx][0])

# Now, you have interpolated_latentsand their corresponding filenames
print(f"Generated {len(interpolated_latents)} vertebrae latent codes.")

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 128
voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset = np.array([0.0, 0.0, 0.0])
scale = 1.0
icp_transform = NumpyTransform(np.eye(4))
objects = 1

# Path params
n_frames = 120 * 9 # TO DO: Adjust the number of frames to resample by

# Setup Offscreen Renderers (4)
width, height = 640, 480
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1.0, 1.0, 1.0, 1.0]

# Video Writer (2x2 grid → 1280x960)
video_path = "spinal_walkthrough" # TO DO: Adjust filename
video_path = video_path + ".mp4"
fps = 8
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

generated_mesh_count = 0
for i in range(n_vertebrae):
    try:
        generated_mesh_count += 1
        spinal_path_img = plot_spinal_path(generated_mesh_count, n_vertebrae)
        spinal_path_img_bgr = cv2.cvtColor(spinal_path_img, cv2.COLOR_RGB2BGR)
        images = []
        for species_id in species_ids:
            print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{n_vertebrae} for species_id: {species_id}\033[0m")
            latent_code = species_latents_dict[species_id][i]
            new_latent = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(device)

            # Generate mesh
            mesh_out = create_mesh(
                decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
                voxel_origin=voxel_origin, voxel_size=voxel_size,
                path_original_mesh=None, offset=offset, scale=scale,
                icp_transform=icp_transform, objects=objects,
                verbose=False, device=device
            )
            mesh_out = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
            mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_geometry()
            mesh_pv = mesh_pv.compute_normals(cell_normals=False, point_normals=True, inplace=False)
            mesh_o3d = pv_to_o3d(mesh_pv)

            # Setup renderer
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            renderer.scene.set_background([0, 0, 0, 1])
            renderer.scene.add_geometry("mesh", mesh_o3d, material)

            # Fixed SIDE VIEW camera
            pts = np.asarray(mesh_o3d.vertices)
            center = pts.mean(axis=0)
            r = np.linalg.norm(pts - center, axis=1).max()
            distance = 2.5 * r
            cam_pos = center + np.array([0, -distance, 0])
            up_dir = [0, 0, 1]
            renderer.setup_camera(60, center, cam_pos, up_dir)

            img = np.asarray(renderer.render_to_image())
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Label species
            parts = species_id.split("_")
            family = parts[0].capitalize() if len(parts) > 0 else species_id
            genus = parts[1].capitalize() if len(parts) > 1 else ""
            marker = get_marker(species_id)
            life_hist = get_life_history_label(marker)
            if incl_spec:
                species = parts[-1].capitalize() if len(parts) > 1 else ""
                label = f"{family}\n{genus}\n{species}\n{life_hist}"
            else:
                label = f"{family}\n{genus}\n{life_hist}"
            print(label)
            # Put the text
            text_y = 100
            for j, line in enumerate(label.split("\n")):
                y = text_y + j*20
                cv2.putText(img_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            images.append(img_bgr)

            renderer.scene.clear_geometry()
            del renderer, mesh_out, mesh_pv, mesh_o3d, new_latent
            gc.collect()

        # Compose images into 2×2 grid
        top = np.hstack([images[0], images[1]])
        bottom = np.hstack([images[2], images[3]])
        combined = np.vstack([top, bottom])

        # Overlay latent_path_img in the middle of the 2x2 grid
        # Calculate overlay position (centered horizontally & vertically)
        center_x = combined.shape[1] // 2
        center_y = combined.shape[0] // 2
        h, w, _ = spinal_path_img_bgr.shape
        x_start = center_x - w // 2
        y_start = center_y - h // 2
        # Add transparency by blending (optional, here full opaque)
        combined[y_start:y_start+h, x_start:x_start+w] = spinal_path_img_bgr

        # Write to video
        out_video.write(combined)

    except Exception as e:
        print(f"Error at frame {i}: {e}")
    finally:
        for var in ['mesh_out', 'mesh_pv', 'mesh_o3d', 'new_latent']:
            if var in locals():
                del locals()[var]
        gc.collect()

out_video.release()
print("Video saved as", video_path)
