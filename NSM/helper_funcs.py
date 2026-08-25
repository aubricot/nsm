# Utility functions for loading trained models and inspecting results
import os, json, torch, numpy as np, pyvista as pv, vtk
from NSM.models import TriplanarDecoder
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import re
import pymskt.mesh.meshes as meshes
import torch.nn.functional as F
import cv2
from NSM.mesh import create_mesh
import colorsys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
try:
    import open3d as o3d
except:
    print("Error importing open3d")

# ICP transform
class NumpyTransform:
    def __init__(self, matrix):
        self.matrix = matrix
    def GetMatrix(self):
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, self.matrix[i, j])
        return vtk_mat

# Pyvista to Open3D    
def pv_to_o3d(mesh_pv):
    pts = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces)
    tris = faces.reshape(-1,4)[:,1:4]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tris)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

# Convert ply file to vtk
def convert_ply_to_vtk(input_file, output_file=None, output_dir=None, save=False):
    if not os.path.exists(input_file):
        print(f"File does not exist: {input_file}")
        return None, None
    mesh = pv.read(input_file)
    if input_file.lower().endswith(".ply"):
        return mesh, input_file
    if output_file is None:
        name = os.path.splitext(os.path.basename(input_file))[0] + ".vtk"
        output_file = os.path.join(output_dir or os.path.dirname(input_file), name)
    if save:
        mesh.save(output_file)
        print(f"Converted {input_file} -> {output_file}")
    return mesh, output_file

# Load model config file
def load_config(config_path='model_params_config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\033[92mLoaded config from {config_path}\033[0m")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: model_params_config.json not found at {config_path}")

# Load model and latents
def load_model_and_latents(MODEL_PATH, LC_PATH, config, device):
    # Load model
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
    # Load latents
    latent_ckpt = torch.load(LC_PATH, map_location=device)
    latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()
    return model, latent_ckpt, latent_codes

# begin monkey patch
def safe_load_mesh_scalars(self):
    try:
        if hasattr(self, 'mesh'):
            mesh = self.mesh
        elif hasattr(self, '_mesh'):
            mesh = self._mesh
        else:
            raise AttributeError("No mesh attribute found in Mesh object.")
        point_scalars = mesh.point_data
        cell_scalars = mesh.cell_data
        if point_scalars and len(point_scalars.keys()) > 0:
            self.mesh_scalar_names = list(point_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        elif cell_scalars and len(cell_scalars.keys()) > 0:
            self.mesh_scalar_names = list(cell_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        else:
            self.mesh_scalar_names = []
            self.scalar_name = None
            print("No scalar data found in mesh. Proceeding without scalars.")
    except Exception as e:
        print(f"Failed to load mesh scalars: {e}")
        self.mesh_scalar_names = []
        self.scalar_name = None

def fixed_point_coords(self):
    if self.n_points < 1:
        raise AttributeError(f"No points found in mesh '{self}'")
    return self.points

def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device='cuda'):
    n_pts_total = samples.shape[0]
    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects, device=device) 
    if batch_size > n_pts_total:
        #print('WARNING: batch_size is greater than the number of samples, setting batch_size to the number of samples')
        batch_size = n_pts_total
    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)
        sdf_values[current_idx : current_idx + current_batch_size, :] = decode_sdf(
            decoder, latent_vector, sampled_pts, device) 
        current_idx += current_batch_size
        #print(f"Processed {current_idx} / {n_pts_total} points")
    return sdf_values

def decode_sdf(decoder, latent_vector, queries, device='cuda'):
    num_samples = queries.shape[0]
    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1).to(device)
        inputs = torch.cat([latent_repeat, queries], dim=1)
    inputs = inputs.to(next(decoder.parameters()).device)  
    return decoder(inputs)
# end monkey patch

def parse_labels_from_filepaths(filepaths, regex_pattern=r"^(?P<species>[\w\s\-]+)[\-_ ]+[\w\d]+[\-_ ]+(?P<vertebra>[CTL]?\d+)", show_debug=False):
    r_p = re.compile(regex_pattern, re.IGNORECASE)
    labels = []
    unmatched_files = []
    matched_examples = []
    for f in filepaths:
        fname = os.path.basename(f)
        match = r_p.match(fname)
        if match:
            species = match.group("species").strip()
            vertebra = match.group("vertebra").strip()
            labels.append((species, vertebra))
            if len(matched_examples) < 1:
                matched_examples.append(fname)
        else:
            labels.append((None, None))
            unmatched_files.append(fname)
            print(f"\033[31mFile unmatched by regex: {fname}\033[0m")
    print(f"Extracted labels for {len(labels)} out of {len(filepaths)} files.")
    
    # --- Optional debug output ---
    if show_debug:
        print("\nRegex pattern used:")
        print(f"  {regex_pattern}")
        if matched_examples:
            print("\nExample MATCH:")
            print(f"  ✓ {matched_examples[0]}")
        else:
            print("\nExample MATCH:")
            print("  (none found)")
        if unmatched_files:
            print("\nExample NON-MATCH:")
            print(f"  ✗ {unmatched_files[0]}")
        else:
            print("\nExample NON-MATCH:")
            print("  (none found)")
        print(f"\nTotal unmatched files: {len(unmatched_files)}")
    return labels, unmatched_files

# Get species name using regex from filenames (ex: Scincidae_Tribolonotus_novaeguineae)
def extract_species_prefix(filename):
    match = re.match(r"([A-Za-z]+_[A-Za-z]+_[a-z]+)", filename.lower())
    if match:
        return match.group(1)
    else:
        return None
    
def average_across_regions(regex_pattern, vert_region, vert_region_files, vert_region_codes):
    specimen_pattern = re.compile(regex_pattern, re.IGNORECASE)
    specimen_latents = {}
    specimen_files = {}
    # Group latents by region for averaging using regex
    for fname, latent in zip(vert_region_files, vert_region_codes):
        match = specimen_pattern.match(fname)
        if match:
            specimen_id = match.group(1)
            if specimen_id not in specimen_latents:
                specimen_latents[specimen_id] = []
                specimen_files[specimen_id] = []
            specimen_latents[specimen_id].append(latent.numpy())
            specimen_files[specimen_id].append(fname)
        else:
            print(f"\033[93mWarning: could not extract specimen ID from {fname}\033[0m")
    # Average the latent codes per specimen
    avg_latent_codes = []
    avg_specimen_ids = []
    for specimen_id, latents in specimen_latents.items():
        avg_latent = np.mean(latents, axis=0)
        avg_latent_codes.append(avg_latent)
        avg_specimen_ids.append(specimen_id + '_' + vert_region)
    # Convert to NumPy array
    avg_latent_codes = np.array(avg_latent_codes)
    print(f"\nAveraged latent codes for {len(avg_specimen_ids)} specimens.\nSample specimen IDs: {avg_specimen_ids[:5]}")
    vert_region_codes = avg_latent_codes
    vert_region_files = avg_specimen_ids
    return vert_region_files, vert_region_codes

# Overlay specimen name info onto each frame
def overlay_text_on_frame(frame, i, loop_sequence_names):
    specimen_name = loop_sequence_names[i]
    parts = specimen_name.split("_")
    family = parts[0] if len(parts) > 0 else specimen_name
    genus = parts[1]  if len(parts) > 1 else ""
    region = parts[-1] if len(parts) > 2 else ""
    if 'C' in region:
        reg_full = 'Cervical'
    elif 'T' in region:
        reg_full = 'Thoracic'
    elif 'L' in region:
        reg_full = 'Lumbar'
    else:
        reg_full = ''
    text = f"Closest Specimen: \n{family}\n{genus}\n{reg_full}"
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Position: center of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    text_x = center_x - 120
    text_y = center_y
    # Put the text
    for j, line in enumerate(text.split("\n")):
            y = text_y + j * (text_size[1] + 10)
            cv2.putText(frame, line, (text_x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    return frame

def render_cameras(renderers, mesh_o3d, i, material, loop_sequence, n_rotations):
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
    angle_deg = (i /  loop_sequence) * 360 * n_rotations
    angle_rad = np.deg2rad(angle_deg)
    cam_positions = [center + np.array([  # Top Left: rotating
                    distance * np.cos(angle_rad) * np.cos(elevation),
                    distance * np.sin(angle_rad) * np.cos(elevation),
                    distance * np.sin(elevation)]),
                    center + np.array([0, -distance, 0]),  # Top Right: side
                    center + np.array([distance, 0, 0]),  # Bottom Left: back (90° CCW from side)
                    center + np.array([0, 0, distance])]    # Bottom Right: top-down (90° CCW from side)
    ups = [[0, 0, 1],  # rotating
            [0, 0, 1],  # front
            [0, 0, 1],  # side
            [0, 1, 0],]  # top-down
    # Define other camera positions (side, top-down, etc.)
    for idx, (rdr, pos, up) in enumerate(zip(renderers, cam_positions, ups)):
        rdr.setup_camera(60, center, pos, up)
    # Render images
    imgs = [np.asarray(r.render_to_image()) for r in renderers]
    imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    # Compose 4 views into 2x2 grid (width=640, height=480)
    top = np.hstack([imgs_bgr[0], imgs_bgr[1]])
    bottom = np.hstack([imgs_bgr[2], imgs_bgr[3]])
    combined = np.vstack([top, bottom])
    return combined

def generate_and_render_mesh(latent_code, loop_sequence, i, device, model, n_pts_per_axis,
                             voxel_origin, voxel_size, offset, scale, icp_transform, objects, generated_mesh_count,
                             loop_sequence_names=None):
    generated_mesh_count += 1
    print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{loop_sequence}\033[0m")
    if loop_sequence_names is not None:
        print(f"Frame {i}: Closest to {loop_sequence_names[i]}")
    new_latent = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(device)
    mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device)
    mesh_out = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
    mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_geometry()
    mesh_pv = mesh_pv.compute_normals(cell_normals=False, point_normals=True, inplace=False)
    mesh_o3d = pv_to_o3d(mesh_pv)
    return mesh_o3d
    
# Define the vertebra_sort_key function for vertebrae sorting based on region and position
def vertebra_sort_key(item):
    # Define the region order: C < T < L
    region_order = {'C': 0, 'T': 1, 'L': 2}
    vertebra_label = item[0]  # Extract vertebra label (e.g., C4, T1, L2)
    region = vertebra_label[0].upper()  # Extract the region (C, T, L)
    try:
        number = int(vertebra_label[1:])  # Extract the numeric part and convert to integer
    except ValueError:
        number = float('inf')  # In case of malformed labels, put them at the end
    # Return a tuple with region order and the numeric value for sorting
    return (region_order.get(region, 200), number)

# Function to classify vertebrae by region
def get_region(vertebra_label):
    if vertebra_label.startswith('c') or vertebra_label.startswith('C'):
        return 'Cervical'
    elif vertebra_label.startswith('t') or vertebra_label.startswith('T'):
        return 'Thoracic'
    elif vertebra_label.startswith('l') or vertebra_label.startswith('L'):
        return 'Lumbar'
    return 'Unknown'

# Find shape completion files
def find_shape_completion_files(root_dir):
    return sorted(
        str(p)
        for p in Path(root_dir).rglob("*")
        if (p.is_file()
            and "zzz_" in p.name.lower()
            and p.name.lower().endswith("shape_completion.vtk")))