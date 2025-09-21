import os

os.environ["LOC_SDF_CACHE"] = "/dataNAS/people/aagatti/projects/deepsdf_femur/cache"

from NSM.datasets import MultiSurfaceSDFSamples

import json

with open(
    "/dataNAS/people/aagatti/projects/OAI_Segmentation/data_splits_July.5.2023.json", "r"
) as f:
    dict_meshes = json.load(f)

list_mesh_paths = list(zip(dict_meshes["train"]["bone"], dict_meshes["train"]["cartilage"]))

sigma_near = 0.00025 ** (1 / 2) * 40
sigma_far = 0.0025 ** (1 / 2) * 40

sdf_dataset = MultiSurfaceSDFSamples(
    list_mesh_paths=list_mesh_paths[:3],
    subsample=17000,
    print_filename=True,
    n_pts=[500000, 500000],
    p_near_surface=0.45,
    p_further_from_surface=0.45,
    sigma_near=sigma_near,
    sigma_far=sigma_far,
    rand_function="normal",
    center_pts=False,
    scale_all_meshes=True,
    center_all_meshes=False,
    mesh_to_scale=0,
    norm_pts=False,
    scale_method="max_rad",
    random_seed=52122,
    reference_mesh=0,
    verbose=True,
    save_cache=False,
    equal_pos_neg=True,
    fix_mesh=False,
    load_cache=False,
)
