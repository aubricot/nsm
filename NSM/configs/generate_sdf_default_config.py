import json

config = {
    "project_name": "deep-sdf-femur",
    "entity_name": "bone-modeling",
    "run_name": None,
    "tags": ["deepsdf", "nif", "generative"],
    # model parameters:
    "layer_dimensions": [
        512,
    ]
    * 8,  # deepSDF: [512,] * 8
    "layers_with_dropout": list(range(8)),  # deepSDF: list(range(8)),
    "dropout_prob": 0.2,  # deepSDF: 0.2,
    "layers_with_norm": list(range(8)),  # deepSDF: list(range(8)),
    "layer_latent_in": [4],  # deepSDF: [4],
    "xyz_in_all": False,  # deepSDF: False,
    # "use_tanh": False,                                    # deepSDF: False,
    "latent_dropout": False,  # deepSDF: False,
    "weight_norm": True,  # deepSDF: True,
    "activation": "relu",  # deepSDF: "relu",  - siren alternative... "sin"
    "final_activation": "tanh",  # deepSDF: "tanh",  - siren alternative... "sin" or "linear" (which is effectively none)
    # initialization
    "seed": 52122,
    # dateloader parameters:
    "n_pts_per_object": 500000,
    "percent_near_surface": 0.4,
    "percent_further_from_surface": 0.4,
    "sigma_near": 0.01,
    "sigma_far": 0.1,
    "random_function": "normal",
    "center_pts": True,
    # "axis_align": False,
    "normalize_pts": True,
    "scale_method": "max_rad",
    # data used in training
    "objects_per_batch": 64,
    "batch_split": 1,
    "samples_per_object_per_batch": 16384,
    "num_data_loader_threads": 16,
    "n_epochs": 2001,
    # validation data?
    "validation_split": 0.1,
    "chamfer": True,
    "emd": True,
    "val_paths": None,
    # Loss function related parameters
    "enforce_minmax": True,
    "clamp_dist": 0.1,
    # Eikonal loss parameters (for enforcing ||∇f|| = 1 constraint)
    "eikonal_weight": 0.0,  # Weight for eikonal loss (0 to disable, typically 0.1)
    "sample_difficulty_weight": None,  # curriculum deepsdf equation 6
    "sample_difficulty_cooldown": None,
    "sample_difficulty_weight_schedule": "linear",  # 'linear', 'exponential', 'exponential_plateau', 'constant'
    "surface_accuracy_e": None,  # curriculum deepsdf equation 5
    "surface_accuracy_schedule": "linear",  # 'linear', 'exponential', 'exponential_plateau', 'constant'
    "surface_accuracy_cooldown": None,
    # my tweeks on curriculum deepsdf
    "sample_difficulty_lx": None,  # use inverse exponential surface distance to weight errors (1/(sdf_gt**x)) where x = order (1, 2, .. etc)
    "sample_difficulty_lx_schedule": "linear",  # 'linear', 'exponential', 'exponential_plateau', 'constant'
    "sample_difficulty_lx_cooldown": None,  # cooldown period for sample difficulty lx weighting
    "sample_difficulty_lx_epsilon": 1e-4,  # epsilon to avoid divide by zero errors
    # optimizer
    "optimizer": "Adam",  #  "AdamW"
    "weight_decay": 0.0001,
    # Learning Rate:
    "LearningRateSchedule": [
        {"Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5},
        {"Type": "Step", "Initial": 0.001, "Interval": 500, "Factor": 0.5},
    ],
    # regularize learning:
    "grad_clip": None,
    # checkpointing:
    "checkpoint_epochs": 1000,
    "additional_checkpoints": [100, 500],
    # pytorch
    "device": "cuda:0",
    # saving results
    "experiment_directory": None,
    # initalizing latent codes:
    "latent_size": 256,
    "latent_init_std": 0.01,
    "latent_bound": 1.0,  # CodeBound from deepsdf
    # code regualrizations:
    "code_regularization": True,
    "code_regularization_weight": 1e-4,
    "code_regularization_type_prior": "spherical",  # 'spherical', 'identity', 'kld_diagonal'
    "code_cyclic_anneal": False,  # https://github.com/haofuml/cyclical_annealing - cyclic anneal the latent weight(s)
}

with open("./default_config.json", "w") as f:
    json.dump(config, f, indent=4)
