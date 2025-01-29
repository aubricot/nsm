import torch 
import os
import math
import json
try:
    import schedulefree
except ImportError:
    print("schedulefree not found, skipping import")
    schedulefree = None
import warnings


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))

class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length

class LogAnnealLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, final, n_epochs):
        self.initial = initial
        self.final = final
        self.n_epochs = n_epochs

    def get_learning_rate(self, epoch):
        return self.initial * math.exp(math.log(self.final / self.initial) * epoch / self.n_epochs)

def get_learning_rate_schedules(config):

    schedule_specs = config["LearningRateSchedule"]

    schedules = []

    for schedule_spec in schedule_specs:

        if schedule_spec["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_spec["Interval"],
                    schedule_spec["Factor"],
                )
            )
        elif schedule_spec["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_spec["Final"],
                    schedule_spec["Length"],
                )
            )
        elif schedule_spec["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_spec["Value"]))
        
        elif schedule_spec["Type"] == "LogAnneal":
            schedules.append(LogAnnealLearningRateSchedule(
                schedule_spec["Initial"],
                schedule_spec["Final"],
                config["n_epochs"],
            ))

        else:
            raise ValueError(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_spec["Type"]
                )
            )

    return schedules

def adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False):
    if verbose is True:
        print("optimizer param groups: ", optimizer.param_groups)
        print('lr_schedules: ', lr_schedules)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

def save_latent_vectors(config, epoch, latent_vec, latent_codes_subdir="latent_codes"):
    filename = f'{epoch}.pth'
    folder_save = os.path.join(config['experiment_directory'], latent_codes_subdir)
    if not os.path.exists(folder_save):
        os.makedirs(folder_save, exist_ok=True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(folder_save, filename),
    )

def save_model(config, epoch, decoder, model_subdir="model", optimizer=None):
    if type(decoder) not in (list, tuple):
        decoder = [decoder]
    
    filename = f'{epoch}.pth'
    
    for decoder_idx, decoder_ in enumerate(decoder):
        if len(decoder) > 1:
            model_subdir_ = model_subdir + f'_{decoder_idx}'
        else:
            model_subdir_ = model_subdir
            
        folder_save = os.path.join(config['experiment_directory'], model_subdir_)
        if not os.path.exists(folder_save):
            os.makedirs(folder_save, exist_ok=True)
        
        dict_ = {
            "epoch": epoch, 
            "model": decoder_.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else "None"
        }
        
        torch.save(
            dict_,
            os.path.join(folder_save, filename),
        )
    
def save_model_params(config, list_mesh_paths):

    if not os.path.exists(config['experiment_directory']):
        os.makedirs(config['experiment_directory'], exist_ok=True)

    path_save = os.path.join(config['experiment_directory'], 'model_params_config.json')

    if os.path.exists(path_save):
        return

    dict_save = {
        "list_mesh_paths": list_mesh_paths,
    }
    dict_save.update(config)

    dict_save = filter_non_jsonable(dict_save)

    with open(path_save, 'w') as f:
        json.dump(dict_save, f, indent=4)

def get_checkpoints(config):
    checkpoints = list(
        range(
            config['checkpoint_epochs'],
            config['n_epochs'] + 1,
            config['checkpoint_epochs'],
        )
    )

    for checkpoint in config['additional_checkpoints']:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    return checkpoints

def get_latent_vecs(num_objects, config):
    if ('variational' in config) and (config['variational'] is True):
        latent_size = config['latent_size'] * 2
        latent_bound = 1000
    else:
        latent_size = config['latent_size']
        latent_bound = config['latent_bound']

    lat_vecs = torch.nn.Embedding(num_objects, latent_size, max_norm=latent_bound)
    
    if ('latent_init_normal' in config) and (config['latent_init_normal'] is True):
        torch.nn.init.normal_(
            lat_vecs.weight.data,
            0.0,
            config['latent_init_std'] / math.sqrt(latent_size),
        )

    return lat_vecs

def get_optimizer(model, latent_vecs, lr_schedules, optimizer="Adam", weight_decay=0.0001):
    if type(model) not in (list, tuple):
        model = [model]
    
    list_params = [
        {
            "params": latent_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        }
    ]
    for model_ in model:
        list_params.append(
            {
                "params": model_.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            }
        )

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(list_params)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(list_params, weight_decay=weight_decay)
    elif optimizer == "schedule_free_AdamW":
        if schedulefree is None:
            raise ImportError("schedulefree not imported, because not installed")
        optimizer = schedulefree.AdamWScheduleFree(list_params, weight_decay=weight_decay)
    elif optimizer == "schedule_free_SGD":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return optimizer

def symmetric_chammfer(p1, p2, n_pts):
    """
    """
    pass


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def filter_non_jsonable(dict_obj):
    return {k: v for k, v in dict_obj.items() if is_jsonable(v)}

def print_gpu_memory():
    # assert cuda is available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        print('CUDA, GPU Usage:')
        print(f"\tAllocated memory: {allocated / 1024**3:.2f} GB")
        print(f"\tCached memory: {cached / 1024**3:.2f} GB")
    else:
        print('CUDA not available - GPU stats not available')
    

def clear_gpu_cache(device):
    if 'cuda' in device:
        torch.cuda.empty_cache()
    elif 'mps' in device:
        torch.mps.empty_cache()
    else:
        warnings.warn('Not clearing cache because not cuda or mps (apple metal)')
