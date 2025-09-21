# Multi-Surface Rigid Registration

This document describes the new multi-surface rigid registration functionality added to the NSM dataset code.

## Overview

The rigid registration code has been updated to support registering to multiple surfaces simultaneously instead of just one. This is useful for cases where you want to register to anatomically related surfaces together, such as:

- Medial + lateral menisci
- Multiple cartilage regions
- Bone + cartilage combined

## Key Changes

### 1. New `combine_meshes()` Function

```python
def combine_meshes(meshes, mesh_indices):
    """
    Combine multiple meshes into a single mesh using Mesh addition operator.
    
    Args:
        meshes (list): List of Mesh objects
        mesh_indices (list or int): Indices of meshes to combine
    
    Returns:
        Mesh: Combined mesh object
    """
```

This function uses the Mesh `+` operator to combine multiple mesh surfaces into a single mesh for registration.

### 2. Updated `mesh_to_scale` Parameter

The `mesh_to_scale` parameter in `read_meshes_get_sampled_pts()` now accepts:

- **Integer** (original behavior): Uses single mesh for registration
- **List of integers** (new): Combines multiple meshes for joint registration

### 3. Enhanced Registration Logic

```python
if isinstance(mesh_to_scale, (list, tuple)):
    print(f'Registering to multiple surfaces: {mesh_to_scale}')
    # Combine multiple meshes for registration
    combined_mesh = combine_meshes(orig_meshes, mesh_to_scale)
    registration_mesh = combined_mesh
else:
    # Single mesh registration (original behavior)
    registration_mesh = orig_meshes[mesh_to_scale]

icp_transform = registration_mesh.rigidly_register(...)
```

### 4. Updated Scaling and Centering

The scaling and centering logic has been enhanced to handle multiple reference surfaces when `mesh_to_scale` is a list.

### 5. Enhanced Mean Mesh Creation

The mean mesh creation logic in `reconstruct_mesh()` has been updated to support multi-surface mean meshes:

```python
if isinstance(mesh_to_scale, (list, tuple)):
    print(f'Combining mean meshes for multi-surface registration: {mesh_to_scale}')
    # Combine multiple mean meshes for registration
    mean_mesh = combine_meshes(mean_mesh, mesh_to_scale)
else:
    # Single mesh selection (original behavior)
    mean_mesh = mean_mesh[mesh_to_scale]
```

### 6. Updated Reference Mesh Loading

The `load_reference_mesh()` method now supports creating reference meshes from multiple surfaces:

```python
if isinstance(self.mesh_to_scale, (list, tuple)):
    # When mesh_to_scale is a list, create reference mesh by combining multiple surfaces
    meshes = [Mesh(self.list_mesh_paths[self.reference_mesh][idx]) for idx in self.mesh_to_scale]
    self.reference_mesh = combine_meshes(meshes, list(range(len(meshes))))
```

## Usage Examples

### Single Surface Registration (Original Behavior)

```python
result = read_meshes_get_sampled_pts(
    paths=mesh_paths,
    mesh_to_scale=0,  # Register to first mesh only
    register_to_mean_first=True,
    mean_mesh=mean_mesh
)
```

### Multi-Surface Registration (New)

```python
result = read_meshes_get_sampled_pts(
    paths=mesh_paths,
    mesh_to_scale=[1, 2],  # Register to meshes 1 and 2 combined
    register_to_mean_first=True,
    mean_mesh=mean_mesh
)
```

### Common Scenarios

1. **Medial + Lateral Menisci**:
   ```python
   mesh_to_scale=[1, 2]  # indices for medial and lateral menisci
   ```

2. **Bone + Cartilage**:
   ```python
   mesh_to_scale=[0, 3]  # indices for bone and cartilage
   ```

3. **Multiple Cartilage Regions**:
   ```python
   mesh_to_scale=[1, 2, 3]  # indices for different cartilage regions
   ```

## Technical Details

### Mesh Addition

Since Mesh objects support the `+` operator, the combination is very simple:

```python
# Start with the first mesh and add subsequent meshes
combined_mesh = meshes[mesh_indices[0]]

for idx in mesh_indices[1:]:
    if meshes[idx] is not None:
        combined_mesh = combined_mesh + meshes[idx]
```

This is much simpler than the original VTK append approach and leverages the built-in functionality of Mesh objects.

### Consistency Across Pipeline

The multi-surface support has been implemented consistently across all stages:

1. **Mean mesh creation**: Uses same combination logic
2. **Reference mesh loading**: Combines multiple reference surfaces
3. **Registration**: Uses combined meshes for ICP
4. **Scaling/centering**: Handles multiple reference surfaces

### Backward Compatibility

The changes are fully backward compatible. Existing code using `mesh_to_scale=0` (integer) will continue to work exactly as before.

### Benefits

1. **Improved Registration**: Joint registration of related surfaces can provide better alignment than registering to a single surface
2. **Anatomical Accuracy**: Multiple related structures can be considered together during registration
3. **Flexibility**: Can combine any number of surfaces for registration
4. **Robustness**: More landmarks available for ICP registration when multiple surfaces are used
5. **Consistency**: Same logic applied across mean mesh creation, reference mesh loading, and registration

## Files Modified

- `NSM/datasets/sdf_dataset.py`:
  - Added `combine_meshes()` function
  - Updated `read_meshes_get_sampled_pts()` registration logic
  - Enhanced scaling and centering for multiple surfaces
  - Updated `load_reference_mesh()` for multi-surface support
  - Updated docstrings and class documentation

- `NSM/reconstruct/main.py`:
  - Added `combine_meshes` import
  - Updated mean mesh creation logic for multi-surface support
  - Enhanced mean mesh selection to handle list inputs

## Testing

A test script `test_multi_surface_registration.py` has been provided to demonstrate the new functionality and show usage examples.

## Migration Guide

### For Existing Code

No changes needed! Existing code using integer `mesh_to_scale` values will continue to work.

### For New Multi-Surface Registration

Simply change:
```python
mesh_to_scale=0  # Single surface
```

To:
```python
mesh_to_scale=[0, 1]  # Multiple surfaces
```

The rest of your code remains the same. The mean mesh creation and reference mesh loading will automatically adapt to use the same multi-surface approach. 