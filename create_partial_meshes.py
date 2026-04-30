# Create partial meshes to use for shape completion validation against original/ground truth meshes
# Subtract segment PLY files generated using ATLAS from original/ground truth meshes

import os
import vtk
import json
import argparse
import random
import pyvista as pv
from scipy.spatial import cKDTree

def save_ply(polydata, path, binary=True):
    if not isinstance(polydata, pv.PolyData):
        polydata = pv.wrap(polydata)
    polydata.save(path, binary=binary)

def clean_triangulate(pd):
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(pd); cl.ConvertStripsToPolysOn(); cl.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cl.GetOutputPort()); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(tri.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    
    return pv.wrap(conn.GetOutput())

def fast_subtract(original_pd, segment_pd, eps=0.02):
    # Delete faces near segment to remove
    orig = pv.wrap(original_pd)
    seg = pv.wrap(segment_pd)
    # Get face centers
    orig.compute_cell_sizes(length=False, area=False, volume=False)
    face_centers = orig.cell_centers().points
    # Find faces near segment
    tree = cKDTree(seg.points)
    dists, _ = tree.query(face_centers, k=1)
    # Keep faces far from segment
    orig['keep_face'] = (dists > eps).astype(int)
    result = orig.threshold(0.5, scalars='keep_face', invert=False)
    return result.extract_surface(algorithm='dataset_surface').clean().triangulate()

def flip_to_ras(polydata):
    # LPS to RAS coords (x flip)
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()

def create_partial_mesh(original_ply, segment_ply, output_ply):
    original = pv.wrap(clean_triangulate(pv.read(original_ply)))    
    segment = pv.wrap(clean_triangulate(pv.read(segment_ply)))
    segment = pv.wrap(flip_to_ras(segment))
    # Debug: colored mesh by pt distances
    #original_with_dist = original.compute_implicit_distance(segment, inplace=False)
    #original_with_dist.save(output_ply.replace(".ply", "_distances.vtp"))
    partial = fast_subtract(original, segment, eps=0.01)
    print(f"  Original: {original.GetNumberOfPoints()} vertices")
    print(f"  Segment:  {segment.GetNumberOfPoints()} vertices")
    print(f"  Partial:  {partial.GetNumberOfPoints()} vertices")
    save_ply(partial, output_ply)
    print(f"  ✓ Saved: {output_ply}")
    return partial.GetNumberOfPoints()

def create_validation_dataset(original_dir, segments_dir, output_dir, num_rand_to_remove=1, segment_ids_to_remove=[], total_num_segments=7):
    # Create output directory for partial meshes
    partial_dir = os.path.join(output_dir, "partial_meshes")
    os.makedirs(partial_dir, exist_ok=True)
    # Loop over input meshes
    results = []
    for specimen_basename in os.listdir(original_dir):
        print(f"\n{'='*60}")
        print(f"Processing: {specimen_basename}") 
        # Determine segments to remove
        specimen_name = os.path.splitext(specimen_basename)[0]
        if num_rand_to_remove:
            seg_ids = random.sample(list(range(total_num_segments)), k=num_rand_to_remove)
        elif segment_ids_to_remove:
            seg_ids = segment_ids_to_remove
        else:
            raise ValueError("num_rand_to_remove or segment_ids_to_remove must be valid.")
        # Remove segments
        try:
            result = {
                'base_name': specimen_name,
                'ground_truth': specimen_basename,  # Just reference the original
                'partial': os.path.join(partial_dir, f"{specimen_name}_partial.ply"),
                'removed_segment_ids': [],  # IDs of segments which were removed from original
                'removed_segments': [],  # Files of segments which were removed from original
                'partial_vertices': None
            }
            # Create partial mesh by removing each segment recursively
            in_path = os.path.join(original_dir, specimen_basename)
            for seg_id in seg_ids:
                # Check if segment exists for this specimen
                segment_path = os.path.join(segments_dir, specimen_name + f"_seg_{seg_id:02d}.ply")
                if not os.path.exists(segment_path):
                    continue
                # Remove segment
                n_vertices = create_partial_mesh(in_path, segment_path, result['partial'])
                print(f"  Removed segment: {seg_id}")
                result['partial_vertices'] = n_vertices
                result['removed_segment_ids'].append(seg_id)
                result['removed_segments'].append(segment_path)
                in_path = result['partial'] # Recursively remove segments
            results.append(result)
            print(f"  ✓ Success! Removed segments: {result['removed_segment_ids']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'segments_removed': f"{num_rand_to_remove} random segments per mesh" if num_rand_to_remove else segment_ids_to_remove,
        'n_meshes': len(results),
        'original_dir': os.path.abspath(original_dir),
        'segments_dir': os.path.abspath(segments_dir),
        'meshes': results
    }
    summary_path = os.path.join(output_dir, "partial_meshing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Created {len(results)} partial meshes")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"\nDirectories:")
    print(f"  - Partial meshes (input to shape completion): {partial_dir}")
    print(f"  - Ground truth (compare with output):        {original_dir}")
    print(f"  - Removed segments (for reference):          {segments_dir}")
    print(f"{'='*70}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Create partial meshes for shape completion validation")
    parser.add_argument("original_dir", 
                       help="Directory with original meshes (ground truth)")
    parser.add_argument("segments_dir",
                       help="Directory with segment PLYs (*_seg_XX.ply)")
    parser.add_argument("output_dir",
                       help="Output directory for partial meshes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num_rand_segs", type=int, default=0,
                       help="Number of random segments to remove from each mesh (default: 0)")
    group.add_argument("--seg_ids", type=str, default="6",
                       help="Comma separated segment ids to remove (default: 6). If --num_rand_segs is provided, --seg_ids is ignored.")
    parser.add_argument("--total_segs", type=int, default=7,
                       help="Total number of segments per mesh (default: 7)")
    args = parser.parse_args()
    create_validation_dataset(
        args.original_dir,
        args.segments_dir,
        args.output_dir,
        num_rand_to_remove=args.num_rand_segs,
        segment_ids_to_remove=[int(id) for id in str.split(args.seg_ids, ",") if id],
        total_num_segments=args.total_segs)

if __name__ == "__main__":
    main()