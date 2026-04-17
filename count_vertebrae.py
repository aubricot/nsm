#!/usr/bin/env python3
"""
Count vertebrae types (cervical, thoracic, lumbar) for each specimen in the vertebrae_meshes directory.

Filename structure: {family}_{genus}_{species}_{specimen_id}_{file_num}-{vertebrae_type}{vertebrae_number}.vtk
Specimen is the combination of: family, genus, species, and specimen_id
Vertebrae types: c = cervical, t = thoracic, l = lumbar
"""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse a vertebrae mesh filename to extract specimen ID and vertebrae type.
    
    Returns:
        Tuple of (specimen_id, vertebrae_type) or None if parsing fails.
    """
    # Remove .vtk extension
    if filename.endswith('.vtk'):
        filename = filename[:-4]
    
    # Pattern 1: {family}_{genus}_{species}_{specimen_id}_{file_num}-{vertebrae_type}{vertebrae_number}
    # The vertebrae type is right after the hyphen: c, t, or l
    match = re.match(r'^(.+?)_(\d+)-([ctl])(\d+)$', filename)
    
    if match:
        specimen_part = match.group(1)  # everything before _{file_num}-
        vertebrae_type = match.group(3)  # c, t, or l
        return specimen_part, vertebrae_type
    
    # Pattern 2: Handle files with format like: {family}_{specimen}-{filenum}-{vertebrae_type}{number}
    # Example: cordylidae_cham-aenea-26-l10.vtk
    match2 = re.match(r'^(.+?)-(\d+)-([ctl])(\d+)$', filename)
    if match2:
        specimen_part = match2.group(1)  # everything before -{file_num}-
        vertebrae_type = match2.group(3)  # c, t, or l
        return specimen_part, vertebrae_type
    
    # Pattern 3: Try alternative pattern for files with different naming
    # Some files might have format like: family_genus_species_specimen_filenum-type
    parts = filename.rsplit('-', 1)
    if len(parts) == 2:
        specimen_and_num = parts[0]
        type_and_num = parts[1]
        
        # Extract vertebrae type (first character should be c, t, or l)
        if type_and_num and type_and_num[0].lower() in ['c', 't', 'l']:
            vertebrae_type = type_and_num[0].lower()
            
            # Extract specimen by removing the file number from the end
            # Pattern: specimen_part_filenum
            specimen_match = re.match(r'^(.+?)_(\d+)$', specimen_and_num)
            if specimen_match:
                specimen_id = specimen_match.group(1)
                return specimen_id, vertebrae_type
    
    return None


def count_vertebrae(directory: str) -> Dict[str, Dict[str, int]]:
    """
    Count vertebrae types for each specimen in the directory.
    
    Returns:
        Dictionary mapping specimen IDs to counts of each vertebrae type.
    """
    counts = defaultdict(lambda: {'cervical': 0, 'thoracic': 0, 'lumbar': 0})
    type_map = {'c': 'cervical', 't': 'thoracic', 'l': 'lumbar'}
    
    skipped_files = []
    
    for filename in os.listdir(directory):
        if not filename.endswith('.vtk'):
            continue
        
        result = parse_filename(filename)
        if result:
            specimen_id, vertebrae_type = result
            full_type = type_map.get(vertebrae_type)
            if full_type:
                counts[specimen_id][full_type] += 1
        else:
            skipped_files.append(filename)
    
    if skipped_files:
        print(f"\nWarning: Could not parse {len(skipped_files)} files:")
        for f in skipped_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    
    return dict(counts)


def main():
    # Path to vertebrae meshes directory
    script_dir = Path(__file__).parent
    meshes_dir = script_dir / "vertebrae_meshes"
    output_csv = script_dir / "vertebrae_counts.csv"
    
    if not meshes_dir.exists():
        print(f"Error: Directory not found: {meshes_dir}")
        return
    
    print(f"Analyzing vertebrae meshes in: {meshes_dir}\n")
    print("=" * 80)
    
    counts = count_vertebrae(str(meshes_dir))
    
    # Sort specimens alphabetically
    sorted_specimens = sorted(counts.keys())
    
    # Print header
    print(f"\n{'Specimen':<60} {'Cervical':>10} {'Thoracic':>10} {'Lumbar':>10} {'Total':>10}")
    print("-" * 100)
    
    # Prepare CSV data
    csv_rows = []
    
    # Print counts for each specimen
    total_cervical = 0
    total_thoracic = 0
    total_lumbar = 0
    
    for specimen in sorted_specimens:
        c = counts[specimen]['cervical']
        t = counts[specimen]['thoracic']
        l = counts[specimen]['lumbar']
        total = c + t + l
        
        total_cervical += c
        total_thoracic += t
        total_lumbar += l
        
        print(f"{specimen:<60} {c:>10} {t:>10} {l:>10} {total:>10}")
        csv_rows.append([specimen, c, t, l, total])
    
    # Print summary
    print("-" * 100)
    grand_total = total_cervical + total_thoracic + total_lumbar
    print(f"{'TOTAL':<60} {total_cervical:>10} {total_thoracic:>10} {total_lumbar:>10} {grand_total:>10}")
    print(f"\nTotal specimens: {len(sorted_specimens)}")
    
    # Write CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['specimen', 'cervical', 'thoracic', 'lumbar', 'total'])
        writer.writerows(csv_rows)
        writer.writerow(['TOTAL', total_cervical, total_thoracic, total_lumbar, grand_total])
    
    print(f"\nCSV output saved to: {output_csv}")


if __name__ == "__main__":
    main()
