"""
VTK filename parsing logic for taxonomic data.

Parses VTK filenames into taxonomic components (family, genus, species, etc.)

Format: {family}_{genus}_{species}_{specimen_id}_{sequence}-{vertebra}.vtk
Example: agamidae_agama_atra_uf180711_06-t1.vtk
"""
import os
import re


# Regex: capture specimen prefix (family_genus_species_specimenid) and vertebra
# Matches filenames like: agamidae_agama_atra_uf180711_01-c3.vtk
_FNAME_PAT = re.compile(
    r"^(?P<specimen>[\w\s\-]+)[\-_ ]+[\w\d]+[\-_ ]+(?P<vertebra>[CTL]?\d+)",
    re.IGNORECASE
)


def parse_vtk_filename(path):
    """
    Parse a VTK filename into taxonomic components via regex.

    Format: {family}_{genus}_{species}_{specimen_id}_{num}-{vertebra}.vtk
    Example: agamidae_agama_atra_uf180711_06-t1.vtk

    Args:
        path: str, path or filename of VTK file

    Returns:
        dict with keys: family, genus, species
    """
    fname = os.path.basename(path)
    m = _FNAME_PAT.match(fname)
    if not m:
        return {'family': None, 'genus': None, 'species': None}

    # Split specimen prefix by underscore to get taxonomic components
    parts = [p.strip() for p in m.group('specimen').split('_') if p.strip()]
    return {
        'family':  parts[0] if len(parts) > 0 else None,
        'genus':   parts[1] if len(parts) > 1 else None,
        'species': parts[2] if len(parts) > 2 else None,
    }


def add_filename_exception(pattern, replacement):
    """
    Add a new filename exception pattern.

    Args:
        pattern: str, the pattern to match (e.g., 'genus-species')
        replacement: str, the corrected pattern (e.g., 'genus_species')
    """
    FILENAME_EXCEPTIONS[pattern] = replacement
