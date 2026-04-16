"""
Continuous normalized spine position mapping.

Maps each vertebra to a continuous 0-1 value representing its proportional
position along the spine, using ordinal numbers from filenames and total
vertebra counts from vertebrae_counts.csv.

Formula: normalized_position = (ordinal - 0.5) / total

This centers each vertebra in its "slot" so the first vertebra is ~0.02
and the last is ~0.98, never exactly 0 or 1.
"""

import csv
import os
import re
from typing import Dict, Optional


class SpinePositionMapper:
    """Maps mesh filenames to continuous normalized spine positions."""

    def __init__(self, csv_path: str):
        self._specimens: Dict[str, Dict] = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                specimen = row["specimen"].strip()
                if specimen.upper() == "TOTAL":
                    continue
                total = int(row["total"])
                if total <= 1:
                    continue
                self._specimens[specimen] = {
                    "cervical": int(row["cervical"]),
                    "thoracic": int(row["thoracic"]),
                    "lumbar": int(row["lumbar"]),
                    "total": total,
                }

    @property
    def specimens(self) -> Dict[str, Dict]:
        return dict(self._specimens)

    def _extract_ordinal_and_specimen(self, filename: str) -> Optional[tuple]:
        """Extract the ordinal number and specimen key from a mesh filename.

        Filenames look like:
            family_genus_species_specimenid_NN-c1.vtk
            family_genus_species_specimenid-NN-c1.vtk

        Returns (ordinal, specimen_key) or None.
        """
        base = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Extract position suffix (-c1, -t5, -l3)
        match_pos = re.search(r"-([ctlCTL])(\d+)$", base)
        if not match_pos:
            return None
        base = base[: match_pos.start()]

        # Extract ordinal from end of remaining string
        match_ord = re.search(r"[-_](\d+)$", base)
        if not match_ord:
            return None
        ordinal = int(match_ord.group(1))
        specimen_key = base[: match_ord.start()]

        # Normalize specimen key to match CSV format (underscores, lowercase)
        specimen_key = specimen_key.lower().replace("-", "_")

        return ordinal, specimen_key

    def _find_specimen(self, specimen_key: str) -> Optional[str]:
        """Find the CSV specimen name matching the given key.

        Tries exact match first, then substring matching for cases where
        the filename specimen ID doesn't exactly match the CSV key.
        """
        if specimen_key in self._specimens:
            return specimen_key

        # Try matching by checking if the CSV key starts with the filename key
        # or vice versa
        for csv_key in self._specimens:
            if csv_key == specimen_key:
                return csv_key
            # Normalize both for comparison (handle hyphens vs underscores)
            norm_csv = csv_key.replace("-", "_").lower()
            norm_key = specimen_key.replace("-", "_").lower()
            if norm_csv == norm_key:
                return csv_key

        return None

    def get_normalized_position(self, filename: str) -> Optional[float]:
        """Compute the continuous normalized position for a mesh filename.

        Returns (ordinal - 0.5) / total, or None if the specimen is not
        found in the CSV or the filename can't be parsed.
        """
        result = self._extract_ordinal_and_specimen(filename)
        if result is None:
            return None

        ordinal, specimen_key = result
        csv_key = self._find_specimen(specimen_key)
        if csv_key is None:
            return None

        total = self._specimens[csv_key]["total"]
        return (ordinal - 0.5) / total

    def get_region_boundaries(self, filename: str) -> Optional[Dict[str, float]]:
        """Get the normalized boundary positions between spinal regions.

        Returns a dict with:
            cervical_end: normalized position where cervical region ends
            thoracic_end: normalized position where thoracic region ends
        (Everything after thoracic_end is lumbar.)

        Returns None if the specimen is not found.
        """
        result = self._extract_ordinal_and_specimen(filename)
        if result is None:
            return None

        _, specimen_key = result
        csv_key = self._find_specimen(specimen_key)
        if csv_key is None:
            return None

        info = self._specimens[csv_key]
        total = info["total"]
        cervical_end = info["cervical"] / total
        thoracic_end = (info["cervical"] + info["thoracic"]) / total

        return {
            "cervical_end": cervical_end,
            "thoracic_end": thoracic_end,
        }

    def get_derived_region(self, filename: str) -> Optional[str]:
        """Derive the categorical spinal region from the continuous position.

        Uses specimen-specific boundaries from the CSV to determine whether
        a vertebra falls in the Cervical, Thoracic, or Lumbar region.
        """
        pos = self.get_normalized_position(filename)
        if pos is None:
            return None

        boundaries = self.get_region_boundaries(filename)
        if boundaries is None:
            return None

        if pos < boundaries["cervical_end"]:
            return "Cervical"
        elif pos < boundaries["thoracic_end"]:
            return "Thoracic"
        else:
            return "Lumbar"

    def get_ordinal(self, filename: str) -> Optional[int]:
        """Extract just the ordinal number from a filename."""
        result = self._extract_ordinal_and_specimen(filename)
        if result is None:
            return None
        return result[0]

    def get_total(self, filename: str) -> Optional[int]:
        """Get the total vertebra count for the specimen in a filename."""
        result = self._extract_ordinal_and_specimen(filename)
        if result is None:
            return None
        _, specimen_key = result
        csv_key = self._find_specimen(specimen_key)
        if csv_key is None:
            return None
        return self._specimens[csv_key]["total"]
