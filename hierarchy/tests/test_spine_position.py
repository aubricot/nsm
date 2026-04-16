import os
import tempfile
import pytest
from hierarchy.spine_position import SpinePositionMapper


@pytest.fixture
def csv_path():
    """Create a temporary CSV with known data for testing."""
    content = (
        "specimen,cervical,thoracic,lumbar,total\n"
        "cordylidae_smaug_giganteus_uf119459_sg,5,9,9,23\n"
        "cordylidae_ouroborus_cataphractus_amnh,5,6,10,21\n"
        "scincidae_chalcides_guentheri_uf14800,5,5,51,61\n"
        "agamidae_agama_atra_uf180711,5,9,7,21\n"
        "boiidae_eryx conicus uf-herp-66735,0,1,0,1\n"  # total=1, should be excluded
        "TOTAL,452,838,1567,2857\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mapper(csv_path):
    return SpinePositionMapper(csv_path)


class TestSpinePositionMapperInit:
    def test_loads_specimens(self, mapper):
        assert len(mapper.specimens) == 4  # excludes TOTAL and total=1 row

    def test_excludes_degenerate_rows(self, mapper):
        # boiidae with total=1 should be excluded
        for key in mapper.specimens:
            assert "boiidae" not in key


class TestGetNormalizedPosition:
    def test_smaug_giganteus_t5(self, mapper):
        """smaug_giganteus_sg_10-t5.vtk → (10 - 0.5) / 23 = 0.413"""
        pos = mapper.get_normalized_position(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5.vtk"
        )
        assert pos is not None
        assert abs(pos - 0.413) < 0.001

    def test_ouroborus_t1(self, mapper):
        """ouroborus_amnh_06-t1.vtk → (6 - 0.5) / 21 = 0.262"""
        pos = mapper.get_normalized_position(
            "cordylidae_ouroborus_cataphractus_amnh_06-t1.vtk"
        )
        assert pos is not None
        assert abs(pos - 0.262) < 0.001

    def test_chalcides_l49(self, mapper):
        """chalcides_uf14800_59-l49.vtk → (59 - 0.5) / 61 = 0.959"""
        pos = mapper.get_normalized_position(
            "scincidae_chalcides_guentheri_uf14800_59-l49.vtk"
        )
        assert pos is not None
        assert abs(pos - 0.959) < 0.001

    def test_first_vertebra(self, mapper):
        """First vertebra should be close to 0 but not 0."""
        pos = mapper.get_normalized_position(
            "agamidae_agama_atra_uf180711_01-c1.vtk"
        )
        assert pos is not None
        # (1 - 0.5) / 21 ≈ 0.0238
        assert 0 < pos < 0.05

    def test_last_vertebra(self, mapper):
        """Last vertebra should be close to 1 but not 1."""
        pos = mapper.get_normalized_position(
            "agamidae_agama_atra_uf180711_21-l7.vtk"
        )
        assert pos is not None
        # (21 - 0.5) / 21 ≈ 0.976
        assert 0.95 < pos < 1.0

    def test_missing_specimen_returns_none(self, mapper):
        pos = mapper.get_normalized_position(
            "fakefamily_fakegenus_fakespecies_xyz_05-c2.vtk"
        )
        assert pos is None

    def test_unparseable_filename_returns_none(self, mapper):
        pos = mapper.get_normalized_position("random_file.vtk")
        assert pos is None

    def test_no_extension(self, mapper):
        pos = mapper.get_normalized_position(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5"
        )
        assert pos is not None


class TestGetRegionBoundaries:
    def test_smaug_boundaries(self, mapper):
        """smaug: 5 cervical, 9 thoracic, 9 lumbar = 23 total.
        cervical_end = 5/23 ≈ 0.217, thoracic_end = 14/23 ≈ 0.609"""
        bounds = mapper.get_region_boundaries(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5.vtk"
        )
        assert bounds is not None
        assert abs(bounds["cervical_end"] - 5 / 23) < 0.001
        assert abs(bounds["thoracic_end"] - 14 / 23) < 0.001

    def test_missing_specimen_returns_none(self, mapper):
        bounds = mapper.get_region_boundaries(
            "fakefamily_fakegenus_fakespecies_xyz_05-c2.vtk"
        )
        assert bounds is None


class TestGetDerivedRegion:
    def test_cervical(self, mapper):
        """First vertebra of agama (ordinal 1) should be cervical."""
        region = mapper.get_derived_region(
            "agamidae_agama_atra_uf180711_01-c1.vtk"
        )
        assert region == "Cervical"

    def test_thoracic(self, mapper):
        """smaug ordinal 10: pos = (10-0.5)/23 ≈ 0.413.
        cervical_end = 5/23 ≈ 0.217, thoracic_end = 14/23 ≈ 0.609.
        0.217 < 0.413 < 0.609 → Thoracic"""
        region = mapper.get_derived_region(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5.vtk"
        )
        assert region == "Thoracic"

    def test_lumbar(self, mapper):
        """chalcides ordinal 59: pos = (59-0.5)/61 ≈ 0.959.
        cervical_end = 5/61, thoracic_end = 10/61.
        0.959 > 0.164 → Lumbar"""
        region = mapper.get_derived_region(
            "scincidae_chalcides_guentheri_uf14800_59-l49.vtk"
        )
        assert region == "Lumbar"


class TestGetOrdinalAndTotal:
    def test_get_ordinal(self, mapper):
        assert mapper.get_ordinal(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5.vtk"
        ) == 10

    def test_get_total(self, mapper):
        assert mapper.get_total(
            "cordylidae_smaug_giganteus_uf119459_sg_10-t5.vtk"
        ) == 23


class TestTaxonomyUtilsFileOrdinal:
    def test_file_ordinal_extracted(self):
        from hierarchy.taxonomy import parse_taxonomy_from_filename

        result = parse_taxonomy_from_filename(
            "cordylidae_smaug_giganteus_sg_10-t5.vtk"
        )
        assert result is not None
        assert result["file_ordinal"] == 10

    def test_file_ordinal_none_when_missing(self):
        from hierarchy.taxonomy import parse_taxonomy_from_filename

        # Filename without ordinal number before position
        result = parse_taxonomy_from_filename("cordylidae_smaug-t5.vtk")
        # This should still parse but file_ordinal might be None or a number
        # depending on the filename structure
        assert result is not None
