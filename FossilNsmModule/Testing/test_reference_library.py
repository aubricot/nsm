import os
import shutil
import sys
import tempfile
import unittest

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MODULE_DIR, "utils"))

import reference_library as rl

# one fake manifest row, reused across tests
SAMPLE = [{
    "lookup_key": "agamidae_agama_atra_uf180711_02_c4",
    "mesh_name": "Agamidae_Agama_atra_UF180711_02-C4_align.vtk",
    "hf_path": "Agamidae_Agama_atra_UF180711_02-C4_draco.glb",
}]


class TestLookupKey(unittest.TestCase):
    def test_vtk_and_glb_names_match(self):
        # the .vtk name the model emits and the .glb in the dataset must key the same
        vtk = rl.compute_lookup_key("agamidae_agama_atra_uf180711_02-c4_align.vtk")
        glb = rl.compute_lookup_key("Agamidae_Agama_atra_UF180711_02-C4_draco.glb")
        self.assertEqual(vtk, glb)
        self.assertEqual(vtk, "agamidae_agama_atra_uf180711_02_c4")

    def test_idempotent(self):
        k = rl.compute_lookup_key("Agamidae_Agama_atra_UF180711_02-C4_draco.glb")
        self.assertEqual(rl.compute_lookup_key(k), k)


class TestManifest(unittest.TestCase):
    def test_lookup(self):
        m = rl.ReferenceManifest(SAMPLE)
        row = m.lookup("agamidae_agama_atra_uf180711_02-c4_align.vtk")
        self.assertEqual(row["hf_path"], "Agamidae_Agama_atra_UF180711_02-C4_draco.glb")
        self.assertIsNone(m.lookup("something_else_01-c3_align.vtk"))

    def test_derive_from_listing(self):
        # a dataset with no manifest.csv: build the map from the .glb file names
        m = rl.ReferenceManifest.from_hf_paths(
            ["Agamidae_Agama_atra_UF180711_02-C4_draco.glb", "readme.txt"])
        self.assertEqual(len(m.rows()), 1)


class TestLocalFolder(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_resolves_by_key(self):
        # file on disk is .ply, the model asks for .vtk - should still match by key
        open(os.path.join(self.root, "foo_align.ply"), "w").close()
        backend = rl.LocalFolderBackend(self.root)
        self.assertEqual(backend.resolve("foo_align.vtk").kind, "path")

    def test_missing(self):
        backend = rl.LocalFolderBackend(self.root)
        self.assertEqual(backend.resolve("nope.vtk").kind, "missing")


class TestHuggingFace(unittest.TestCase):
    def test_needs_a_repo(self):
        self.assertEqual(rl.HuggingFaceBackend().resolve("x.vtk").state, rl.STATE_NO_REPO)

    def test_downloads_the_name_the_repo_actually_has(self):
        # manifest has a mixed-case name but the repo ships lowercase, and HF paths
        # are case-sensitive, so the listing's real name has to win
        backend = rl.HuggingFaceBackend(repo_id="test/repo",
                                        manifest=rl.ReferenceManifest(SAMPLE))
        name = "Agamidae_Agama_atra_UF180711_02-C4_align.vtk"
        real = "agamidae_agama_atra_uf180711_02-c4_draco.glb"
        backend._repo_index = {rl.compute_lookup_key(name): real}
        self.assertEqual(backend._hf_path_for(name, backend.manifest().lookup(name)), real)

    def test_picks_manifest_csv(self):
        self.assertEqual(rl._pick_manifest_file(["manifest.csv", "a_draco.glb"]), "manifest.csv")
        self.assertEqual(rl._pick_manifest_file(["manifest_v44.csv"]), "manifest_v44.csv")
        self.assertIsNone(rl._pick_manifest_file(["a_draco.glb"]))


class TestToken(unittest.TestCase):
    def test_reads_env(self):
        self.assertEqual(rl.resolve_hf_token(env={"HF_TOKEN": " tok "}), "tok")

    def test_raises_when_none_found(self):
        with self.assertRaises(ValueError):
            rl.resolve_hf_token(env={}, stored_path=None, cached_token_provider=lambda: None)


if __name__ == "__main__":
    unittest.main()
