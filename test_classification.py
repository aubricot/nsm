import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import classification_slicer


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _args(self):
        return argparse.Namespace(
            config=os.path.join(self.temp_dir, "model_params_config.json"),
            model=os.path.join(self.temp_dir, "model.pth"),
            latent_codes=os.path.join(self.temp_dir, "latent_codes.pth"),
            input_mesh=os.path.join(self.temp_dir, "input.vtk"),
            output_dir=self.temp_dir,
            result=os.path.join(self.temp_dir, "top5.json"),
            iterations=1,
            learning_rate=1e-3,
        )

    @patch("NSM.optimization.optimize_latent")
    @patch("NSM.optimization.get_top_k_pcs")
    @patch("NSM.helper_funcs.load_model_and_latents")
    @patch("NSM.helper_funcs.load_config")
    @patch("NSM.datasets.SDFSamples")
    def test_classify_writes_ranked_top_five_json(
        self,
        mock_dataset,
        mock_load_config,
        mock_load_model_and_latents,
        mock_get_top_k_pcs,
        mock_optimize_latent,
    ):
        """The worker stores five filename/distance matches in ascending order."""
        args = self._args()
        mock_load_config.return_value = {
            "latent_size": 2,
            "samples_per_object_per_batch": 10,
            "n_pts_per_object": 10,
            "percent_near_surface": 0.5,
            "percent_further_from_surface": 0.5,
            "sigma_near": 0.01,
            "sigma_far": 0.1,
            "random_function": "uniform",
            "center_pts": True,
            "normalize_pts": True,
            "scale_method": "radius",
            "verbose": False,
            "cache": False,
            "equal_pos_neg": True,
            "fix_mesh": False,
            "list_mesh_paths": ["/references/mesh_{}.vtk".format(index) for index in range(6)],
        }
        latent_codes = torch.tensor([
            [0.99, 0.01], [0.90, 0.10], [0.70, 0.30],
            [0.20, 0.80], [0.00, 1.00], [-1.00, 0.00],
        ])
        mock_load_model_and_latents.return_value = (MagicMock(), None, latent_codes)
        mock_get_top_k_pcs.return_value = (None, 2)
        mock_dataset.return_value.__getitem__.return_value = (
            {"xyz": torch.zeros(10, 3), "gt_sdf": torch.zeros(10, 1)}, None
        )
        mock_optimize_latent.return_value = torch.tensor([[1.0, 0.0]])

        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            result = json.load(stream)
        matches = result["matches"]
        self.assertEqual(result["metric"], "cosine distance")
        self.assertEqual(len(matches), 5)
        self.assertEqual([match["rank"] for match in matches], [1, 2, 3, 4, 5])
        self.assertEqual([match["mesh_name"] for match in matches],
                         ["mesh_0.vtk", "mesh_1.vtk", "mesh_2.vtk", "mesh_3.vtk", "mesh_4.vtk"])
        self.assertEqual([match["latent_index"] for match in matches], [0, 1, 2, 3, 4])
        self.assertEqual(sorted(match["cosine_distance"] for match in matches),
                         [match["cosine_distance"] for match in matches])
        mock_optimize_latent.assert_called_once()

    @patch("pyvista.read")
    def test_prepare_mesh_converts_ply_into_result_directory(self, mock_read):
        mesh_path = os.path.join(self.temp_dir, "input.ply")
        converted_mesh = MagicMock()
        mock_read.return_value = converted_mesh

        output = classification_slicer._prepare_mesh(mesh_path, self.temp_dir)

        self.assertEqual(output, os.path.join(self.temp_dir, "input.vtk"))
        mock_read.assert_called_once_with(mesh_path)
        converted_mesh.save.assert_called_once_with(output)


if __name__ == "__main__":
    unittest.main()
