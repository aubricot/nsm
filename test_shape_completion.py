import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import shutil
import numpy as np
import torch
import pyvista as pv

# Configure paths so imports succeed
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
NSM_PATH = os.path.join(PROJECT_ROOT, 'NSM')
if NSM_PATH not in sys.path:
    sys.path.insert(0, NSM_PATH)

from shape_completion_slicer import pv_to_tiny3d, main

class TestShapeCompletion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def _get_mock_config(self):
        return {
            "latent_size": 256,
            "samples_per_object_per_batch": 1000,
            "n_pts_per_object": 5000,
            "percent_near_surface": 0.5,
            "percent_further_from_surface": 0.5,
            "sigma_near": 0.01,
            "sigma_far": 0.1,
            "random_function": "uniform",
            "center_pts": True,
            "normalize_pts": True,
            "scale_method": "radius",
            "scale_jointly": True,
            "verbose": False,
            "cache": False,
            "equal_pos_neg": True,
            "fix_mesh": False
        }
        
    def test_pv_to_tiny3d(self):
        # Create a simple pyvista PolyData mesh (a single triangle)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mesh_pv = pv.PolyData(points, faces)
        
        # Convert
        mesh_o3d = pv_to_tiny3d(mesh_pv)
        
        # Verify
        import tiny3d as o3d
        self.assertIsInstance(mesh_o3d, o3d.geometry.TriangleMesh)
        self.assertEqual(len(mesh_o3d.vertices), 3)
        self.assertEqual(len(mesh_o3d.triangles), 1)

    @patch('NSM.helper_funcs.load_config')
    @patch('NSM.helper_funcs.load_model_and_latents')
    @patch('NSM.optimization.get_top_k_pcs')
    @patch('NSM.datasets.SDFSamples')
    @patch('NSM.optimization.optimize_latent_partial')
    @patch('NSM.mesh.create_mesh')
    @patch('NSM.optimization.get_norm_params')
    @patch('NSM.optimization.normalize_mesh')
    def test_main_pipeline_vtk(self, mock_normalize_mesh, mock_get_norm_params, mock_create_mesh,
                               mock_optimize_latent, mock_sdf_samples, mock_get_top_k_pcs,
                               mock_load_model_latents, mock_load_config):
        
        # 1. Setup mock configs & returns
        mock_load_config.return_value = self._get_mock_config()
        mock_model = MagicMock()
        mock_latent_codes = torch.zeros(5, 256)
        mock_load_model_latents.return_value = (mock_model, None, mock_latent_codes)
        mock_get_top_k_pcs.return_value = (None, torch.zeros(5, 256))
        
        # Mock SDF dataset instance
        mock_dataset_inst = MagicMock()
        mock_sample_dict = {'xyz': torch.zeros(1000, 3), 'gt_sdf': torch.zeros(1000, 1)}
        mock_dataset_inst.__getitem__.return_value = (mock_sample_dict, None)
        mock_sdf_samples.return_value = mock_dataset_inst
        
        # Mock optimizer
        mock_optimize_latent.return_value = (torch.zeros(1, 256), None)
        
        # Mock create_mesh return value (a pyvista mesh)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh = pv.PolyData(points, faces)
        mock_create_mesh.return_value = mock_mesh
        
        # Mock normalization parameters
        mock_get_norm_params.return_value = (np.array([0.0, 0.0, 0.0]), 1.0)
        mock_normalize_mesh.return_value = mock_mesh
        
        # 2. Run main with temporary file paths
        dummy_config = os.path.join(self.temp_dir, "config.json")
        dummy_model = os.path.join(self.temp_dir, "model.pth")
        dummy_latents = os.path.join(self.temp_dir, "latents.pth")
        dummy_mesh = os.path.join(self.temp_dir, "input.vtk")
        
        # Write empty dummy files so checks on existence pass
        for fpath in [dummy_config, dummy_model, dummy_latents, dummy_mesh]:
            with open(fpath, 'w') as f:
                f.write('')
                
        # Call main
        main(
            config_path=dummy_config,
            model_path=dummy_model,
            latent_codes_path=dummy_latents,
            input_mesh_path=dummy_mesh,
            output_folder_path=self.temp_dir
        )
        
        # Verify output files were created
        output_mesh_path = os.path.join(self.temp_dir, "input_shape_completion.vtk")
        done_file_path = os.path.join(self.temp_dir, "input.done")
        
        self.assertTrue(os.path.exists(output_mesh_path))
        self.assertTrue(os.path.exists(done_file_path))

    @patch('NSM.helper_funcs.load_config')
    @patch('NSM.helper_funcs.load_model_and_latents')
    @patch('NSM.optimization.get_top_k_pcs')
    @patch('NSM.datasets.SDFSamples')
    @patch('NSM.optimization.optimize_latent_partial')
    @patch('NSM.mesh.create_mesh')
    @patch('NSM.optimization.get_norm_params')
    @patch('NSM.optimization.normalize_mesh')
    @patch('shape_completion_slicer.convert_ply_to_vtk', create=True)
    def test_main_pipeline_ply(self, mock_convert_ply, mock_normalize_mesh, mock_get_norm_params, mock_create_mesh,
                               mock_optimize_latent, mock_sdf_samples, mock_get_top_k_pcs,
                               mock_load_model_latents, mock_load_config):
        """Test the workflow when the input mesh is a PLY file."""
        mock_load_config.return_value = self._get_mock_config()
        mock_model = MagicMock()
        mock_latent_codes = torch.zeros(5, 256)
        mock_load_model_latents.return_value = (mock_model, None, mock_latent_codes)
        mock_get_top_k_pcs.return_value = (None, torch.zeros(5, 256))
        
        # Mock SDF dataset instance
        mock_dataset_inst = MagicMock()
        mock_sample_dict = {'xyz': torch.zeros(1000, 3), 'gt_sdf': torch.zeros(1000, 1)}
        mock_dataset_inst.__getitem__.return_value = (mock_sample_dict, None)
        mock_sdf_samples.return_value = mock_dataset_inst
        
        # Mock optimizer
        mock_optimize_latent.return_value = (torch.zeros(1, 256), None)
        
        # Mock create_mesh return value (a pyvista mesh)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh = pv.PolyData(points, faces)
        mock_create_mesh.return_value = mock_mesh
        
        # Mock normalization parameters
        mock_get_norm_params.return_value = (np.array([0.0, 0.0, 0.0]), 1.0)
        mock_normalize_mesh.return_value = mock_mesh
        
        # Mock ply to vtk converter
        mock_convert_ply.return_value = (mock_mesh, os.path.join(self.temp_dir, "input.vtk"))
        
        # Setup temp files
        dummy_config = os.path.join(self.temp_dir, "config.json")
        dummy_model = os.path.join(self.temp_dir, "model.pth")
        dummy_latents = os.path.join(self.temp_dir, "latents.pth")
        dummy_mesh_ply = os.path.join(self.temp_dir, "input.ply")
        
        for fpath in [dummy_config, dummy_model, dummy_latents, dummy_mesh_ply]:
            with open(fpath, 'w') as f:
                f.write('')
                
        # Call main with the PLY file
        main(
            config_path=dummy_config,
            model_path=dummy_model,
            latent_codes_path=dummy_latents,
            input_mesh_path=dummy_mesh_ply,
            output_folder_path=self.temp_dir
        )
        
        # Verify PLY conversion was triggered
        mock_convert_ply.assert_called_once_with(dummy_mesh_ply, save=True)

    @patch('NSM.helper_funcs.load_config')
    @patch('NSM.helper_funcs.load_model_and_latents')
    @patch('NSM.optimization.get_top_k_pcs')
    @patch('NSM.datasets.SDFSamples')
    @patch('NSM.optimization.optimize_latent_partial')
    @patch('NSM.mesh.create_mesh')
    @patch('NSM.optimization.get_norm_params')
    @patch('NSM.optimization.normalize_mesh')
    @patch('os.chdir')
    def test_main_default_paths(self, mock_chdir, mock_normalize_mesh, mock_get_norm_params, mock_create_mesh,
                                mock_optimize_latent, mock_sdf_samples, mock_get_top_k_pcs,
                                mock_load_model_latents, mock_load_config):
        """Test default path construction and directory changing when model_path is None."""
        mock_load_config.return_value = self._get_mock_config()
        mock_model = MagicMock()
        mock_latent_codes = torch.zeros(5, 256)
        mock_load_model_latents.return_value = (mock_model, None, mock_latent_codes)
        mock_get_top_k_pcs.return_value = (None, torch.zeros(5, 256))
        
        mock_dataset_inst = MagicMock()
        mock_sample_dict = {'xyz': torch.zeros(1000, 3), 'gt_sdf': torch.zeros(1000, 1)}
        mock_dataset_inst.__getitem__.return_value = (mock_sample_dict, None)
        mock_sdf_samples.return_value = mock_dataset_inst
        mock_optimize_latent.return_value = (torch.zeros(1, 256), None)
        
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh = pv.PolyData(points, faces)
        mock_create_mesh.return_value = mock_mesh
        mock_get_norm_params.return_value = (np.array([0.0, 0.0, 0.0]), 1.0)
        mock_normalize_mesh.return_value = mock_mesh
        
        # Call main with model_path=None
        dummy_config = os.path.join(self.temp_dir, "config.json")
        dummy_mesh = os.path.join(self.temp_dir, "input.vtk")
        with open(dummy_config, 'w') as f: f.write('')
        with open(dummy_mesh, 'w') as f: f.write('')
        
        main(
            config_path=dummy_config,
            model_path=None,
            latent_codes_path="dummy_lc.pth",
            input_mesh_path=dummy_mesh,
            output_folder_path=self.temp_dir
        )
        
        # Verify directory change to "run_v57" was triggered
        mock_chdir.assert_called_once_with("run_v57")

    @patch('NSM.helper_funcs.load_config')
    @patch('NSM.helper_funcs.load_model_and_latents')
    @patch('NSM.optimization.get_top_k_pcs')
    @patch('NSM.datasets.SDFSamples')
    @patch('NSM.optimization.optimize_latent_partial')
    @patch('NSM.mesh.create_mesh')
    @patch('NSM.optimization.get_norm_params')
    @patch('NSM.optimization.normalize_mesh')
    @patch('os.listdir')
    @patch('random.sample')
    def test_main_random_mesh_selection(self, mock_random_sample, mock_listdir, mock_normalize_mesh, mock_get_norm_params,
                                        mock_create_mesh, mock_optimize_latent, mock_sdf_samples, mock_get_top_k_pcs,
                                        mock_load_model_latents, mock_load_config):
        """Test behavior when input_mesh_path is None (random selection from directory)."""
        mock_load_config.return_value = self._get_mock_config()
        mock_model = MagicMock()
        mock_latent_codes = torch.zeros(5, 256)
        mock_load_model_latents.return_value = (mock_model, None, mock_latent_codes)
        mock_get_top_k_pcs.return_value = (None, torch.zeros(5, 256))
        
        mock_dataset_inst = MagicMock()
        mock_sample_dict = {'xyz': torch.zeros(1000, 3), 'gt_sdf': torch.zeros(1000, 1)}
        mock_dataset_inst.__getitem__.return_value = (mock_sample_dict, None)
        mock_sdf_samples.return_value = mock_dataset_inst
        mock_optimize_latent.return_value = (torch.zeros(1, 256), None)
        
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh = pv.PolyData(points, faces)
        mock_create_mesh.return_value = mock_mesh
        mock_get_norm_params.return_value = (np.array([0.0, 0.0, 0.0]), 1.0)
        mock_normalize_mesh.return_value = mock_mesh
        
        # Mock mesh directory contents and selection
        mock_listdir.return_value = ["mesh1.vtk", "mesh2.vtk", "mesh3.vtk", "mesh4.vtk", "mesh5.vtk"]
        mock_random_sample.return_value = ["mesh1.vtk", "mesh2.vtk", "mesh3.vtk", "mesh4.vtk", "mesh5.vtk"]
        
        # Setup files
        dummy_config = os.path.join(self.temp_dir, "config.json")
        dummy_model = os.path.join(self.temp_dir, "model.pth")
        dummy_latents = os.path.join(self.temp_dir, "latents.pth")
        with open(dummy_config, 'w') as f: f.write('')
        with open(dummy_model, 'w') as f: f.write('')
        with open(dummy_latents, 'w') as f: f.write('')
        
        # Mock existence check for selected meshes
        original_exists = os.path.exists
        def mock_exists(path):
            if "fossils/models_smooth_hollow/aligned" in path:
                return True
            return original_exists(path)
            
        with patch('os.path.exists', mock_exists):
            main(
                config_path=dummy_config,
                model_path=dummy_model,
                latent_codes_path=dummy_latents,
                input_mesh_path=None,
                output_folder_path=self.temp_dir
            )
            
        # Verify random.sample was called to pick 5 meshes
        mock_random_sample.assert_called_once()
        self.assertEqual(mock_optimize_latent.call_count, 10) # 2 phases per mesh, for 5 meshes = 10 calls

if __name__ == '__main__':
    unittest.main()
