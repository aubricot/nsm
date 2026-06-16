
import os
import pyvista as pv
import cv2
import numpy as np

# --- Settings ---
MESH_DIR = "vertebrae_meshes"   # folder with your meshes
FPS = 15
OUT_VIDEO = "inspect_meshes_front.mp4"
DURATION_PER_MESH = 1.0
WIDTH, HEIGHT = 640, 480
TEXT_COLOR = (0, 0, 0)
FONT_SCALE = 0.5
THICKNESS = 1
CAMERA_POSITION = [(1.5, 0, 0.2), (0, 0, 0), (0, 0, 1)] 

# --- Collect all VTK files ---
mesh_files = [f for f in os.listdir(MESH_DIR) if f.endswith((".vtk"))]
mesh_files.sort()

# --- Setup OpenCV video writer ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

# --- Offscreen PyVista Plotter ---
plotter = pv.Plotter(off_screen=True, window_size=(WIDTH, HEIGHT))
plotter.show(auto_close=False)
plotter.background_color = 'white'

# --- Render loop ---
for fname in mesh_files:
    mesh_path = os.path.join(MESH_DIR, fname)
    mesh = pv.read(mesh_path)
    mesh.translate(-np.array(mesh.center), inplace=True)
    mesh.compute_normals(inplace=True)

    n_frames = int(DURATION_PER_MESH * FPS)
    for _ in range(n_frames):
        plotter.clear()
        # add mesh with smooth shading
        plotter.add_mesh(mesh, color=(0.8,0.8,0.8), smooth_shading=True)
        # add light for proper shading
        plotter.add_light(pv.Light(position=(0, -2, 1), focal_point=(0,0,0),
                                   color='white', intensity=1.0))
        plotter.camera_position = CAMERA_POSITION
        plotter.render()

        # screenshot
        frame = plotter.screenshot(return_img=True)
        frame = np.ascontiguousarray(frame[:, :, :3])
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, fname, (10, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR,
                    THICKNESS, lineType=cv2.LINE_AA)
        out_video.write(frame_bgr)

    print(f"Rendered {fname}")


# --- Cleanup ---
out_video.release()
plotter.close()
print("Video saved as", OUT_VIDEO)
