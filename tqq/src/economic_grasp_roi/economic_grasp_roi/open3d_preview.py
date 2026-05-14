import os
import sys

import numpy as np


def _import_preview_grasp_group(economic_grasp_repo_dir: str):
    if not hasattr(np, 'float'):
        setattr(np, 'float', float)
    third_party_dir = os.path.dirname(os.path.expanduser(economic_grasp_repo_dir))
    candidates = [
        os.path.join(third_party_dir, 'franka-graspnet-master', 'graspnetAPI'),
        os.path.join(os.path.expanduser(economic_grasp_repo_dir), 'graspnetAPI'),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    from graspnetAPI import GraspGroup
    return GraspGroup


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    preview_path = sys.argv[1]
    try:
        import open3d as o3d
    except ImportError:
        return 3

    try:
        data = np.load(preview_path, allow_pickle=False)
        points = np.asarray(data['points'], dtype=np.float64)
        colors = np.asarray(data['colors'], dtype=np.float64)
        origin = np.asarray(data['origin'], dtype=np.float64).reshape(3)
        rotation = np.asarray(data['rotation'], dtype=np.float64).reshape(3, 3)
        grasp_array = np.asarray(data['grasp_array'], dtype=np.float64).reshape(1, 17)
        frame_size = float(np.asarray(data['frame_size']).reshape(-1)[0])
        title = str(np.asarray(data['title']).reshape(-1)[0])
        economic_grasp_repo_dir = str(np.asarray(data['economic_grasp_repo_dir']).reshape(-1)[0])

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
            origin=origin.tolist(),
        )
        pose_frame.rotate(rotation, center=origin.tolist())
        geometries = [cloud, pose_frame]

        try:
            grasp_group = _import_preview_grasp_group(economic_grasp_repo_dir)
            geometries.extend(grasp_group(grasp_array).to_open3d_geometry_list())
        except Exception:
            pass

        o3d.visualization.draw_geometries(
            geometries,
            window_name=title or 'EconomicGrasp ROI grasp preview',
            width=1280,
            height=720,
        )
    finally:
        try:
            os.remove(preview_path)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
