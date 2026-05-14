import os
import sys

import numpy as np


def _create_mesh_box(o3d, width, height, depth):
    box = o3d.geometry.TriangleMesh()
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [width, 0.0, 0.0],
        [0.0, 0.0, depth],
        [width, 0.0, depth],
        [0.0, height, 0.0],
        [width, height, 0.0],
        [0.0, height, depth],
        [width, height, depth],
    ], dtype=np.float64)
    triangles = np.asarray([
        [4, 7, 5], [4, 6, 7],
        [0, 2, 4], [2, 6, 4],
        [0, 1, 2], [1, 3, 2],
        [1, 5, 7], [1, 7, 3],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 1], [1, 4, 5],
    ], dtype=np.int32)
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def _create_gripper_mesh(o3d, rotation, translation, width, depth, score):
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    left = _create_mesh_box(o3d, depth + depth_base + finger_width, finger_width, height)
    right = _create_mesh_box(o3d, depth + depth_base + finger_width, finger_width, height)
    bottom = _create_mesh_box(o3d, finger_width, width, height)
    tail = _create_mesh_box(o3d, tail_length, finger_width, height)

    left_points = np.asarray(left.vertices)
    left_triangles = np.asarray(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2.0 + finger_width
    left_points[:, 2] -= height / 2.0

    right_points = np.asarray(right.vertices)
    right_triangles = np.asarray(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2.0
    right_points[:, 2] -= height / 2.0

    bottom_points = np.asarray(bottom.vertices)
    bottom_triangles = np.asarray(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2.0
    bottom_points[:, 2] -= height / 2.0

    tail_points = np.asarray(tail.vertices)
    tail_triangles = np.asarray(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2.0
    tail_points[:, 2] -= height / 2.0

    vertices = np.concatenate(
        [left_points, right_points, bottom_points, tail_points],
        axis=0,
    )
    vertices = rotation.dot(vertices.T).T + translation
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles],
        axis=0,
    )
    color = np.asarray([score, 0.0, 1.0 - score], dtype=np.float64)
    colors = np.tile(color.reshape(1, 3), (len(vertices), 1))

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    gripper.compute_vertex_normals()
    return gripper


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
        gripper_rotation = np.asarray(data['gripper_rotation'], dtype=np.float64).reshape(3, 3)
        gripper_width = float(np.asarray(data['gripper_width']).reshape(-1)[0])
        gripper_depth = float(np.asarray(data['gripper_depth']).reshape(-1)[0])
        gripper_score = float(np.asarray(data['gripper_score']).reshape(-1)[0])
        frame_size = float(np.asarray(data['frame_size']).reshape(-1)[0])
        title = str(np.asarray(data['title']).reshape(-1)[0])

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
            origin=origin.tolist(),
        )
        pose_frame.rotate(rotation, center=origin.tolist())
        geometries = [cloud, pose_frame]
        geometries.append(
            _create_gripper_mesh(
                o3d,
                gripper_rotation,
                origin,
                gripper_width,
                gripper_depth,
                gripper_score,
            )
        )

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
