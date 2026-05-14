import os
import sys

import numpy as np


def _create_gripper_lineset(o3d, rotation, translation, width, height, depth):
    finger_thickness = max(0.006, min(0.018, height * 0.6))
    palm_depth = max(0.008, min(0.025, depth * 0.35))
    opening = max(0.002, width)
    finger_depth = max(0.015, depth)
    finger_height = max(0.012, height)
    palm_width = opening + 2.0 * finger_thickness

    boxes = [
        (
            np.asarray([0.0, 0.5 * opening + 0.5 * finger_thickness, 0.5 * finger_depth]),
            np.asarray([finger_height, finger_thickness, finger_depth]),
        ),
        (
            np.asarray([0.0, -0.5 * opening - 0.5 * finger_thickness, 0.5 * finger_depth]),
            np.asarray([finger_height, finger_thickness, finger_depth]),
        ),
        (
            np.asarray([0.0, 0.0, -0.5 * palm_depth]),
            np.asarray([finger_height, palm_width, palm_depth]),
        ),
    ]
    cube_lines = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    points = []
    lines = []
    colors = []
    for center, size in boxes:
        base = len(points)
        half = 0.5 * size
        corners = [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ]
        for corner in corners:
            points.append(rotation.dot(center + np.asarray(corner, dtype=np.float64)) + translation)
        for start, end in cube_lines:
            lines.append([base + start, base + end])
            colors.append([0.0, 0.85, 0.2])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


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
        gripper_width = float(np.asarray(data['gripper_width']).reshape(-1)[0])
        gripper_height = float(np.asarray(data['gripper_height']).reshape(-1)[0])
        gripper_depth = float(np.asarray(data['gripper_depth']).reshape(-1)[0])
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
            _create_gripper_lineset(
                o3d,
                rotation,
                origin,
                gripper_width,
                gripper_height,
                gripper_depth,
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
