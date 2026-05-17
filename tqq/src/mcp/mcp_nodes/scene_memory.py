from .mcp_shared import *


class SceneMemoryMixin:
    def _tool_observe_scene(self, arguments: Dict) -> Tuple[bool, str, Dict]:
        if not self.vision_enabled:
            return False, 'observe_scene failed: vision is disabled', {}
        question = str(arguments.get('question') or '').strip()
        if not question:
            object_names = arguments.get('object_names', [])
            object_names_text = '、'.join(self._aliases_from_argument(object_names))
            if object_names_text:
                question = (
                    f'请优先框出当前画面中的这些目标：{object_names_text}。'
                    '同时也请框出画面中其他清晰可见、可能成为机械臂运动障碍物的独立物体。'
                    '每个目标都必须单独给框。'
                )
            else:
                question = (
                    '请框出当前画面中所有清晰可见、适合机械臂抓取、放置参照'
                    '或可能成为机械臂运动障碍物的独立物体。每个目标都必须单独给框。'
                )
        try:
            detections = self._detect_objects_with_vision_api(question, target_name='', max_results=0)
        except Exception as exc:
            return False, f'observe_scene failed: {exc}', {}
        if not detections:
            return False, 'observe_scene failed: 视觉 API 没有返回可记忆目标。', {}

        scene_memory = self._remember_api_detections(
            detections,
            'observe_scene',
            aliases=self._aliases_from_argument(arguments.get('object_names', [])),
        )
        output_path = self._save_and_show_api_detection_result(
            detections,
            answer={'objects': detections, 'scene_memory': scene_memory},
            mode='observe_scene',
            status='Qwen-Omni scene memory',
        )
        remembered = scene_memory.get('remembered', [])
        parts = []
        for item in remembered:
            names = item.get('names', [])
            base_xyz = item.get('base_xyz', [])
            label = '/'.join(names[:2]) if names else 'object'
            if len(base_xyz) == 3:
                parts.append(
                    f'{label} base=({base_xyz[0]:.3f},{base_xyz[1]:.3f},{base_xyz[2]:.3f})'
                )
            else:
                parts.append(label)
        return True, (
            f'observe_scene success: remembered {len(remembered)} object(s): '
            + ('; '.join(parts) if parts else 'none')
            + f'; saved={output_path or "disabled"}'
        ), {'detections': detections, 'scene_memory': scene_memory, 'saved': output_path}

    @staticmethod
    def _remembered_entries_from_observe_result(observe_result: Dict) -> List[Dict]:
        if not isinstance(observe_result, dict):
            return []
        scene_memory = observe_result.get('scene_memory', {})
        if not isinstance(scene_memory, dict):
            return []
        remembered = scene_memory.get('remembered', [])
        if not isinstance(remembered, list):
            return []
        return [copy.deepcopy(entry) for entry in remembered if isinstance(entry, dict)]

    def _scene_entries_for_llm(self, entries: List[Dict]) -> List[Dict]:
        summary = []
        for index, entry in enumerate(entries):
            bbox = entry.get('bbox_xyxy', [])
            base_xyz = entry.get('base_xyz', [])
            item = {
                'index': index,
                'id': str(entry.get('id') or ''),
                'names': [str(name) for name in entry.get('names', [])],
                'class_name': str(entry.get('class_name') or ''),
                'label_zh': str(entry.get('label_zh') or ''),
                'is_container_candidate': self._is_container_entry(entry),
            }
            if isinstance(bbox, list) and len(bbox) == 4:
                item['bbox_xyxy'] = [round(float(value), 1) for value in bbox]
            if isinstance(base_xyz, list) and len(base_xyz) == 3:
                item['base_xyz'] = [round(float(value), 4) for value in base_xyz]
            summary.append(item)
        return summary

    def _container_model_from_scene(self, container: Dict) -> Dict:
        center_xyz = [float(value) for value in container.get('base_xyz', [0.0, 0.0, 0.0])[:3]]
        height = self.container_height_m
        bottom_z = center_xyz[2] + self.container_collision_z_offset_m
        top_z = bottom_z + height
        return {
            'id_prefix': f'container_{self._scene_name_key(container.get("class_name") or "box")}',
            'base_frame': self.base_frame,
            'center_xyz': [center_xyz[0], center_xyz[1], bottom_z + height * 0.5],
            'bottom_center_xyz': [center_xyz[0], center_xyz[1], bottom_z],
            'bottom_z': bottom_z,
            'top_z': top_z,
            'length_x_m': self.container_length_x_m,
            'width_y_m': self.container_width_y_m,
            'height_m': height,
            'wall_thickness_m': min(
                self.container_wall_thickness_m,
                self.container_length_x_m * 0.45,
                self.container_width_y_m * 0.45,
            ),
            'bottom_thickness_m': min(self.container_bottom_thickness_m, height * 0.45),
        }

    def _is_container_entry(self, entry: Dict) -> bool:
        names = ' '.join(str(name).lower() for name in entry.get('names', []))
        class_name = str(entry.get('class_name') or '').lower()
        label_zh = str(entry.get('label_zh') or '')
        text = f'{names} {class_name} {label_zh}'
        return any(
            token in text
            for token in ('box', 'container', 'bin', 'basket', '箱', '盒', '筐', '篮')
        )

    def _is_fruit_entry(self, entry: Dict) -> bool:
        if self._is_container_entry(entry):
            return False
        names = ' '.join(str(name).lower() for name in entry.get('names', []))
        class_name = str(entry.get('class_name') or '').lower()
        label_zh = str(entry.get('label_zh') or '')
        text = f'{names} {class_name} {label_zh}'
        return any(
            token in text
            for token in (
                'apple', 'orange', 'banana', 'pear', 'peach', 'fruit',
                '苹果', '橘', '橙', '香蕉', '梨', '桃', '水果',
            )
        )

    @staticmethod
    def _entry_center_inside_container_xy(entry: Dict, model: Dict) -> bool:
        xyz = entry.get('base_xyz', [])
        if not isinstance(xyz, list) or len(xyz) < 2:
            return False
        center_x, center_y, _ = [float(value) for value in model['center_xyz']]
        length_x = float(model['length_x_m'])
        width_y = float(model['width_y_m'])
        wall = float(model['wall_thickness_m'])
        margin = 0.01
        half_x = max(0.0, length_x * 0.5 - wall + margin)
        half_y = max(0.0, width_y * 0.5 - wall + margin)
        return (
            abs(float(xyz[0]) - center_x) <= half_x
            and abs(float(xyz[1]) - center_y) <= half_y
        )

    def _is_ignored_scene_obstacle(self, entry: Dict) -> bool:
        names = ' '.join(str(name).lower() for name in entry.get('names', []))
        class_name = str(entry.get('class_name') or '').lower()
        label_zh = str(entry.get('label_zh') or '')
        text = f'{names} {class_name} {label_zh}'
        return any(token in text for token in ('table', 'desk', 'surface', '桌', '桌面', '台面'))

    def _scene_collision_model_for_entry(self, entry: Dict) -> Dict:
        if self._is_container_entry(entry):
            model = self._container_model_from_scene(entry)
            return {
                'kind': 'container',
                'marker_id': self._scene_marker_id(entry),
                'object_ids': [
                    collision.id
                    for collision in self._container_collision_objects(model)
                ],
                'container_model': model,
            }
        if self._is_fruit_entry(entry):
            return self._fruit_collision_model_from_entry(entry)
        if self._is_ignored_scene_obstacle(entry):
            return {}
        return self._generic_object_collision_model_from_entry(entry)

    def _fruit_collision_model_from_entry(self, entry: Dict) -> Dict:
        size_xyz = entry.get('base_size_xyz', [])
        if not isinstance(size_xyz, list) or len(size_xyz) != 3:
            size_xyz = [self.held_object_default_radius_m * 2.0] * 3
        dims = [
            max(0.001, float(value) + self.held_object_padding_m * 2.0)
            for value in size_xyz[:3]
        ]
        sorted_dims = sorted(dims)
        object_id = self._scene_world_object_id(entry)
        if sorted_dims[2] > self.fruit_collision_box_long_ratio * max(0.001, sorted_dims[1]):
            return {
                'kind': 'fruit',
                'shape': 'box',
                'object_id': object_id,
                'marker_id': self._scene_marker_id(entry),
                'dimensions': dims,
            }
        radius = max(dims) * 0.5
        radius = min(self.held_object_max_radius_m, max(self.held_object_min_radius_m, radius))
        return {
            'kind': 'fruit',
            'shape': 'sphere',
            'object_id': object_id,
            'marker_id': self._scene_marker_id(entry),
            'radius': radius,
        }

    def _generic_object_collision_model_from_entry(self, entry: Dict) -> Dict:
        size_xyz = entry.get('base_size_xyz', [])
        if not isinstance(size_xyz, list) or len(size_xyz) != 3:
            size_xyz = [self.object_collision_min_size_m] * 3
        dimensions = []
        for value in size_xyz[:3]:
            dimension = float(value) + self.object_collision_padding_m * 2.0
            dimension = min(
                self.object_collision_max_size_m,
                max(self.object_collision_min_size_m, dimension),
            )
            dimensions.append(dimension)
        return {
            'kind': 'object',
            'shape': 'box',
            'object_id': self._scene_world_object_id(entry),
            'marker_id': self._scene_marker_id(entry),
            'dimensions': dimensions,
        }

    def _scene_marker_id(self, entry: Dict) -> int:
        object_id = self._scene_world_object_id(entry)
        return int(zlib.crc32(object_id.encode('utf-8')) & 0x7fffffff)

    def _scene_world_object_id(self, entry: Dict) -> str:
        names = entry.get('names', [])
        primary_name = str(names[0]) if names else str(entry.get('class_name') or '')
        key = self._scene_name_key(primary_name)
        if not key:
            key = self._scene_name_key(str(entry.get('id') or uuid.uuid4().hex[:8]))
        key = key or uuid.uuid4().hex[:8]
        return f'scene_{key}'

    def _apply_scene_collision_models(self, entries: List[Dict]) -> Dict:
        if not self.scene_collision_auto_apply or not entries:
            return {'applied': False, 'message': 'scene collision auto apply disabled or empty'}
        should_clear_previous = (
            self.scene_collision_clear_previous
            and any(
                source in str(entry.get('source') or '')
                for entry in entries
                for source in ('observe_scene', 'list_api_objects', 'box_api_object')
            )
        )
        if should_clear_previous:
            stale_object_ids = self._scene_collision_object_ids_for_entries(entries)
            self._clear_scene_world_models(
                'clear previous scene models',
                extra_object_ids=stale_object_ids,
            )
            self._clear_scene_markers()
        collisions = []
        visual_entries = {}
        for entry in entries:
            if self._is_container_entry(entry):
                if self.scene_collision_show_containers:
                    collisions.extend(
                        self._container_collision_objects(self._container_model_from_scene(entry))
                    )
                    visual_entries[self._scene_world_object_id(entry)] = copy.deepcopy(entry)
                continue
            if self._is_fruit_entry(entry) and self.scene_collision_show_fruits:
                collision = self._fruit_world_collision_object(entry)
                if collision is not None:
                    collisions.append(collision)
                    visual_entries[self._scene_world_object_id(entry)] = copy.deepcopy(entry)
                continue
            if self.scene_collision_show_other_objects:
                if self._is_ignored_scene_obstacle(entry):
                    continue
                collision = self._generic_world_collision_object(entry)
                if collision is not None:
                    collisions.append(collision)
                    visual_entries[self._scene_world_object_id(entry)] = copy.deepcopy(entry)
        if self.scene_collision_show_static_right_arm:
            collisions.extend(self._static_right_arm_collision_objects())
        if not collisions:
            return {'applied': False, 'message': 'no supported scene collision models'}
        ok, message = self._apply_collision_objects(collisions, 'apply scene collision models')
        if ok:
            with self.scene_collision_lock:
                self.active_scene_collision_ids.update(collision.id for collision in collisions)
            with self.scene_marker_lock:
                if should_clear_previous:
                    self.active_scene_visual_entries = visual_entries
                else:
                    self.active_scene_visual_entries.update(visual_entries)
                marker_entries = list(self.active_scene_visual_entries.values())
            self._publish_scene_markers(marker_entries, include_attached=True)
        return {
            'applied': ok,
            'message': message,
            'object_ids': [collision.id for collision in collisions],
        }

    def _static_right_arm_collision_objects(self) -> List[CollisionObject]:
        pad = float(self.static_right_arm_collision_padding_m)
        link_poses = self._static_right_arm_link_poses()
        mesh_specs = [
            ('static_right_arm_link0', 'link0', self._franka_arm_mesh_resource('link0', visual=False)),
            ('static_right_arm_link1', 'link1', self._franka_arm_mesh_resource('link1', visual=False)),
            ('static_right_arm_link2', 'link2', self._franka_arm_mesh_resource('link2', visual=False)),
            ('static_right_arm_link3', 'link3', self._franka_arm_mesh_resource('link3', visual=False)),
            ('static_right_arm_link4', 'link4', self._franka_arm_mesh_resource('link4', visual=False)),
            ('static_right_arm_link5', 'link5', self._franka_arm_mesh_resource('link5', visual=False)),
            ('static_right_arm_link6', 'link6', self._franka_arm_mesh_resource('link6', visual=False)),
            ('static_right_arm_link7', 'link7', self._franka_arm_mesh_resource('link7', visual=False)),
            ('static_right_arm_hand', 'hand', self._franka_hand_mesh_resource('hand', visual=False)),
        ]
        collisions = []
        mesh_load_failed = False
        for object_id, link_name, mesh_resource in mesh_specs:
            link_pose = link_poses.get(link_name)
            if link_pose is None:
                continue
            collision = self._mesh_collision_object(
                object_id,
                mesh_resource,
                link_pose[0],
                link_pose[1],
                scale=1.0,
            )
            if collision is None:
                mesh_load_failed = True
                break
            collisions.append(collision)
        if collisions and not mesh_load_failed:
            return collisions

        def padded(dimensions: List[float]) -> List[float]:
            return [max(0.01, float(value) + pad * 2.0) for value in dimensions]

        specs = [
            ('static_right_arm_link0', 'link0', [0.28, 0.28, 0.30], [0.0, 0.0, 0.15]),
            ('static_right_arm_link1', 'link1', [0.24, 0.24, 0.42], [0.0, 0.0, -0.12]),
            ('static_right_arm_link2', 'link2', [0.24, 0.24, 0.32], [0.0, 0.0, 0.0]),
            ('static_right_arm_link3', 'link3', [0.24, 0.24, 0.34], [0.0, 0.0, -0.10]),
            ('static_right_arm_link4', 'link4', [0.24, 0.24, 0.32], [0.0, 0.0, 0.0]),
            ('static_right_arm_link5', 'link5', [0.24, 0.24, 0.42], [0.0, 0.0, -0.14]),
            ('static_right_arm_link6', 'link6', [0.20, 0.20, 0.26], [0.0, 0.0, -0.02]),
            ('static_right_arm_link7', 'link7', [0.18, 0.18, 0.24], [0.0, 0.0, 0.04]),
            ('static_right_arm_hand', 'hand', [0.18, 0.16, 0.24], [0.0, 0.0, 0.07]),
        ]
        collisions = []
        for object_id, link_name, dimensions, local_position in specs:
            link_pose = link_poses.get(link_name)
            if link_pose is None:
                continue
            position, orientation = self._offset_pose(
                link_pose[0],
                link_pose[1],
                local_position,
                (0.0, 0.0, 0.0, 1.0),
            )
            collisions.append(
                self._box_collision_object(
                    object_id,
                    padded(dimensions),
                    position,
                    orientation=orientation,
                )
            )
        return collisions

    def _static_right_arm_link_poses(
        self,
    ) -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        transforms = {}
        current = self._transform_from_xyz_rpy(self.static_right_arm_offset_xyz_m, (0.0, 0.0, 0.0))
        transforms['link0'] = current

        joint_origins = [
            ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
            ((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            ((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            ((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            ((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
        ]
        joint_angles = [
            math.radians(float(value))
            for value in self.static_right_arm_joint_positions_deg[:len(joint_origins)]
        ]
        for index, (origin_xyz, origin_rpy) in enumerate(joint_origins, start=1):
            current = self._compose_transform(current, self._transform_from_xyz_rpy(origin_xyz, origin_rpy))
            current = self._compose_transform(current, self._transform_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, joint_angles[index - 1])))
            transforms[f'link{index}'] = current

        link8 = self._compose_transform(
            current,
            self._transform_from_xyz_rpy((0.0, 0.0, 0.107), (0.0, 0.0, 0.0)),
        )
        hand = self._compose_transform(
            link8,
            self._transform_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, -math.pi / 4.0)),
        )
        transforms['link8'] = link8
        transforms['hand'] = hand
        transforms['leftfinger'] = self._compose_transform(
            hand,
            self._transform_from_xyz_rpy((0.0, 0.04, 0.0584), (0.0, 0.0, 0.0)),
        )
        transforms['rightfinger'] = self._compose_transform(
            hand,
            self._transform_from_xyz_rpy((0.0, -0.04, 0.0584), (0.0, 0.0, math.pi)),
        )
        return {
            name: self._position_quaternion_from_transform(transform)
            for name, transform in transforms.items()
        }

    @staticmethod
    def _identity_transform() -> List[List[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _transform_from_xyz_rpy(
        self,
        xyz: Sequence[float],
        rpy: Sequence[float],
    ) -> List[List[float]]:
        roll, pitch, yaw = [float(value) for value in rpy[:3]]
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        rotation = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        transform = self._identity_transform()
        for row in range(3):
            for col in range(3):
                transform[row][col] = rotation[row][col]
        transform[0][3] = float(xyz[0])
        transform[1][3] = float(xyz[1])
        transform[2][3] = float(xyz[2])
        return transform

    @staticmethod
    def _compose_transform(
        left: List[List[float]],
        right: List[List[float]],
    ) -> List[List[float]]:
        result = [[0.0 for _ in range(4)] for _ in range(4)]
        for row in range(4):
            for col in range(4):
                result[row][col] = sum(left[row][k] * right[k][col] for k in range(4))
        return result

    def _offset_pose(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        local_position: Sequence[float],
        local_orientation: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        rotated_offset = self._rotate_vector_by_quaternion(
            (float(local_position[0]), float(local_position[1]), float(local_position[2])),
            orientation,
        )
        next_position = (
            float(position[0]) + rotated_offset[0],
            float(position[1]) + rotated_offset[1],
            float(position[2]) + rotated_offset[2],
        )
        next_orientation = self._quaternion_multiply(orientation, local_orientation)
        return next_position, next_orientation

    @staticmethod
    def _position_quaternion_from_transform(
        transform: List[List[float]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        position = (float(transform[0][3]), float(transform[1][3]), float(transform[2][3]))
        quaternion = SceneMemoryMixin._quaternion_from_rotation_matrix(transform)
        return position, quaternion

    @staticmethod
    def _quaternion_from_rotation_matrix(
        matrix: List[List[float]],
    ) -> Tuple[float, float, float, float]:
        m00, m01, m02 = matrix[0][0], matrix[0][1], matrix[0][2]
        m10, m11, m12 = matrix[1][0], matrix[1][1], matrix[1][2]
        m20, m21, m22 = matrix[2][0], matrix[2][1], matrix[2][2]
        trace = m00 + m11 + m22
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (m21 - m12) / scale
            qy = (m02 - m20) / scale
            qz = (m10 - m01) / scale
        elif m00 > m11 and m00 > m22:
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / scale
            qx = 0.25 * scale
            qy = (m01 + m10) / scale
            qz = (m02 + m20) / scale
        elif m11 > m22:
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / scale
            qx = (m01 + m10) / scale
            qy = 0.25 * scale
            qz = (m12 + m21) / scale
        else:
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / scale
            qx = (m02 + m20) / scale
            qy = (m12 + m21) / scale
            qz = 0.25 * scale
        return SceneMemoryMixin._normalize_quaternion((qx, qy, qz, qw))

    @staticmethod
    def _normalize_quaternion(
        quaternion: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        qx, qy, qz, qw = quaternion
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return (qx / norm, qy / norm, qz / norm, qw / norm)

    @staticmethod
    def _quaternion_multiply(
        left: Tuple[float, float, float, float],
        right: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        ax, ay, az, aw = left
        bx, by, bz, bw = right
        return SceneMemoryMixin._normalize_quaternion((
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ))

    @staticmethod
    def _pose_from_position_quaternion(
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> Pose:
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
        return pose

    def _scene_collision_object_ids_for_entries(self, entries: List[Dict]) -> List[str]:
        object_ids = []
        for entry in entries:
            model = entry.get('collision_model') or self._scene_collision_model_for_entry(entry)
            if model.get('kind') == 'container':
                object_ids.extend(str(item) for item in model.get('object_ids', []) if item)
                continue
            object_id = str(model.get('object_id') or self._scene_world_object_id(entry))
            if object_id:
                object_ids.append(object_id)
        return sorted(set(object_ids))

    def _clear_scene_world_models(
        self,
        label: str,
        extra_object_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, str]:
        with self.scene_collision_lock:
            object_ids = set(self.active_scene_collision_ids)
            self.active_scene_collision_ids.clear()
        object_ids.update(str(item) for item in (extra_object_ids or []) if item)
        object_ids = sorted(object_ids)
        if not object_ids:
            return True, f'{label}: no previous world models'
        removals = [
            self._remove_collision_object(object_id, self.base_frame)
            for object_id in object_ids
        ]
        return self._apply_collision_objects(removals, label)

    def _fruit_world_collision_object(self, entry: Dict) -> Optional[CollisionObject]:
        model = entry.get('collision_model') or self._fruit_collision_model_from_entry(entry)
        if model.get('kind') != 'fruit':
            return None
        center = entry.get('base_xyz', [])
        if not isinstance(center, list) or len(center) != 3:
            return None
        primitive = self._primitive_from_collision_model(model)
        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
        pose.orientation.w = 1.0
        collision = CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = str(model.get('object_id') or self._scene_world_object_id(entry))
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = CollisionObject.ADD
        return collision

    def _generic_world_collision_object(self, entry: Dict) -> Optional[CollisionObject]:
        model = entry.get('collision_model') or self._generic_object_collision_model_from_entry(entry)
        if model.get('kind') != 'object':
            return None
        center = entry.get('base_xyz', [])
        if not isinstance(center, list) or len(center) != 3:
            return None
        return self._box_collision_object(
            str(model.get('object_id') or self._scene_world_object_id(entry)),
            [float(value) for value in model.get('dimensions', [self.object_collision_min_size_m] * 3)[:3]],
            [float(center[0]), float(center[1]), float(center[2])],
        )

    def _remove_world_collision_for_entry(self, entry: Dict, label: str) -> Tuple[bool, str]:
        model = entry.get('collision_model') or self._scene_collision_model_for_entry(entry)
        object_ids = []
        if model.get('kind') in ('fruit', 'object'):
            object_ids = [str(model.get('object_id') or self._scene_world_object_id(entry))]
        elif model.get('kind') == 'container':
            object_ids = [str(item) for item in model.get('object_ids', [])]
        if not object_ids:
            return True, f'{label}: no world collision object id'
        removals = [
            self._remove_collision_object(object_id, self.base_frame)
            for object_id in object_ids
        ]
        ok, message = self._apply_collision_objects(removals, label)
        if ok:
            with self.scene_collision_lock:
                for object_id in object_ids:
                    self.active_scene_collision_ids.discard(object_id)
        return ok, message

    def _remove_collision_object(self, object_id: str, frame_id: str) -> CollisionObject:
        collision = CollisionObject()
        collision.header.frame_id = str(frame_id or self.base_frame)
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = str(object_id)
        collision.operation = CollisionObject.REMOVE
        return collision

    def _attach_held_object(self, entry: Dict) -> Tuple[bool, str]:
        if not self.held_object_collision_enabled:
            attached_id = self._scene_world_object_id(entry)
            with self.held_object_lock:
                self.current_attached_object_id = attached_id
                self.held_object_visual = copy.deepcopy(entry)
            with self.scene_marker_lock:
                self.active_scene_visual_entries.pop(attached_id, None)
                marker_entries = list(self.active_scene_visual_entries.values())
            self._publish_scene_markers(marker_entries, include_attached=False)
            return True, f'held object collision disabled; tracked held object {attached_id} without MoveIt attach'

        model = entry.get('collision_model') or self._scene_collision_model_for_entry(entry)
        if model.get('kind') == 'container':
            return True, 'held object is a container; attach skipped'
        if model.get('kind') not in ('fruit', 'object'):
            return True, 'held object collision model is unavailable; attach skipped'
        attached_id = str(model.get('object_id') or self._scene_world_object_id(entry))
        attached = AttachedCollisionObject()
        attached.link_name = self.held_object_link_name or self.end_effector_frame
        attached.object.header.frame_id = attached.link_name
        attached.object.header.stamp = self.get_clock().now().to_msg()
        attached.object.id = attached_id
        attached.object.primitives = [self._primitive_from_collision_model(model)]
        pose = Pose()
        pose.position.x = float(self.held_object_offset_xyz_tcp[0])
        pose.position.y = float(self.held_object_offset_xyz_tcp[1])
        pose.position.z = float(self.held_object_offset_xyz_tcp[2])
        pose.orientation.w = 1.0
        attached.object.primitive_poses = [pose]
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = list(self.held_object_touch_links)
        ok, message = self._apply_collision_objects(
            [],
            f'attach held object {attached_id}',
            attached=[attached],
        )
        if ok:
            with self.held_object_lock:
                self.current_attached_object_id = attached_id
                self.held_object_visual = copy.deepcopy(entry)
            with self.scene_marker_lock:
                self.active_scene_visual_entries.pop(attached_id, None)
                marker_entries = list(self.active_scene_visual_entries.values())
            self._publish_scene_markers(marker_entries, include_attached=True)
        return ok, message

    def _detach_current_held_object(self) -> Tuple[bool, str]:
        with self.held_object_lock:
            attached_id = self.current_attached_object_id
        if not attached_id:
            return True, 'no attached held object to detach'
        if not self.held_object_collision_enabled:
            with self.held_object_lock:
                if self.current_attached_object_id == attached_id:
                    self.current_attached_object_id = ''
                    self.held_object_visual = None
            with self.scene_marker_lock:
                marker_entries = list(self.active_scene_visual_entries.values())
            self._publish_scene_markers(marker_entries, include_attached=False)
            return True, (
                f'held object collision disabled; cleared tracked held object {attached_id} '
                'without MoveIt detach'
            )
        attached = AttachedCollisionObject()
        attached.link_name = self.held_object_link_name or self.end_effector_frame
        attached.object.header.frame_id = attached.link_name
        attached.object.header.stamp = self.get_clock().now().to_msg()
        attached.object.id = attached_id
        attached.object.operation = CollisionObject.REMOVE
        removal = self._remove_collision_object(attached_id, self.base_frame)
        ok, message = self._apply_collision_objects(
            [removal],
            f'detach held object {attached_id}',
            attached=[attached],
        )
        if ok:
            with self.held_object_lock:
                if self.current_attached_object_id == attached_id:
                    self.current_attached_object_id = ''
                    self.held_object_visual = None
            with self.scene_marker_lock:
                marker_entries = list(self.active_scene_visual_entries.values())
            self._publish_scene_markers(marker_entries, include_attached=False)
        return ok, message

    def _publish_scene_markers(
        self,
        entries: List[Dict],
        include_attached: bool = True,
    ) -> None:
        if not self.scene_markers_enabled:
            return
        marker_array = MarkerArray()
        next_ids = set()
        for entry in entries:
            marker = self._marker_for_scene_entry(entry)
            if marker is None:
                continue
            marker_array.markers.append(marker)
            next_ids.add((marker.ns, marker.id))

        attached_entry = None
        if include_attached and self.held_object_collision_enabled:
            with self.held_object_lock:
                attached_entry = copy.deepcopy(self.held_object_visual)
        if attached_entry:
            attached_marker = self._marker_for_scene_entry(
                attached_entry,
                frame_id=self.held_object_link_name or self.end_effector_frame,
                position=self.held_object_offset_xyz_tcp,
                namespace='mcp_attached_objects',
            )
            if attached_marker is not None:
                marker_array.markers.append(attached_marker)
                next_ids.add((attached_marker.ns, attached_marker.id))

        if self.scene_collision_show_static_right_arm:
            for marker in self._static_right_arm_markers():
                marker_array.markers.append(marker)
                next_ids.add((marker.ns, marker.id))

        with self.scene_marker_lock:
            stale = self.active_scene_marker_ids - next_ids
            self.active_scene_marker_ids = next_ids
        for ns, marker_id in stale:
            delete_marker = Marker()
            delete_marker.header.frame_id = self.base_frame
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = ns
            delete_marker.id = int(marker_id)
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        if marker_array.markers:
            self.scene_markers_pub.publish(marker_array)

    def _clear_scene_markers(self) -> None:
        if not self.scene_markers_enabled:
            return
        with self.scene_marker_lock:
            marker_ids = list(self.active_scene_marker_ids)
            self.active_scene_marker_ids.clear()
        if not marker_ids:
            return
        with self.scene_marker_lock:
            self.active_scene_visual_entries.clear()
        marker_array = MarkerArray()
        for ns, marker_id in marker_ids:
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = int(marker_id)
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.scene_markers_pub.publish(marker_array)

    def _static_right_arm_markers(self) -> List[Marker]:
        markers = []
        link_poses = self._static_right_arm_link_poses()
        visual_specs = [
            ('link0', self._franka_arm_mesh_resource('link0', visual=True)),
            ('link1', self._franka_arm_mesh_resource('link1', visual=True)),
            ('link2', self._franka_arm_mesh_resource('link2', visual=True)),
            ('link3', self._franka_arm_mesh_resource('link3', visual=True)),
            ('link4', self._franka_arm_mesh_resource('link4', visual=True)),
            ('link5', self._franka_arm_mesh_resource('link5', visual=True)),
            ('link6', self._franka_arm_mesh_resource('link6', visual=True)),
            ('link7', self._franka_arm_mesh_resource('link7', visual=True)),
            ('hand', self._franka_hand_mesh_resource('hand', visual=True)),
            ('leftfinger', self._franka_hand_mesh_resource('finger', visual=True)),
            ('rightfinger', self._franka_hand_mesh_resource('finger', visual=True)),
        ]
        for index, (link_name, mesh_resource) in enumerate(visual_specs):
            link_pose = link_poses.get(link_name)
            if link_pose is None:
                continue
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'mcp_static_right_arm'
            marker.id = int(self.static_right_arm_marker_base_id + index)
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.mesh_resource = mesh_resource
            marker.mesh_use_embedded_materials = True
            marker.pose = self._pose_from_position_quaternion(link_pose[0], link_pose[1])
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = 0.85
            marker.color.g = 0.85
            marker.color.b = 0.85
            marker.color.a = 1.0
            markers.append(marker)
        return markers

    @staticmethod
    def _franka_arm_mesh_resource(link_name: str, visual: bool) -> str:
        subdir = 'visual' if visual else 'collision'
        suffix = 'dae' if visual else 'stl'
        return f'package://franka_description/meshes/robot_arms/fr3/{subdir}/{link_name}.{suffix}'

    @staticmethod
    def _franka_hand_mesh_resource(name: str, visual: bool) -> str:
        subdir = 'visual' if visual else 'collision'
        suffix = 'dae' if visual else 'stl'
        return f'package://franka_description/meshes/robot_ee/franka_hand_white/{subdir}/{name}.{suffix}'

    def _mesh_collision_object(
        self,
        object_id: str,
        mesh_resource: str,
        position: Sequence[float],
        orientation: Sequence[float],
        scale: float = 1.0,
    ) -> Optional[CollisionObject]:
        mesh = self._load_stl_mesh(mesh_resource, scale=scale)
        if mesh is None:
            return None
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])

        collision = CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = str(object_id)
        collision.meshes = [mesh]
        collision.mesh_poses = [pose]
        collision.operation = CollisionObject.ADD
        return collision

    def _load_stl_mesh(self, mesh_resource: str, scale: float = 1.0) -> Optional[Mesh]:
        mesh_path = self._resolve_package_resource(mesh_resource)
        if mesh_path is None:
            self.get_logger().warn(f'Could not resolve mesh resource: {mesh_resource}')
            return None
        cache_key = f'{mesh_path}:{float(scale):.6f}'
        cached = self._mesh_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        try:
            mesh = self._read_binary_stl_mesh(mesh_path, scale=scale)
        except Exception as exc:
            self.get_logger().warn(f'Could not load STL mesh {mesh_resource}: {exc}')
            return None
        self._mesh_cache[cache_key] = copy.deepcopy(mesh)
        return mesh

    def _resolve_package_resource(self, resource: str) -> Optional[Path]:
        text = str(resource or '').strip()
        if not text:
            return None
        if text.startswith('package://'):
            package_and_path = text[len('package://'):]
            package_name, sep, relative_path = package_and_path.partition('/')
            if not sep or not package_name or not relative_path:
                return None
            share_dir = None
            if get_package_share_directory is not None:
                try:
                    share_dir = get_package_share_directory(package_name)
                except Exception:
                    share_dir = None
            candidates = []
            if share_dir:
                candidates.append(Path(share_dir) / relative_path)
            candidates.extend([
                Path('/home/tqq/TQQ_ws/franka/install') / package_name / 'share' / package_name / relative_path,
                Path('/home/tqq/TQQ_ws/franka/src') / package_name / relative_path,
                Path('/opt/ros/humble/share') / package_name / relative_path,
            ])
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None
        path = Path(text)
        return path if path.exists() else None

    @staticmethod
    def _read_binary_stl_mesh(path: Path, scale: float = 1.0) -> Mesh:
        data = path.read_bytes()
        if len(data) < 84:
            raise ValueError('STL file too small')
        triangle_count = struct.unpack_from('<I', data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if len(data) < expected_size:
            raise ValueError(
                f'incomplete binary STL: expected at least {expected_size} bytes, got {len(data)}'
            )
        mesh = Mesh()
        vertex_index: Dict[Tuple[int, int, int], int] = {}
        scale = float(scale)

        def add_vertex(values: Tuple[float, float, float]) -> int:
            point = tuple(float(value) * scale for value in values)
            key = tuple(int(round(value * 1_000_000.0)) for value in point)
            existing = vertex_index.get(key)
            if existing is not None:
                return existing
            vertex_index[key] = len(mesh.vertices)
            vertex = Point()
            vertex.x = point[0]
            vertex.y = point[1]
            vertex.z = point[2]
            mesh.vertices.append(vertex)
            return vertex_index[key]

        offset = 84
        for _ in range(triangle_count):
            values = struct.unpack_from('<12f', data, offset)
            offset += 50
            triangle = MeshTriangle()
            triangle.vertex_indices = [
                add_vertex((values[3], values[4], values[5])),
                add_vertex((values[6], values[7], values[8])),
                add_vertex((values[9], values[10], values[11])),
            ]
            mesh.triangles.append(triangle)
        return mesh

    def _marker_for_scene_entry(
        self,
        entry: Dict,
        frame_id: Optional[str] = None,
        position: Optional[List[float]] = None,
        namespace: Optional[str] = None,
    ) -> Optional[Marker]:
        model = entry.get('collision_model') or self._scene_collision_model_for_entry(entry)
        if model.get('kind') not in ('fruit', 'container', 'object'):
            return None
        marker_id = int(model.get('marker_id') or self._scene_marker_id(entry))
        marker = Marker()
        marker_frame = str(frame_id or self.base_frame)
        marker.header.frame_id = marker_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        if namespace:
            marker.ns = namespace
        elif model.get('kind') == 'container':
            marker.ns = 'mcp_containers'
        elif model.get('kind') == 'fruit':
            marker.ns = 'mcp_fruits'
        else:
            marker.ns = 'mcp_obstacles'
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.frame_locked = marker_frame != self.base_frame
        marker.pose.orientation.w = 1.0
        if position is not None and len(position) >= 3:
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
        else:
            center = entry.get('base_xyz', [])
            if not isinstance(center, list) or len(center) != 3:
                return None
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

        if model.get('kind') == 'container':
            container_model = model.get('container_model') or self._container_model_from_scene(entry)
            marker.type = Marker.CUBE
            marker.pose.position.x = float(container_model['center_xyz'][0])
            marker.pose.position.y = float(container_model['center_xyz'][1])
            marker.pose.position.z = float(container_model['center_xyz'][2])
            marker.scale.x = float(container_model['length_x_m'])
            marker.scale.y = float(container_model['width_y_m'])
            marker.scale.z = float(container_model['height_m'])
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.1
            marker.color.a = 0.25
            return marker

        if model.get('kind') == 'object':
            marker.type = Marker.CUBE
            dims = model.get('dimensions', [self.object_collision_min_size_m] * 3)
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color.r = 0.1
            marker.color.g = 0.45
            marker.color.b = 1.0
            marker.color.a = 0.55
            return marker

        if model.get('shape') == 'box':
            marker.type = Marker.CUBE
            dims = model.get('dimensions', [self.held_object_default_radius_m * 2.0] * 3)
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
        else:
            marker.type = Marker.SPHERE
            diameter = float(model.get('radius') or self.held_object_default_radius_m) * 2.0
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
        marker.color.r = 1.0
        marker.color.g = 0.45
        marker.color.b = 0.05
        marker.color.a = 0.8
        return marker

    def _primitive_from_collision_model(self, model: Dict) -> SolidPrimitive:
        primitive = SolidPrimitive()
        if model.get('shape') == 'box':
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [
                float(value)
                for value in model.get(
                    'dimensions',
                    [self.held_object_default_radius_m * 2.0] * 3,
                )[:3]
            ]
            return primitive
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [
            float(model.get('radius') or self.held_object_default_radius_m)
        ]
        return primitive

    def _apply_container_collision_model(self, model: Dict) -> Tuple[bool, str]:
        collisions = self._container_collision_objects(model)
        ok, message = self._apply_collision_objects(collisions, 'apply container planning scene')
        if not ok:
            return False, message
        message = (
            f'container collision model applied: {len(collisions)} walls, '
            f'size={model["length_x_m"]:.3f}x{model["width_y_m"]:.3f}x{model["height_m"]:.3f}m'
        )
        self._publish_status(message)
        return True, message

    def _apply_collision_objects(
        self,
        collisions: List[CollisionObject],
        label: str,
        attached: Optional[List[AttachedCollisionObject]] = None,
    ) -> Tuple[bool, str]:
        if not collisions and not attached:
            return True, f'{label}: no collision objects to apply'
        if not self.apply_planning_scene_client.wait_for_service(
            timeout_sec=self.service_wait_timeout_sec,
        ):
            return False, (
                f'ApplyPlanningScene service not available: {self.apply_planning_scene_service}'
            )
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collisions
        scene.object_colors = self._object_colors_for_collision_objects(collisions)
        if attached:
            scene.robot_state.attached_collision_objects = attached
            scene.robot_state.is_diff = True
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_planning_scene_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.container_planning_scene_wait_timeout_sec,
            label,
        )
        if response is None:
            return False, f'{label}: timed out applying planning scene'
        if not bool(response.success):
            return False, f'{label}: MoveIt rejected planning scene update'
        message = (
            f'{label}: applied {len(collisions)} world object(s) and '
            f'{len(attached or [])} attached object(s)'
        )
        self._publish_status(message)
        return True, message

    def _container_collision_objects(self, model: Dict) -> List[CollisionObject]:
        prefix = str(model.get('id_prefix') or 'container')
        length_x = float(model['length_x_m'])
        width_y = float(model['width_y_m'])
        height = float(model['height_m'])
        wall = float(model['wall_thickness_m'])
        bottom_thickness = float(model['bottom_thickness_m'])
        center_x, center_y, center_z = [float(value) for value in model['center_xyz']]
        bottom_z = float(model['bottom_z'])

        specs = [
            (
                f'{prefix}_left_wall',
                [wall, width_y, height],
                [center_x - length_x * 0.5 + wall * 0.5, center_y, center_z],
            ),
            (
                f'{prefix}_right_wall',
                [wall, width_y, height],
                [center_x + length_x * 0.5 - wall * 0.5, center_y, center_z],
            ),
            (
                f'{prefix}_front_wall',
                [length_x, wall, height],
                [center_x, center_y + width_y * 0.5 - wall * 0.5, center_z],
            ),
            (
                f'{prefix}_back_wall',
                [length_x, wall, height],
                [center_x, center_y - width_y * 0.5 + wall * 0.5, center_z],
            ),
            (
                f'{prefix}_bottom',
                [length_x, width_y, bottom_thickness],
                [center_x, center_y, bottom_z + bottom_thickness * 0.5],
            ),
        ]
        return [
            self._box_collision_object(object_id, dimensions, position)
            for object_id, dimensions, position in specs
        ]

    def _box_collision_object(
        self,
        object_id: str,
        dimensions: List[float],
        position: List[float],
        orientation: Optional[Tuple[float, float, float, float]] = None,
    ) -> CollisionObject:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(value) for value in dimensions]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        if orientation is None:
            pose.orientation.w = 1.0
        else:
            pose.orientation.x = float(orientation[0])
            pose.orientation.y = float(orientation[1])
            pose.orientation.z = float(orientation[2])
            pose.orientation.w = float(orientation[3])

        collision = CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = str(object_id)
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = CollisionObject.ADD
        return collision

    @staticmethod
    def _object_colors_for_collision_objects(collisions: List[CollisionObject]) -> List[ObjectColor]:
        colors = []
        for collision in collisions:
            if not str(collision.id).startswith('static_right_arm_'):
                continue
            color = ObjectColor()
            color.id = str(collision.id)
            color.color.r = 0.78
            color.color.g = 0.78
            color.color.b = 0.78
            color.color.a = 0.35
            colors.append(color)
        return colors

    @staticmethod
    def _aliases_from_argument(value) -> List[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            text = value.strip()
            for separator in ('，', '、', ';', '；', '|'):
                text = text.replace(separator, ',')
            raw_items = text.split(',')
        else:
            raw_items = []
        aliases = []
        for item in raw_items:
            alias = normalize_object_name(str(item))
            if alias and alias not in aliases:
                aliases.append(alias)
        return aliases

    @staticmethod
    def _scene_name_key(name: str) -> str:
        text = normalize_object_name(name).lower()
        return ''.join(text.split())

    @staticmethod
    def _container_name_keys(name: str) -> List[str]:
        key = SceneMemoryMixin._scene_name_key(name)
        aliases = {
            'box': ['box', 'container', 'bin', 'basket', '箱', '箱子', '盒', '盒子', '筐', '篮', '篮子'],
            'container': ['box', 'container', 'bin', 'basket', '箱', '箱子', '盒', '盒子', '筐', '篮', '篮子'],
            'basket': ['basket', 'bin', 'container', 'box', '篮', '篮子', '筐', '箱', '箱子'],
            'bin': ['bin', 'basket', 'container', 'box', '筐', '篮', '篮子', '箱', '箱子'],
            '箱': ['箱', '箱子', '盒', '盒子', '筐', '篮', '篮子', 'box', 'container', 'basket', 'bin'],
            '箱子': ['箱', '箱子', '盒', '盒子', '筐', '篮', '篮子', 'box', 'container', 'basket', 'bin'],
            '盒': ['盒', '盒子', '箱', '箱子', '筐', '篮', '篮子', 'box', 'container'],
            '盒子': ['盒', '盒子', '箱', '箱子', '筐', '篮', '篮子', 'box', 'container'],
            '筐': ['筐', '篮', '篮子', '箱', '箱子', 'basket', 'bin', 'box', 'container'],
            '篮': ['篮', '篮子', '筐', '箱', '箱子', 'basket', 'bin', 'box', 'container'],
            '篮子': ['篮', '篮子', '筐', '箱', '箱子', 'basket', 'bin', 'box', 'container'],
        }
        keys = aliases.get(key, [key])
        result = []
        for item in keys:
            item_key = SceneMemoryMixin._scene_name_key(item)
            if item_key and item_key not in result:
                result.append(item_key)
        return result

    def _remember_api_detections(
        self,
        detections: List[Dict],
        source: str,
        aliases: Optional[List[str]] = None,
    ) -> Dict:
        aliases = aliases or []
        remembered = []
        failures = []
        for index, detection in enumerate(detections):
            if len(detections) == 1:
                detection_aliases = aliases
            else:
                detection_aliases = self._matched_aliases_for_detection(detection, aliases)
            entry, error = self._scene_entry_from_detection(
                detection,
                source,
                index,
                detection_aliases,
            )
            if entry is None:
                failures.append({
                    'class_name': str(detection.get('class_name') or ''),
                    'label_zh': str(detection.get('label_zh') or ''),
                    'error': error,
                })
                continue
            entry['collision_model'] = self._scene_collision_model_for_entry(entry)
            self._store_scene_entry(entry)
            remembered.append(copy.deepcopy(entry))

        collision_result = self._apply_scene_collision_models(remembered)
        if (
            collision_result.get('applied')
            and self.scene_collision_preview_sec > 0.0
        ):
            self._publish_status(
                f'scene collision models visible in RViz; preview '
                f'{self.scene_collision_preview_sec:.2f}s before action'
            )
            time.sleep(self.scene_collision_preview_sec)

        snapshot = self._scene_memory_snapshot()
        if remembered:
            self._publish_status(
                f'scene_memory updated from {source}: '
                + ', '.join('/'.join(item.get('names', [])[:2]) for item in remembered)
            )
        elif failures:
            self.get_logger().warn(
                f'scene_memory update from {source} stored no 3D objects: {failures}'
            )
        return {
            'remembered': remembered,
            'failures': failures,
            'collision_scene': collision_result,
            'objects': snapshot,
            'count': len(snapshot),
        }

    def _matched_aliases_for_detection(self, detection: Dict, aliases: List[str]) -> List[str]:
        if not aliases:
            return []
        class_name = self._scene_name_key(str(detection.get('class_name') or ''))
        label_zh = self._scene_name_key(str(detection.get('label_zh') or ''))
        keys = [key for key in (class_name, label_zh) if key]
        matched = []
        for alias in aliases:
            alias_key = self._scene_name_key(alias)
            if not alias_key:
                continue
            if any(alias_key == key or alias_key in key or key in alias_key for key in keys):
                matched.append(alias)
        return matched

    def _scene_entry_from_memory_result(self, scene_memory: Dict, name: str) -> Optional[Dict]:
        remembered = scene_memory.get('remembered', []) if isinstance(scene_memory, dict) else []
        key = self._scene_name_key(name)
        for entry in remembered:
            names = [self._scene_name_key(item) for item in entry.get('names', [])]
            if key and key in names:
                return copy.deepcopy(entry)
        if remembered:
            return copy.deepcopy(remembered[0])
        return self._lookup_scene_object(name)

    def _scene_entry_from_detection(
        self,
        detection: Dict,
        source: str,
        index: int,
        aliases: List[str],
    ) -> Tuple[Optional[Dict], str]:
        bbox = detection.get('bbox_xyxy', [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None, 'missing bbox_xyxy'
        point, details = self._base_point_from_bbox(bbox)
        if point is None:
            return None, details.get('error', 'failed to estimate 3D point from bbox')

        class_name = self._normalize_object_name(str(detection.get('class_name') or 'object')).lower()
        label_zh = self._normalize_object_name(str(detection.get('label_zh') or ''))
        names = []
        for candidate in [class_name, label_zh] + aliases:
            name = self._normalize_object_name(candidate)
            if name and name not in names:
                names.append(name)
        if not names:
            names.append(f'object_{index}')

        now_ros = self.get_clock().now().to_msg()
        return {
            'id': f'{self._scene_name_key(names[0])}_{uuid.uuid4().hex[:8]}',
            'names': names,
            'class_name': class_name,
            'label_zh': label_zh,
            'confidence': float(detection.get('confidence', 0.0)),
            'bbox_xyxy': [float(value) for value in bbox],
            'center_xy': self._float_pair(detection.get('center_xy', [])),
            'base_xyz': [float(point[0]), float(point[1]), float(point[2])],
            'camera_xyz': details.get('camera_xyz', []),
            'base_bounds': details.get('base_bounds', {}),
            'base_size_xyz': details.get('base_size_xyz', []),
            'collision_model': {},
            'base_frame': self.base_frame,
            'camera_frame': details.get('camera_frame', ''),
            'depth_m': float(details.get('depth_m', 0.0)),
            'depth_sample_count': int(details.get('depth_sample_count', 0)),
            'source': source,
            'stamp': time.time(),
            'ros_stamp': {'sec': int(now_ros.sec), 'nanosec': int(now_ros.nanosec)},
        }, ''

    @staticmethod
    def _float_pair(value) -> List[float]:
        if not isinstance(value, list) or len(value) < 2:
            return []
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return []

    def _store_scene_entry(self, entry: Dict) -> None:
        self._prune_scene_memory()
        with self.scene_memory_lock:
            for name in entry.get('names', []):
                key = self._scene_name_key(name)
                if key:
                    self.scene_memory[key] = copy.deepcopy(entry)

    def _scene_memory_snapshot(self) -> List[Dict]:
        self._prune_scene_memory()
        unique = {}
        with self.scene_memory_lock:
            for entry in self.scene_memory.values():
                entry_id = str(entry.get('id') or '')
                if entry_id:
                    unique[entry_id] = copy.deepcopy(entry)
        return sorted(unique.values(), key=lambda item: float(item.get('stamp', 0.0)), reverse=True)

    def _prune_scene_memory(self) -> None:
        if self.scene_memory_max_age_sec <= 0.0:
            return
        cutoff = time.time() - self.scene_memory_max_age_sec
        with self.scene_memory_lock:
            stale = [
                key for key, entry in self.scene_memory.items()
                if float(entry.get('stamp', 0.0)) < cutoff
            ]
            for key in stale:
                self.scene_memory.pop(key, None)

    def _lookup_scene_object(self, name: str) -> Optional[Dict]:
        self._prune_scene_memory()
        key = self._scene_name_key(name)
        if not key:
            return None
        with self.scene_memory_lock:
            entry = self.scene_memory.get(key)
            if entry is not None:
                return copy.deepcopy(entry)
            for candidate in self.scene_memory.values():
                names = [self._scene_name_key(item) for item in candidate.get('names', [])]
                if key in names:
                    return copy.deepcopy(candidate)
        return None

    def _lookup_container_scene_object(self, name: str) -> Optional[Dict]:
        self._prune_scene_memory()
        keys = self._container_name_keys(name)
        if not keys:
            return None
        with self.scene_memory_lock:
            for key in keys:
                entry = self.scene_memory.get(key)
                if entry is not None and self._is_container_entry(entry):
                    return copy.deepcopy(entry)
            for candidate in self.scene_memory.values():
                if not self._is_container_entry(candidate):
                    continue
                names = [self._scene_name_key(item) for item in candidate.get('names', [])]
                names.extend([
                    self._scene_name_key(str(candidate.get('class_name') or '')),
                    self._scene_name_key(str(candidate.get('label_zh') or '')),
                ])
                if any(key in names for key in keys):
                    return copy.deepcopy(candidate)
            for candidate in self.scene_memory.values():
                if self._is_container_entry(candidate):
                    return copy.deepcopy(candidate)
        return None

    def _latest_depth_and_camera_info(self) -> Tuple[RosImage, CameraInfo]:
        with self._depth_image_lock:
            depth_msg = self._latest_depth_image
            depth_time = self._latest_depth_image_time
        with self._camera_info_lock:
            camera_info = self._latest_camera_info
            camera_info_time = self._latest_camera_info_time
        if depth_msg is None:
            raise RuntimeError(f'No aligned depth image received on {self.vision_depth_topic}.')
        if camera_info is None:
            raise RuntimeError(f'No camera info received on {self.vision_camera_info_topic}.')
        depth_age = time.monotonic() - depth_time
        info_age = time.monotonic() - camera_info_time
        if self.vision_image_max_age_sec > 0.0 and depth_age > self.vision_image_max_age_sec:
            raise RuntimeError(
                f'Latest depth image is stale: {depth_age:.1f}s old on {self.vision_depth_topic}.'
            )
        if self.vision_image_max_age_sec > 0.0 and info_age > self.vision_image_max_age_sec:
            raise RuntimeError(
                f'Latest camera_info is stale: {info_age:.1f}s old on {self.vision_camera_info_topic}.'
            )
        return depth_msg, camera_info

    def _base_point_from_bbox(self, bbox: List[float]) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
        try:
            import numpy as np
        except ImportError:
            return None, {'error': 'python3-numpy is not installed'}

        try:
            depth_msg, camera_info = self._latest_depth_and_camera_info()
            depth = self._ros_depth_image_to_meters(depth_msg, np)
        except Exception as exc:
            return None, {'error': str(exc)}

        height, width = depth.shape[:2]
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None, {'error': f'invalid bbox values: {bbox}'}
        left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
        right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
        top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
        bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
        erode = self.scene_memory_erode_px
        if right - left > erode * 2 + 2:
            left += erode
            right -= erode
        if bottom - top > erode * 2 + 2:
            top += erode
            bottom -= erode
        if right <= left or bottom <= top:
            return None, {'error': f'bbox is empty after clipping: {bbox}'}

        roi = depth[top:bottom + 1, left:right + 1]
        valid = roi[np.isfinite(roi)]
        valid = valid[
            (valid >= self.scene_memory_min_depth_m)
            & (valid <= self.scene_memory_max_depth_m)
        ]
        if valid.size < 8:
            return None, {'error': f'not enough valid depth samples in bbox: {valid.size}'}
        front_depth = float(np.percentile(valid, self.scene_memory_depth_percentile))
        depth_limit = front_depth + self.scene_memory_depth_margin_m

        rows, cols = np.indices(roi.shape)
        mask = (
            np.isfinite(roi)
            & (roi >= self.scene_memory_min_depth_m)
            & (roi <= min(self.scene_memory_max_depth_m, depth_limit))
        )
        if int(np.count_nonzero(mask)) < 8:
            mask = (
                np.isfinite(roi)
                & (roi >= self.scene_memory_min_depth_m)
                & (roi <= self.scene_memory_max_depth_m)
            )
        if int(np.count_nonzero(mask)) < 8:
            return None, {'error': 'not enough foreground depth samples after filtering'}

        z_values = roi[mask]
        u_values = cols[mask].astype(float) + float(left)
        v_values = rows[mask].astype(float) + float(top)
        z = float(np.median(z_values))
        u = float(np.median(u_values))
        v = float(np.median(v_values))
        camera_xyz = self._camera_xyz_from_pixel(u, v, z, camera_info)
        sample_count = int(np.count_nonzero(mask))
        stride = max(1, int(math.ceil(sample_count / 300.0)))
        sample_camera_points = []
        for sample_u, sample_v, sample_z in zip(u_values[::stride], v_values[::stride], z_values[::stride]):
            try:
                sample_camera_points.append(
                    self._camera_xyz_from_pixel(float(sample_u), float(sample_v), float(sample_z), camera_info)
                )
            except RuntimeError:
                pass
        try:
            base_xyz = self._transform_xyz_to_base(camera_xyz, depth_msg.header.frame_id or camera_info.header.frame_id)
            base_bounds = self._base_bounds_from_camera_points(
                sample_camera_points,
                depth_msg.header.frame_id or camera_info.header.frame_id,
            )
        except Exception as exc:
            return None, {'error': str(exc)}
        base_size = []
        if base_bounds:
            base_size = [
                float(base_bounds['max_x'] - base_bounds['min_x']),
                float(base_bounds['max_y'] - base_bounds['min_y']),
                float(base_bounds['max_z'] - base_bounds['min_z']),
            ]
        return base_xyz, {
            'camera_xyz': [float(value) for value in camera_xyz],
            'camera_frame': str(depth_msg.header.frame_id or camera_info.header.frame_id),
            'depth_m': z,
            'depth_sample_count': sample_count,
            'base_bounds': base_bounds,
            'base_size_xyz': base_size,
        }

    def _base_bounds_from_camera_points(
        self,
        camera_points: List[Tuple[float, float, float]],
        source_frame: str,
    ) -> Dict:
        if not camera_points:
            return {}
        transformed = [
            self._transform_xyz_to_base(point, source_frame)
            for point in camera_points
        ]
        xs = [point[0] for point in transformed]
        ys = [point[1] for point in transformed]
        zs = [point[2] for point in transformed]
        return {
            'min_x': float(min(xs)),
            'max_x': float(max(xs)),
            'min_y': float(min(ys)),
            'max_y': float(max(ys)),
            'min_z': float(min(zs)),
            'max_z': float(max(zs)),
        }

    @staticmethod
    def _camera_xyz_from_pixel(
        u: float,
        v: float,
        z: float,
        camera_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        k = list(camera_info.k)
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise RuntimeError('camera_info has invalid focal length')
        x = (float(u) - cx) * float(z) / fx
        y = (float(v) - cy) * float(z) / fy
        return x, y, float(z)

    def _transform_xyz_to_base(
        self,
        xyz: Tuple[float, float, float],
        source_frame: str,
    ) -> Tuple[float, float, float]:
        source_frame = str(source_frame or '').strip()
        if not source_frame:
            raise RuntimeError('depth/camera frame_id is empty; cannot transform scene memory')
        self._wait_for_tf_to_base(source_frame, 'scene memory point')
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=self.vision_tf_wait_timeout_sec),
        )
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        rx, ry, rz = self._rotate_vector_by_quaternion(
            (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )
        return (
            rx + float(translation.x),
            ry + float(translation.y),
            rz + float(translation.z),
        )

    def _wait_for_tf_to_base(self, source_frame: str, label: str) -> None:
        source_frame = str(source_frame or '').strip()
        if not source_frame:
            raise RuntimeError(f'{label} frame_id is empty; cannot transform to {self.base_frame}')
        deadline = time.monotonic() + max(0.1, self.vision_tf_wait_timeout_sec)
        last_error = ''
        while rclpy.ok() and not self._shutdown_requested.is_set():
            try:
                self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=max(0.1, self.ik_timeout_sec)),
                )
                return
            except TransformException as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        raise RuntimeError(
            f'Cannot transform {label} {source_frame} -> {self.base_frame} '
            f'after waiting {self.vision_tf_wait_timeout_sec:.1f}s: {last_error}'
        )

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        x, y, z = vector
        qx, qy, qz, qw = quat
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-12:
            return vector
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        return (
            x + qw * tx + (qy * tz - qz * ty),
            y + qw * ty + (qz * tx - qx * tz),
            z + qw * tz + (qx * ty - qy * tx),
        )

    def _ros_depth_image_to_meters(self, msg: RosImage, np_module):
        encoding = str(msg.encoding).lower()
        width = int(msg.width)
        height = int(msg.height)
        if encoding in ('16uc1', 'mono16'):
            dtype = np_module.uint16
            bytes_per_pixel = 2
            scale = 0.001
        elif encoding in ('32fc1',):
            dtype = np_module.float32
            bytes_per_pixel = 4
            scale = 1.0
        else:
            raise RuntimeError(
                f'Unsupported depth image encoding: {msg.encoding}; expected 16UC1 or 32FC1.'
            )
        expected_step = width * bytes_per_pixel
        packed = self._pack_image_rows(bytes(msg.data), height, int(msg.step), expected_step)
        depth = np_module.frombuffer(packed, dtype=dtype).reshape((height, width)).astype(np_module.float32)
        if scale != 1.0:
            depth *= scale
        return depth
