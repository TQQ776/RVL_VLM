from .mcp_shared import *


class PlanningMixin:
    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = copy.deepcopy(msg)

    def go_home_callback(self, request, response):
        if not self.motion_lock.acquire(blocking=False):
            response.success = False
            response.message = 'motion already running'
            return response
        try:
            response.success, response.message = self._go_home()
            return response
        finally:
            self.motion_lock.release()

    def _tool_move_axis(self, axis: str, arguments: Dict) -> Tuple[bool, str, Dict]:
        try:
            centimeters = float(arguments.get('centimeters', 0.0))
        except (TypeError, ValueError):
            return False, f'move_{axis}_cm failed: centimeters must be a number', {}
        if not self.motion_lock.acquire(blocking=False):
            return False, f'move_{axis}_cm failed: motion already running', {}
        try:
            success, message = self._move_axis(axis, centimeters)
            return (
                bool(success),
                f'move_{axis}_cm {"success" if success else "failed"}: {message}',
                {'axis': axis, 'centimeters': centimeters},
            )
        finally:
            self.motion_lock.release()

    def _current_ee_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Cannot get current transform {self.base_frame} -> {self.end_effector_frame}: {exc}'
            )
            return None
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = Quaternion(
            x=transform.transform.rotation.x,
            y=transform.transform.rotation.y,
            z=transform.transform.rotation.z,
            w=transform.transform.rotation.w,
        )
        return pose

    def _move_to_pose_with_move_group(
        self,
        target_pose: PoseStamped,
        label: str,
    ) -> Tuple[bool, str]:
        ok, message = self._execute_pose_with_move_group(
            target_pose,
            label,
            planner_id=self.place_move_group_planner_id,
            planning_attempts=self.place_num_planning_attempts,
            allowed_planning_time=self.place_allowed_planning_time_sec,
            duration_sec=self.place_move_duration_sec,
            velocity_scaling=self.place_max_velocity_scaling,
            acceleration_scaling=self.place_max_acceleration_scaling,
            position_tolerance_m=self.place_goal_position_tolerance_m,
            orientation_tolerance_rad=self.place_goal_orientation_tolerance_rad,
        )
        if ok:
            return True, f'{label}: {message}'
        return False, f'{label}: {message}'

    def _execute_pose_with_move_group(
        self,
        target_pose: PoseStamped,
        label: str,
        planner_id: str = '',
        planning_attempts: Optional[int] = None,
        allowed_planning_time: Optional[float] = None,
        duration_sec: Optional[float] = None,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
        position_tolerance_m: Optional[float] = None,
        orientation_tolerance_rad: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if not self.move_group_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'{label}: MoveGroup action not available: {self.move_group_action}'
        if self.latest_joint_state is None:
            return False, f'{label}: No /joint_states received; cannot plan MoveGroup pose goal.'

        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.planner_id = str(planner_id or self.move_group_planner_id or '')
        goal.request.num_planning_attempts = int(planning_attempts or self.num_planning_attempts)
        goal.request.allowed_planning_time = float(
            allowed_planning_time if allowed_planning_time is not None else self.allowed_planning_time
        )
        goal.request.max_velocity_scaling_factor = float(
            velocity_scaling if velocity_scaling is not None else self.max_velocity_scaling
        )
        goal.request.max_acceleration_scaling_factor = float(
            acceleration_scaling
            if acceleration_scaling is not None
            else self.max_acceleration_scaling
        )
        goal.request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(
            self._pose_goal_constraints(
                target_pose,
                position_tolerance_m or self.goal_joint_tolerance,
                orientation_tolerance_rad or self.place_goal_orientation_tolerance_rad,
            )
        )
        goal.planning_options.plan_only = self.plan_only
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.5
        goal.planning_options.planning_scene_diff.is_diff = True

        planner_text = goal.request.planner_id or 'default'
        self._publish_status(
            f'{label}: MoveGroup pose goal planner={planner_text}, '
            f'attempts={goal.request.num_planning_attempts}, '
            f'time={goal.request.allowed_planning_time:.1f}s'
        )
        return self._send_move_group_goal(
            goal,
            duration_sec if duration_sec is not None else self.motion_duration_sec,
        )

    def _execute_cartesian_pose_path(
        self,
        label: str,
        target_poses: List[PoseStamped],
        duration_sec: float,
        max_step_m: float,
        min_fraction: float,
    ) -> Tuple[bool, str]:
        if self.plan_only:
            return False, f'{label}: plan_only=true; Cartesian execution skipped'
        if self._emergency_stop_requested.is_set():
            return False, f'{label}: emergency stop is active; Cartesian execution skipped'
        if not target_poses:
            return False, f'{label}: no target poses'
        if not self.cartesian_path_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f'{label}: Cartesian path service not available: {self.cartesian_path_service}'
        if self.latest_joint_state is None:
            return False, f'{label}: No /joint_states received; cannot compute Cartesian path.'

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
        request.start_state.is_diff = True
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_frame
        request.waypoints = [copy.deepcopy(target.pose) for target in target_poses]
        request.max_step = max_step_m
        request.jump_threshold = self.axis_cartesian_jump_threshold
        request.avoid_collisions = self.avoid_collisions

        self._publish_status(
            f'{label} Cartesian request: waypoints={len(request.waypoints)}, max_step={max_step_m:.3f}m'
        )
        future = self.cartesian_path_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.allowed_planning_time + 1.0,
            f'{label} Cartesian path',
        )
        if response is None:
            return False, f'{label}: Cartesian path timed out'
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f'{label}: Cartesian path failed with {self._moveit_error_name(response.error_code.val)}',
            )
        if response.fraction < min_fraction:
            return False, (
                f'{label}: Cartesian path incomplete: '
                f'fraction={response.fraction:.3f}, required={min_fraction:.3f}'
            )
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            return False, f'{label}: Cartesian path returned an empty trajectory'

        duration = max(0.5, float(duration_sec))
        self._time_parameterize_cartesian_trajectory(trajectory, duration)
        ok, message = self._execute_joint_trajectory(
            trajectory,
            f'{label} Cartesian trajectory',
            duration + self.action_wait_timeout_sec,
        )
        if ok:
            message = f'{label}: Cartesian path executed successfully; fraction={response.fraction:.3f}'
            self._publish_status(message)
        return ok, message

    @staticmethod
    def _pose_summary(pose: PoseStamped) -> Dict:
        return {
            'frame_id': str(pose.header.frame_id),
            'position': {
                'x': float(pose.pose.position.x),
                'y': float(pose.pose.position.y),
                'z': float(pose.pose.position.z),
            },
            'orientation': {
                'x': float(pose.pose.orientation.x),
                'y': float(pose.pose.orientation.y),
                'z': float(pose.pose.orientation.z),
                'w': float(pose.pose.orientation.w),
            },
        }

    def move_axis_callback(self, axis: str, request: MoveAxis.Request, response: MoveAxis.Response):
        if not self.motion_lock.acquire(blocking=False):
            response.success = False
            response.message = 'motion already running'
            return response
        try:
            response.success, response.message = self._move_axis(axis, float(request.centimeters))
            return response
        finally:
            self.motion_lock.release()

    def _go_home(self):
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'action server not available: {self.trajectory_action}'
        start_positions = self._wait_for_current_joint_positions(self.home_wait_for_joint_state_sec)
        if start_positions is None:
            return (
                False,
                f'no complete joint state received on {self.joint_states_topic}; '
                'cannot build a smooth home trajectory',
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(
                seconds=max(0.0, self.home_trajectory_start_delay_sec)
            )
        ).to_msg()
        goal.trajectory.joint_names = list(self.joint_names)
        goal.trajectory.points = self._make_smooth_home_points(start_positions)
        goal.goal_time_tolerance = self._duration_msg(1.0)

        self._publish_status(
            f'go_home requested: smooth trajectory with {len(goal.trajectory.points)} points, '
            f'duration={self.home_move_duration_sec:.2f}s; '
            + ', '.join(
                f'{name}={deg:.1f}deg' for name, deg in zip(self.joint_names, self.home_joint_positions_deg)
            )
        )
        if self._emergency_stop_requested.is_set():
            return False, 'emergency stop is active; home motion not sent'
        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'home goal')
        if goal_handle is None:
            return False, 'timed out sending home goal'
        if not goal_handle.accepted:
            return False, 'home goal rejected'
        self._register_active_goal('home trajectory', goal_handle)

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.home_move_duration_sec + self.action_wait_timeout_sec,
            'home result',
        )
        self._unregister_active_goal('home trajectory', goal_handle)
        if action_result is None:
            return False, 'timed out waiting for home result'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = 'home motion finished successfully'
            self._publish_status(message)
            return True, message

        message = f'home motion failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _go_home_half_speed(self) -> Tuple[bool, str, Dict]:
        previous_duration = self.home_move_duration_sec
        try:
            self.home_move_duration_sec = max(0.5, previous_duration * 2.0)
            ok, message = self._go_home()
        finally:
            self.home_move_duration_sec = previous_duration
        return ok, message, {
            'home_move_duration_sec': previous_duration * 2.0,
            'speed_scale_vs_button': 0.5,
        }

    def _register_active_goal(self, label: str, goal_handle) -> None:
        if goal_handle is None:
            return
        key = f'{label}:{id(goal_handle)}'
        with self._active_goal_lock:
            self._active_goal_handles[key] = goal_handle

    def _unregister_active_goal(self, label: str, goal_handle) -> None:
        if goal_handle is None:
            return
        key = f'{label}:{id(goal_handle)}'
        with self._active_goal_lock:
            self._active_goal_handles.pop(key, None)

    def _request_emergency_stop(self) -> Tuple[str, Dict]:
        self._emergency_stop_requested.set()
        stamp = time.time()
        canceled = 0
        hold_ok = False
        hold_message = 'not attempted'
        try:
            self._publish_status('EMERGENCY_STOP requested: canceling active goals and holding current arm pose')
            with self.grasp_results_lock:
                pending_grasp_events = list(self.grasp_result_events.values())
            for event in pending_grasp_events:
                event.set()
            try:
                self.emergency_stop_pub.publish(String(data=json.dumps({
                    'event': 'emergency_stop',
                    'stamp': stamp,
                    'source': self.get_name(),
                }, ensure_ascii=False)))
            except Exception as exc:
                self.get_logger().warn(f'Failed to publish emergency stop event: {exc}')

            canceled = self._cancel_active_goals(timeout_sec=1.0)
            hold_ok, hold_message = self._send_hold_current_position(timeout_sec=1.0)

            message = (
                f'emergency_stop requested: cancel requests sent for {canceled} active goal(s); '
                f'hold_current_position={"sent" if hold_ok else "not sent"} ({hold_message})'
            )
            self.get_logger().warn(message)
            self._publish_status(message)
        finally:
            # Keep the flag up briefly so in-flight wait loops see the stop, then make
            # the stop one-shot so the next user command can run normally.
            time.sleep(0.25)
            self._emergency_stop_requested.clear()
            self._publish_status('EMERGENCY_STOP reset: ready for the next command')

        result = {
            'canceled_goals': canceled,
            'hold_sent': hold_ok,
            'hold_message': hold_message,
            'stamp': stamp,
            'ready_for_next_command': True,
        }
        return message, result

    def _cancel_active_goals(self, timeout_sec: float = 1.0) -> int:
        with self._active_goal_lock:
            items = list(self._active_goal_handles.items())
            self._active_goal_handles.clear()
        canceled = 0
        for label, goal_handle in items:
            try:
                future = goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(f'Failed to request cancel for {label}: {exc}')
                continue
            event = threading.Event()
            future.add_done_callback(lambda _: event.set())
            event.wait(timeout=max(0.05, float(timeout_sec)))
            canceled += 1
        return canceled

    def _send_hold_current_position(self, timeout_sec: float = 1.0) -> Tuple[bool, str]:
        if self.latest_joint_state is None:
            return False, 'no /joint_states available'
        positions = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        if not all(name in positions for name in self.joint_names):
            return False, 'current joint state is incomplete'
        if not self.trajectory_client.wait_for_server(timeout_sec=max(0.05, timeout_sec)):
            return False, f'trajectory action unavailable: {self.trajectory_action}'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(positions[name]) for name in self.joint_names]
        point.velocities = [0.0 for _ in self.joint_names]
        point.accelerations = [0.0 for _ in self.joint_names]
        point.time_from_start = self._duration_msg(0.05)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = self._duration_msg(0.1)

        send_future = self.trajectory_client.send_goal_async(goal)
        event = threading.Event()
        send_future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=max(0.05, timeout_sec)):
            return False, 'timed out sending hold trajectory'
        try:
            goal_handle = send_future.result()
        except Exception as exc:
            return False, f'hold trajectory send failed: {exc}'
        if goal_handle is None or not goal_handle.accepted:
            return False, 'hold trajectory rejected'
        return True, 'hold trajectory accepted'

    def _wait_for_current_joint_positions(self, timeout_sec: float) -> Optional[Dict[str, float]]:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while time.monotonic() <= deadline:
            if self.latest_joint_state is not None:
                positions = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
                if all(name in positions for name in self.joint_names):
                    return {
                        name: float(positions[name]) for name in self.joint_names
                    }
            time.sleep(0.02)
        return None

    def _make_smooth_home_points(self, start_positions: Dict[str, float]) -> List[JointTrajectoryPoint]:
        goal_positions = {
            name: math.radians(value)
            for name, value in zip(self.joint_names, self.home_joint_positions_deg)
        }
        duration = max(0.5, self.home_move_duration_sec)
        dt = min(max(0.01, self.home_trajectory_dt_sec), duration)
        steps = max(2, int(math.ceil(duration / dt)))

        points: List[JointTrajectoryPoint] = []
        for index in range(steps + 1):
            elapsed = min(duration, duration * index / steps)
            u = 0.0 if duration <= 0.0 else elapsed / duration
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
            blend_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration * duration)

            point = JointTrajectoryPoint()
            point.positions = [
                start_positions[name] + (goal_positions[name] - start_positions[name]) * blend
                for name in self.joint_names
            ]
            point.velocities = [
                (goal_positions[name] - start_positions[name]) * blend_dot
                for name in self.joint_names
            ]
            point.accelerations = [
                (goal_positions[name] - start_positions[name]) * blend_ddot
                for name in self.joint_names
            ]
            point.time_from_start = self._duration_msg(elapsed)
            points.append(point)
        return points

    def _move_axis(
        self,
        axis: str,
        centimeters: float,
        min_fraction: Optional[float] = None,
        avoid_collisions: Optional[bool] = None,
    ):
        if abs(centimeters) > self.max_single_axis_move_cm:
            message = (
                f'move_{axis}_cm rejected: requested {centimeters:.2f} cm exceeds '
                f'single-step limit {self.max_single_axis_move_cm:.2f} cm'
            )
            self.get_logger().warn(message)
            self._publish_status(message)
            return False, message

        target_pose = self._make_offset_pose(axis, centimeters)
        if target_pose is None:
            return False, 'failed to build target pose from current end-effector transform'

        self._publish_status(
            f'move_{axis}_cm requested: {centimeters:.2f} cm; '
            f'target=({target_pose.pose.position.x:.3f}, '
            f'{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})'
        )

        if self.axis_move_execution_mode == 'cartesian':
            return self._execute_axis_cartesian_path(
                axis,
                centimeters,
                target_pose,
                min_fraction=min_fraction,
                avoid_collisions=avoid_collisions,
            )

        joint_goal = self._compute_ik(target_pose, avoid_collisions=avoid_collisions)
        if joint_goal is None:
            return False, 'MoveIt IK failed'

        if self.axis_move_execution_mode == 'ik_only':
            message = f'IK solved without execution: {self._format_joint_goal(joint_goal)}'
            self._publish_status(message)
            return True, message
        if self.axis_move_execution_mode == 'move_group':
            return self._execute_with_move_group(joint_goal)
        return self._execute_with_trajectory_action(joint_goal)

    def _make_offset_pose(self, axis: str, centimeters: float) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                Time(),
                timeout=Duration(seconds=self.ik_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Cannot get current transform {self.base_frame} -> {self.end_effector_frame}: {exc}'
            )
            return None

        delta_m = centimeters / 100.0
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        if axis == 'x':
            pose.pose.position.x += delta_m
        elif axis == 'y':
            pose.pose.position.y += delta_m
        elif axis == 'z':
            pose.pose.position.z += delta_m
        else:
            raise ValueError(f'unsupported axis: {axis}')
        pose.pose.orientation = Quaternion(
            x=transform.transform.rotation.x,
            y=transform.transform.rotation.y,
            z=transform.transform.rotation.z,
            w=transform.transform.rotation.w,
        )
        return pose

    def _compute_ik(
        self,
        target_pose: PoseStamped,
        avoid_collisions: Optional[bool] = None,
    ) -> Optional[Dict[str, float]]:
        if not self.ik_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().error(f'IK service not available: {self.ik_service}')
            return None
        if self.latest_joint_state is None:
            self.get_logger().error('No /joint_states received; cannot seed MoveIt IK.')
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.move_group_name
        request.ik_request.robot_state.joint_state = copy.deepcopy(self.latest_joint_state)
        request.ik_request.robot_state.is_diff = True
        request.ik_request.avoid_collisions = (
            self.avoid_collisions if avoid_collisions is None else bool(avoid_collisions)
        )
        request.ik_request.ik_link_name = self.end_effector_frame
        request.ik_request.pose_stamped = target_pose
        request.ik_request.timeout = self._duration_msg(self.ik_timeout_sec)

        future = self.ik_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.ik_timeout_sec + 1.0,
            'IK response',
        )
        if response is None:
            return None
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'MoveIt IK failed with {self._moveit_error_name(response.error_code.val)}'
            )
            return None

        positions = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        missing = [name for name in self.joint_names if name not in positions]
        if missing:
            self.get_logger().error(f'IK solution missing joints: {missing}')
            return None

        joint_goal = {name: float(positions[name]) for name in self.joint_names}
        self._publish_status(f'IK solved: {self._format_joint_goal(joint_goal)}')
        return joint_goal

    def _execute_axis_cartesian_path(
        self,
        axis: str,
        centimeters: float,
        target_pose: PoseStamped,
        min_fraction: Optional[float] = None,
        avoid_collisions: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        if self.plan_only:
            return False, 'plan_only=true; Cartesian axis execution skipped'
        if not self.cartesian_path_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f'Cartesian path service not available: {self.cartesian_path_service}'
        if self.latest_joint_state is None:
            return False, 'No /joint_states received; cannot compute Cartesian axis path.'

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
        request.start_state.is_diff = True
        request.group_name = self.move_group_name
        request.link_name = self.end_effector_frame
        request.waypoints = [copy.deepcopy(target_pose.pose)]
        request.max_step = self.axis_cartesian_max_step_m
        request.jump_threshold = self.axis_cartesian_jump_threshold
        request.avoid_collisions = (
            self.avoid_collisions if avoid_collisions is None else bool(avoid_collisions)
        )

        self._publish_status(
            f'Cartesian move_{axis}_cm request: {centimeters:.2f} cm, '
            f'max_step={self.axis_cartesian_max_step_m:.3f}m, '
            f'avoid_collisions={request.avoid_collisions}'
        )
        future = self.cartesian_path_client.call_async(request)
        response = self._wait_for_future(
            future,
            self.service_wait_timeout_sec + self.allowed_planning_time + 1.0,
            f'move_{axis}_cm Cartesian path',
        )
        if response is None:
            return False, f'move_{axis}_cm Cartesian path timed out'
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f'move_{axis}_cm Cartesian path failed with '
                f'{self._moveit_error_name(response.error_code.val)}',
            )
        required_fraction = self.axis_cartesian_min_fraction
        if min_fraction is not None:
            required_fraction = min(1.0, max(0.0, float(min_fraction)))
        if response.fraction < required_fraction:
            return False, (
                f'move_{axis}_cm Cartesian path incomplete: '
                f'fraction={response.fraction:.3f}, required={required_fraction:.3f}'
            )

        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            return False, f'move_{axis}_cm Cartesian path returned an empty trajectory'

        duration = max(
            self.axis_cartesian_min_duration_sec,
            abs(float(centimeters)) * self.axis_cartesian_duration_per_cm_sec,
        )
        self._time_parameterize_cartesian_trajectory(trajectory, duration)
        ok, message = self._execute_joint_trajectory(
            trajectory,
            f'move_{axis}_cm Cartesian trajectory',
            duration + self.action_wait_timeout_sec,
        )
        if ok:
            message = (
                f'move_{axis}_cm Cartesian path executed successfully; '
                f'fraction={response.fraction:.3f}'
            )
            self._publish_status(message)
        return ok, message

    def _time_parameterize_cartesian_trajectory(
        self,
        trajectory: JointTrajectory,
        total_duration_sec: float,
    ) -> None:
        points = trajectory.points
        if not points:
            return
        duration = max(0.1, float(total_duration_sec))
        if len(points) == 1:
            positions = list(points[0].positions)
            points[0].time_from_start = self._duration_msg(duration)
            points[0].velocities = [0.0 for _ in positions]
            points[0].accelerations = [0.0 for _ in positions]
            return

        min_step = 0.05
        effective_duration = max(min_step, duration - min_step)
        last_positions = [float(value) for value in points[-1].positions]
        for index, point in enumerate(points):
            u = index / float(len(points) - 1)
            positions = [float(value) for value in point.positions]
            if index == 0:
                next_positions = [float(value) for value in points[index + 1].positions]
                dt = effective_duration / float(len(points) - 1)
                velocities = [
                    (next_positions[joint_index] - positions[joint_index]) / dt
                    for joint_index in range(len(positions))
                ]
            elif index == len(points) - 1:
                velocities = [0.0 for _ in positions]
            else:
                previous_positions = [float(value) for value in points[index - 1].positions]
                next_positions = [float(value) for value in points[index + 1].positions]
                dt = 2.0 * effective_duration / float(len(points) - 1)
                velocities = [
                    (next_positions[joint_index] - previous_positions[joint_index]) / dt
                    for joint_index in range(len(positions))
                ]
            if index == 0:
                velocities = [0.0 for _ in positions]
            point.positions = positions if index < len(points) - 1 else last_positions
            point.velocities = velocities
            point.accelerations = [0.0 for _ in positions]
            point.time_from_start = self._duration_msg(min_step + effective_duration * u)

    @staticmethod
    def _duration_to_sec(duration_msg: DurationMsg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    def _trajectory_has_valid_timing(self, trajectory: JointTrajectory) -> bool:
        previous: Optional[float] = None
        for point in trajectory.points:
            current = self._duration_to_sec(point.time_from_start)
            if current < 0.0:
                return False
            if previous is not None and current <= previous:
                return False
            previous = current
        return previous is not None and previous > 0.0

    def _trajectory_duration_sec(
        self,
        trajectory: JointTrajectory,
        fallback_duration_sec: float,
    ) -> float:
        if not trajectory.points:
            return max(0.1, float(fallback_duration_sec))
        duration = self._duration_to_sec(trajectory.points[-1].time_from_start)
        return max(0.1, duration if duration > 0.0 else float(fallback_duration_sec))

    def _execute_with_move_group(self, joint_goal: Dict[str, float]):
        if not self.move_group_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'MoveGroup action not available: {self.move_group_action}'
        if self.latest_joint_state is None:
            return False, 'No /joint_states received; cannot plan MoveGroup goal.'

        ok, guard_message = self._joint_goal_within_moveit_limits(
            joint_goal,
            copy.deepcopy(self.latest_joint_state),
            'MoveGroup joint goal',
        )
        if not ok:
            self.get_logger().warn(guard_message)
            self._publish_status(guard_message)
            return False, guard_message

        goal = MoveGroup.Goal()
        goal.request.group_name = self.move_group_name
        goal.request.planner_id = self.move_group_planner_id
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling
        goal.request.start_state.joint_state = copy.deepcopy(self.latest_joint_state)
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(self._joint_goal_constraints(joint_goal))
        goal.planning_options.plan_only = self.plan_only
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.5
        goal.planning_options.planning_scene_diff.is_diff = True

        return self._send_move_group_goal(goal, self.motion_duration_sec)

    def _send_move_group_goal(
        self,
        goal: MoveGroup.Goal,
        duration_sec: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if self._emergency_stop_requested.is_set():
            return False, 'emergency stop is active; MoveGroup goal not sent'
        if not self.plan_only:
            plan_ok, plan_message, planned_robot_trajectory = self._select_guarded_move_group_plan(
                goal,
                duration_sec,
            )
            if not plan_ok:
                return False, plan_message
            planned_trajectory = planned_robot_trajectory.joint_trajectory
            fallback_duration = duration_sec if duration_sec is not None else self.motion_duration_sec
            if not self._trajectory_has_valid_timing(planned_trajectory):
                self._time_parameterize_cartesian_trajectory(
                    planned_trajectory,
                    fallback_duration,
                )
            planned_robot_trajectory.joint_trajectory = planned_trajectory
            execution_timeout = self._trajectory_duration_sec(
                planned_trajectory,
                fallback_duration,
            ) + self.action_wait_timeout_sec
            if self.move_group_execution_backend in {'controller', 'trajectory_controller'}:
                self._publish_status(
                    'MoveGroup guarded trajectory selected; executing via arm controller action'
                )
                return self._execute_joint_trajectory(
                    planned_trajectory,
                    'MoveGroup guarded controller trajectory',
                    execution_timeout,
                )
            return self._execute_moveit_trajectory(
                planned_robot_trajectory,
                'MoveGroup guarded trajectory',
                execution_timeout,
            )

        send_future = self.move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'MoveGroup goal')
        if goal_handle is None:
            return False, 'timed out sending MoveGroup goal'
        if not goal_handle.accepted:
            return False, 'MoveGroup rejected the goal'
        self._register_active_goal('MoveGroup', goal_handle)

        result_future = goal_handle.get_result_async()
        result_timeout_sec = (
            float(goal.request.allowed_planning_time)
            + self.action_wait_timeout_sec
            + float(duration_sec if duration_sec is not None else self.motion_duration_sec)
        ) * self.move_group_result_timeout_scale
        action_result = self._wait_for_future(
            result_future,
            result_timeout_sec,
            'MoveGroup result',
        )
        self._unregister_active_goal('MoveGroup', goal_handle)
        if action_result is None:
            return False, f'timed out waiting for MoveGroup result after {result_timeout_sec:.1f}s'

        result = action_result.result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            mode = 'planned' if self.plan_only else 'planned and executed'
            message = f'MoveGroup {mode} successfully'
            self._publish_status(message)
            return True, message

        message = f'MoveGroup failed with {self._moveit_error_name(result.error_code.val)}'
        self.get_logger().error(message)
        return False, message

    def _select_guarded_move_group_plan(
        self,
        goal: MoveGroup.Goal,
        duration_sec: Optional[float],
    ) -> Tuple[bool, str, Optional[RobotTrajectory]]:
        best_ok_plan: Optional[RobotTrajectory] = None
        best_ok_message = ''
        best_ok_score = float('inf')
        best_rejected_message = ''
        best_rejected_score = float('inf')
        attempts = max(1, int(getattr(self, 'moveit_guard_plan_retries', 1)))
        for attempt in range(attempts):
            plan_ok, plan_message, planned_robot_trajectory = self._plan_move_group_goal_for_guard(
                goal,
                duration_sec,
                attempt_index=attempt,
            )
            if not plan_ok or planned_robot_trajectory is None:
                best_rejected_message = plan_message
                continue
            ok, guard_message, metrics = self._trajectory_within_moveit_limits(
                planned_robot_trajectory.joint_trajectory,
                f'MoveGroup planned trajectory attempt {attempt + 1}/{attempts}',
            )
            score = float(metrics.get('total_delta', float('inf'))) if metrics else float('inf')
            if ok:
                if score < best_ok_score:
                    best_ok_score = score
                    best_ok_plan = planned_robot_trajectory
                    best_ok_message = guard_message
                self._publish_status(guard_message)
            else:
                if score < best_rejected_score:
                    best_rejected_score = score
                    best_rejected_message = guard_message
                self.get_logger().warn(guard_message)
                self._publish_status(guard_message)
        if best_ok_plan is not None:
            message = (
                f'MoveGroup selected guarded trajectory: {best_ok_message}; '
                f'best_total_joint_delta={math.degrees(best_ok_score):.1f}deg'
            )
            self._publish_status(message)
            return True, message, best_ok_plan
        if best_rejected_message:
            return False, best_rejected_message, None
        return False, 'MoveGroup guard could not find a valid plan', None

    def _plan_move_group_goal_for_guard(
        self,
        goal: MoveGroup.Goal,
        duration_sec: Optional[float],
        attempt_index: int = 0,
    ) -> Tuple[bool, str, Optional[RobotTrajectory]]:
        plan_goal = copy.deepcopy(goal)
        plan_goal.planning_options.plan_only = True
        if attempt_index > 0:
            plan_goal.request.num_planning_attempts = max(1, int(goal.request.num_planning_attempts))
        if self._emergency_stop_requested.is_set():
            return False, 'emergency stop is active; MoveGroup guard plan not sent', None
        send_future = self.move_group_client.send_goal_async(plan_goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.action_wait_timeout_sec,
            'MoveGroup guard plan goal',
        )
        if goal_handle is None:
            return False, 'timed out sending MoveGroup guard plan goal', None
        if not goal_handle.accepted:
            return False, 'MoveGroup rejected the guard plan goal', None
        self._register_active_goal('MoveGroup guard plan', goal_handle)
        result_timeout_sec = (
            float(plan_goal.request.allowed_planning_time)
            + self.action_wait_timeout_sec
            + float(duration_sec if duration_sec is not None else self.motion_duration_sec)
        ) * self.move_group_result_timeout_scale
        action_result = self._wait_for_future(
            goal_handle.get_result_async(),
            result_timeout_sec,
            'MoveGroup guard plan result',
        )
        self._unregister_active_goal('MoveGroup guard plan', goal_handle)
        if action_result is None:
            return (
                False,
                f'timed out waiting for MoveGroup guard plan after {result_timeout_sec:.1f}s',
                None,
            )
        result = action_result.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f'MoveGroup guard plan failed with {self._moveit_error_name(result.error_code.val)}',
                None,
            )
        planned_robot_trajectory = copy.deepcopy(result.planned_trajectory)
        trajectory = planned_robot_trajectory.joint_trajectory
        if not trajectory.points:
            return False, 'MoveGroup guard plan returned an empty trajectory', None
        return True, 'MoveGroup guard plan accepted', planned_robot_trajectory

    def _execute_moveit_trajectory(
        self,
        trajectory: RobotTrajectory,
        label: str,
        timeout_sec: float,
    ) -> Tuple[bool, str]:
        if not self.execute_trajectory_client.wait_for_server(
            timeout_sec=self.action_wait_timeout_sec,
        ):
            self._publish_status(
                f'{label}: ExecuteTrajectory unavailable; using controller trajectory action'
            )
            return self._execute_joint_trajectory(
                trajectory.joint_trajectory,
                label,
                timeout_sec,
            )

        ok, guard_message, _ = self._trajectory_within_moveit_limits(
            trajectory.joint_trajectory,
            label,
        )
        if not ok:
            self.get_logger().warn(guard_message)
            self._publish_status(guard_message)
            return False, guard_message

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        if self._emergency_stop_requested.is_set():
            return False, f'{label}: emergency stop is active; ExecuteTrajectory not sent'
        send_future = self.execute_trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.action_wait_timeout_sec,
            f'{label} execute goal',
        )
        if goal_handle is None:
            return False, f'timed out sending {label} to ExecuteTrajectory'
        if not goal_handle.accepted:
            return False, f'ExecuteTrajectory rejected {label}'
        self._register_active_goal(label, goal_handle)

        action_result = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout_sec,
            f'{label} execute result',
        )
        self._unregister_active_goal(label, goal_handle)
        if action_result is None:
            return False, f'timed out waiting for ExecuteTrajectory {label}'
        result = action_result.result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            message = f'{label} executed successfully via MoveIt ExecuteTrajectory'
            self._publish_status(message)
            return True, message
        message = f'{label} ExecuteTrajectory failed with {self._moveit_error_name(result.error_code.val)}'
        self.get_logger().error(message)
        return False, message

    def _trajectory_within_moveit_limits(
        self,
        trajectory: JointTrajectory,
        label: str,
        joint_state: Optional[JointState] = None,
    ) -> Tuple[bool, str, Dict]:
        if self.max_moveit_joint_delta_rad <= 0.0 and self.max_moveit_total_joint_delta_rad <= 0.0:
            return True, f'{label}: joint delta guard disabled', {}
        joint_state = joint_state or self.latest_joint_state
        if joint_state is None:
            return False, f'{label} rejected: no /joint_states for joint delta guard', {}
        current = dict(zip(joint_state.name, joint_state.position))
        names = list(trajectory.joint_names)
        if not names:
            names = list(self.joint_names)
        ignored = set(getattr(self, 'moveit_joint_delta_guard_ignore_joints', set()))
        checked_indexes = [(index, name) for index, name in enumerate(names) if name not in ignored]
        if not checked_indexes:
            return True, f'{label}: all trajectory joints ignored by joint delta guard', {}
        missing = [name for _, name in checked_indexes if name not in current]
        if missing:
            return False, f'{label} rejected: current joint state missing {missing}', {}
        start = {name: float(current[name]) for _, name in checked_indexes}
        max_joint_delta = 0.0
        max_joint_name = ''
        max_total_delta = 0.0
        max_total_deltas: List[Tuple[str, float]] = []
        for point in trajectory.points:
            positions = list(point.positions)
            if len(positions) < len(names):
                return False, f'{label} rejected: trajectory point has incomplete positions', {}
            deltas_by_name = [
                (name, abs(self._angle_delta(float(positions[index]), start[name])))
                for index, name in checked_indexes
            ]
            deltas = [delta for _, delta in deltas_by_name]
            if deltas:
                local_max = max(deltas)
                if local_max > max_joint_delta:
                    max_joint_delta = local_max
                    max_joint_name = checked_indexes[deltas.index(local_max)][1]
                local_total = sum(deltas)
                if local_total > max_total_delta:
                    max_total_delta = local_total
                    max_total_deltas = deltas_by_name
        metrics = {
            'max_joint_delta': max_joint_delta,
            'max_joint_name': max_joint_name,
            'total_delta': max_total_delta,
            'total_deltas': max_total_deltas,
        }
        if (
            self.max_moveit_joint_delta_rad > 0.0
            and max_joint_delta > self.max_moveit_joint_delta_rad
        ):
            return False, (
                f'{label} rejected: joint {max_joint_name} delta '
                f'{math.degrees(max_joint_delta):.1f}deg exceeds limit '
                f'{math.degrees(self.max_moveit_joint_delta_rad):.1f}deg'
            ), metrics
        if (
            self.max_moveit_total_joint_delta_rad > 0.0
            and max_total_delta > self.max_moveit_total_joint_delta_rad
        ):
            top_deltas = self._format_joint_delta_summary(max_total_deltas)
            return False, (
                f'{label} rejected: total joint delta '
                f'{math.degrees(max_total_delta):.1f}deg exceeds limit '
                f'{math.degrees(self.max_moveit_total_joint_delta_rad):.1f}deg'
                f'{top_deltas}'
            ), metrics
        return True, (
            f'{label} accepted: max_joint_delta={math.degrees(max_joint_delta):.1f}deg, '
            f'total_joint_delta={math.degrees(max_total_delta):.1f}deg'
        ), metrics

    @staticmethod
    def _format_joint_delta_summary(deltas_by_name: List[Tuple[str, float]], limit: int = 4) -> str:
        if not deltas_by_name:
            return ''
        top = sorted(deltas_by_name, key=lambda item: item[1], reverse=True)[:limit]
        summary = ', '.join(f'{name}={math.degrees(delta):.1f}deg' for name, delta in top)
        return f'; largest checked joint deltas: {summary}'

    def _joint_goal_within_moveit_limits(
        self,
        joint_goal: Dict[str, float],
        joint_state: Optional[JointState],
        label: str,
    ) -> Tuple[bool, str]:
        if joint_state is None:
            return False, f'{label} rejected: no /joint_states for joint delta guard'
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(joint_goal[name]) for name in self.joint_names]
        trajectory.points = [point]
        ok, message, _ = self._trajectory_within_moveit_limits(
            trajectory,
            label,
            joint_state=joint_state,
        )
        return ok, message

    @staticmethod
    def _angle_delta(target: float, start: float) -> float:
        return math.atan2(math.sin(target - start), math.cos(target - start))

    def _execute_with_trajectory_action(self, joint_goal: Dict[str, float]):
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'Trajectory action not available: {self.trajectory_action}'

        ok, guard_message = self._joint_goal_within_moveit_limits(
            joint_goal,
            self.latest_joint_state,
            'direct trajectory joint goal',
        )
        if not ok:
            self.get_logger().warn(guard_message)
            self._publish_status(guard_message)
            return False, guard_message

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = [joint_goal[name] for name in self.joint_names]
        point.time_from_start = self._duration_msg(self.motion_duration_sec)
        goal.trajectory.points.append(point)
        goal.goal_time_tolerance = self._duration_msg(1.0)

        if self._emergency_stop_requested.is_set():
            return False, 'emergency stop is active; trajectory goal not sent'
        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, self.action_wait_timeout_sec, 'trajectory goal')
        if goal_handle is None:
            return False, 'timed out sending trajectory goal'
        if not goal_handle.accepted:
            return False, 'trajectory controller rejected the goal'
        self._register_active_goal('direct trajectory', goal_handle)

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.motion_duration_sec + self.action_wait_timeout_sec,
            'trajectory result',
        )
        self._unregister_active_goal('direct trajectory', goal_handle)
        if action_result is None:
            return False, 'timed out waiting for trajectory result'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = 'trajectory target executed successfully'
            self._publish_status(message)
            return True, message

        message = f'Trajectory failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _execute_joint_trajectory(
        self,
        trajectory: JointTrajectory,
        label: str,
        timeout_sec: float,
    ) -> Tuple[bool, str]:
        if not self.trajectory_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            return False, f'Trajectory action not available: {self.trajectory_action}'

        ok, guard_message, _ = self._trajectory_within_moveit_limits(trajectory, label)
        if not ok:
            self.get_logger().warn(guard_message)
            self._publish_status(guard_message)
            return False, guard_message

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(
                seconds=max(0.0, self.home_trajectory_start_delay_sec)
            )
        ).to_msg()
        goal.goal_time_tolerance = self._duration_msg(1.0)

        if self._emergency_stop_requested.is_set():
            return False, f'emergency stop is active; {label} not sent'
        send_future = self.trajectory_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.action_wait_timeout_sec,
            f'{label} goal',
        )
        if goal_handle is None:
            return False, f'timed out sending {label}'
        if not goal_handle.accepted:
            return False, f'trajectory controller rejected {label}'
        self._register_active_goal(label, goal_handle)

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(result_future, timeout_sec, f'{label} result')
        self._unregister_active_goal(label, goal_handle)
        if action_result is None:
            return False, f'timed out waiting for {label}'

        result = action_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            message = f'{label} executed successfully'
            self._publish_status(message)
            return True, message

        message = f'{label} failed: {result.error_code} {result.error_string}'
        self.get_logger().error(message)
        return False, message

    def _joint_goal_constraints(self, joint_goal: Dict[str, float]) -> Constraints:
        constraints = Constraints()
        constraints.name = 'mcp_ik_joint_goal'
        for name in self.joint_names:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = joint_goal[name]
            joint_constraint.tolerance_above = self.goal_joint_tolerance
            joint_constraint.tolerance_below = self.goal_joint_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return constraints

    def _pose_goal_constraints(
        self,
        target_pose: PoseStamped,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> Constraints:
        constraints = Constraints()
        constraints.name = 'mcp_pose_goal'

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [max(0.001, float(position_tolerance_m))]
        region = BoundingVolume()
        region.primitives.append(primitive)
        region.primitive_poses.append(copy.deepcopy(target_pose.pose))

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = target_pose.header.frame_id or self.base_frame
        position_constraint.link_name = self.end_effector_frame
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = target_pose.header.frame_id or self.base_frame
        orientation_constraint.link_name = self.end_effector_frame
        orientation_constraint.orientation = copy.deepcopy(target_pose.pose.orientation)
        tolerance = max(0.001, float(orientation_tolerance_rad))
        orientation_constraint.absolute_x_axis_tolerance = tolerance
        orientation_constraint.absolute_y_axis_tolerance = tolerance
        orientation_constraint.absolute_z_axis_tolerance = tolerance
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

        return constraints

    def _wait_for_future(self, future, timeout_sec: float, label: str):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while not event.is_set():
            if (
                self._shutdown_requested.is_set()
                or self._emergency_stop_requested.is_set()
                or not rclpy.ok()
            ):
                self.get_logger().warn(f'Interrupted while waiting for {label}; stop requested.')
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            event.wait(timeout=min(0.1, remaining))
        if not event.is_set():
            self.get_logger().error(f'Timed out waiting for {label}.')
            return None
        try:
            return future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed waiting for {label}: {exc}')
            return None

    @staticmethod
    def _moveit_error_name(code: int) -> str:
        code = int(code)
        for name in dir(MoveItErrorCodes):
            if not name.isupper():
                continue
            try:
                if int(getattr(MoveItErrorCodes, name)) == code:
                    return f'{code}({name})'
            except (TypeError, ValueError):
                continue
        return str(code)

    @staticmethod
    def _duration_msg(seconds: float) -> DurationMsg:
        seconds = max(0.0, float(seconds))
        whole = int(math.floor(seconds))
        nanosec = int(round((seconds - whole) * 1e9))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = DurationMsg()
        msg.sec = whole
        msg.nanosec = nanosec
        return msg

    @staticmethod
    def _format_joint_goal(joint_goal: Dict[str, float]) -> str:
        return ', '.join(f'{name}={value:.3f}' for name, value in joint_goal.items())
