from .mcp_shared import *


class GripperMixin:
    def open_gripper_callback(self, request, response):
        del request
        if not self.gripper_lock.acquire(blocking=False):
            response.success = False
            response.message = 'gripper action already running'
            return response
        try:
            response.success, response.message = self._open_gripper()
            return response
        finally:
            self.gripper_lock.release()

    def close_gripper_callback(self, request, response):
        del request
        if not self.gripper_lock.acquire(blocking=False):
            response.success = False
            response.message = 'gripper action already running'
            return response
        try:
            response.success, response.message = self._close_gripper()
            return response
        finally:
            self.gripper_lock.release()

    def _open_gripper(self, detach_attached: bool = True) -> Tuple[bool, str]:
        client_name, client = self._available_action_client(
            self.gripper_move_clients,
            'gripper move',
        )
        if client is None:
            return False, (
                'no gripper move action available; tried '
                + ', '.join(self.gripper_move_actions)
            )

        goal = GripperMove.Goal()
        goal.width = self.gripper_open_width_m
        goal.speed = self.gripper_open_speed_mps
        ok, message = self._send_gripper_goal(client, goal, client_name, 'open_gripper')
        if ok:
            self._publish_status(message)
            if detach_attached:
                detach_ok, detach_message = self._detach_current_held_object()
                if not detach_ok:
                    return False, f'{message}; failed to detach held object model: {detach_message}'
                if detach_message != 'no attached held object to detach':
                    message = f'{message}; {detach_message}'
        return ok, message

    def _close_gripper(self) -> Tuple[bool, str]:
        client_name, client = self._available_action_client(
            self.gripper_grasp_clients,
            'gripper grasp',
        )
        if client is None:
            return False, (
                'no gripper grasp action available; tried '
                + ', '.join(self.gripper_grasp_actions)
            )

        goal = GripperGrasp.Goal()
        goal.width = self.gripper_close_width_m
        goal.speed = self.gripper_close_speed_mps
        goal.force = self.gripper_close_force_n
        goal.epsilon.inner = self.gripper_grasp_epsilon_inner_m
        goal.epsilon.outer = self.gripper_grasp_epsilon_outer_m
        ok, message = self._send_gripper_goal(client, goal, client_name, 'close_gripper')
        if ok:
            self._publish_status(message)
        return ok, message

    def _available_action_client(self, clients, label: str):
        for action_name, client in clients:
            if client.wait_for_server(timeout_sec=self.gripper_server_wait_timeout_sec):
                return action_name, client
        self.get_logger().warn(f'No available {label} action server.')
        return '', None

    def _send_gripper_goal(self, client, goal, action_name: str, label: str) -> Tuple[bool, str]:
        if self._emergency_stop_requested.is_set():
            return False, f'{label} skipped: emergency stop is active'
        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            self.gripper_action_timeout_sec,
            f'{label} goal',
        )
        if goal_handle is None:
            return False, f'{label} timed out sending goal to {action_name}'
        if not goal_handle.accepted:
            return False, f'{label} rejected by {action_name}'
        self._register_active_goal(f'{label} {action_name}', goal_handle)

        result_future = goal_handle.get_result_async()
        action_result = self._wait_for_future(
            result_future,
            self.gripper_action_timeout_sec,
            f'{label} result',
        )
        self._unregister_active_goal(f'{label} {action_name}', goal_handle)
        if action_result is None:
            return False, f'{label} timed out waiting for result from {action_name}'

        result = action_result.result
        if getattr(result, 'success', False):
            current_width = getattr(result, 'current_width', float('nan'))
            return True, f'{label} succeeded via {action_name}; current_width={current_width:.4f}m'

        error = getattr(result, 'error', '')
        if not error and label == 'close_gripper':
            error = (
                'grasp did not satisfy width/epsilon; increase '
                'gripper_grasp_epsilon_outer_m or set gripper_close_width_m '
                'near the object width'
            )
        return False, f'{label} failed via {action_name}: {error}'
