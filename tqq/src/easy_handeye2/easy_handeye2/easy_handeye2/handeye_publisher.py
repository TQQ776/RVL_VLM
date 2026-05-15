#!/usr/bin/env python

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time
from rclpy.node import ParameterType, ParameterDescriptor
import tf2_ros
import geometry_msgs.msg
from easy_handeye2.handeye_calibration import load_calibration


class HandeyePublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('handeye_publisher')

        self.declare_parameter('name', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('publish_parent_frame', '')
        self.declare_parameter('auto_publish_optical_parent', True)
        self.declare_parameter('parent_lookup_timeout_sec', 3.0)
        name = self.get_parameter('name').get_parameter_value().string_value
        publish_parent_frame = self.get_parameter('publish_parent_frame').get_parameter_value().string_value
        auto_publish_optical_parent = self.get_parameter('auto_publish_optical_parent').get_parameter_value().bool_value
        parent_lookup_timeout_sec = self.get_parameter('parent_lookup_timeout_sec').get_parameter_value().double_value

        self.get_logger().info(f'Loading the calibration with name {name}')

        self.calibration = load_calibration(name)
        parameters = self.calibration.parameters

        if parameters.calibration_type == 'eye_in_hand':
            orig = parameters.robot_effector_frame
        else:
            orig = parameters.robot_base_frame
        dest = parameters.tracking_base_frame
        transform = self.calibration.transform

        target_child = publish_parent_frame.strip()
        if not target_child and auto_publish_optical_parent and dest.endswith('_optical_frame'):
            target_child = dest[:-len('_optical_frame')] + '_frame'

        if target_child and target_child != dest:
            adjusted = self._transform_to_parent_frame(
                transform,
                target_child,
                dest,
                parent_lookup_timeout_sec,
            )
            if adjusted is not None:
                self.get_logger().info(
                    f'Publishing adjusted hand-eye transform {orig} -> {target_child}; '
                    f'camera driver keeps publishing {target_child} -> {dest}.'
                )
                dest = target_child
                transform = adjusted
            else:
                self.get_logger().warn(
                    f'Could not resolve {target_child} -> {dest}; publishing original '
                    f'hand-eye transform {orig} -> {dest}. This may conflict if another '
                    f'node also publishes {dest}.'
                )

        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.static_transformStamped = geometry_msgs.msg.TransformStamped()

        self.static_transformStamped.header.stamp = self.get_clock().now().to_msg()
        self.static_transformStamped.header.frame_id = orig
        self.static_transformStamped.child_frame_id = dest

        self.static_transformStamped.transform = transform

        self.broadcaster.sendTransform(self.static_transformStamped)

    def _transform_to_parent_frame(self, orig_to_child, parent_frame, child_frame, timeout_sec):
        buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buffer, self, spin_thread=True)
        del listener

        deadline = self.get_clock().now().nanoseconds / 1e9 + max(0.1, timeout_sec)
        parent_to_child = None
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < deadline:
            try:
                parent_to_child = buffer.lookup_transform(
                    parent_frame,
                    child_frame,
                    Time(),
                ).transform
                break
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)

        if parent_to_child is None:
            return None

        child_to_parent = self._invert_transform(parent_to_child)
        return self._compose_transforms(orig_to_child, child_to_parent)

    @classmethod
    def _compose_transforms(cls, first, second):
        result = geometry_msgs.msg.Transform()
        result.rotation = cls._normalize_quaternion(
            cls._quaternion_multiply(first.rotation, second.rotation)
        )
        rotated_second_translation = cls._rotate_vector(first.rotation, second.translation)
        result.translation.x = first.translation.x + rotated_second_translation.x
        result.translation.y = first.translation.y + rotated_second_translation.y
        result.translation.z = first.translation.z + rotated_second_translation.z
        return result

    @classmethod
    def _invert_transform(cls, transform):
        result = geometry_msgs.msg.Transform()
        result.rotation.x = -transform.rotation.x
        result.rotation.y = -transform.rotation.y
        result.rotation.z = -transform.rotation.z
        result.rotation.w = transform.rotation.w
        inverse_translation = geometry_msgs.msg.Vector3()
        inverse_translation.x = -transform.translation.x
        inverse_translation.y = -transform.translation.y
        inverse_translation.z = -transform.translation.z
        result.translation = cls._rotate_vector(result.rotation, inverse_translation)
        return result

    @staticmethod
    def _quaternion_multiply(a, b):
        result = geometry_msgs.msg.Quaternion()
        result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
        result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
        result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
        result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        return result

    @staticmethod
    def _normalize_quaternion(q):
        length = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
        if length == 0.0:
            q.w = 1.0
            return q
        q.x /= length
        q.y /= length
        q.z /= length
        q.w /= length
        return q

    @classmethod
    def _rotate_vector(cls, q, vector):
        vec_quat = geometry_msgs.msg.Quaternion()
        vec_quat.x = vector.x
        vec_quat.y = vector.y
        vec_quat.z = vector.z
        vec_quat.w = 0.0

        q_inv = geometry_msgs.msg.Quaternion()
        q_inv.x = -q.x
        q_inv.y = -q.y
        q_inv.z = -q.z
        q_inv.w = q.w

        rotated = cls._quaternion_multiply(
            cls._quaternion_multiply(q, vec_quat),
            q_inv,
        )
        result = geometry_msgs.msg.Vector3()
        result.x = rotated.x
        result.y = rotated.y
        result.z = rotated.z
        return result


def main(args=None):
    rclpy.init(args=args)

    handeye_publisher = HandeyePublisher()

    try:
        rclpy.spin(handeye_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        handeye_publisher.destroy_node()


if __name__ == '__main__':
    main()
