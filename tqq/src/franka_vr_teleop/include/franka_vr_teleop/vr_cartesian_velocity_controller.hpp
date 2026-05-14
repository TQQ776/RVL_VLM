#pragma once

#include <array>
#include <memory>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_cartesian_velocity_interface.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace franka_vr_teleop {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class VRCartesianVelocityController : public controller_interface::ControllerInterface {
 public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
      const rclcpp::Time& time,
      const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static double clamp(double value, double limit);

  std::unique_ptr<franka_semantic_components::FrankaCartesianVelocityInterface>
      franka_cartesian_velocity_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_sub_;

  std::array<double, 6> latest_command_{};
  std::array<double, 6> filtered_command_{};
  rclcpp::Time latest_command_time_{0, 0, RCL_ROS_TIME};

  std::string twist_topic_;
  double command_timeout_sec_{0.20};
  double max_linear_velocity_mps_{0.08};
  double max_angular_velocity_radps_{0.35};
  double max_linear_acceleration_mps2_{0.08};
  double max_angular_acceleration_radps2_{0.35};
  double lowpass_alpha_{0.20};
  bool enabled_{false};
};

}  // namespace franka_vr_teleop
