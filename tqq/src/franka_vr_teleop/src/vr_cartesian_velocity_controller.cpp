#include <franka_vr_teleop/vr_cartesian_velocity_controller.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <pluginlib/class_list_macros.hpp>

namespace franka_vr_teleop {

controller_interface::InterfaceConfiguration
VRCartesianVelocityController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_velocity_->get_command_interface_names();
  return config;
}

controller_interface::InterfaceConfiguration
VRCartesianVelocityController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}

controller_interface::return_type VRCartesianVelocityController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  std::array<double, 6> command{};
  const bool fresh =
      enabled_ && latest_command_time_.nanoseconds() > 0 &&
      (time - latest_command_time_).seconds() <= command_timeout_sec_;

  if (fresh) {
    command = latest_command_;
  }

  for (size_t i = 0; i < filtered_command_.size(); ++i) {
    const double max_acceleration =
        i < 3 ? max_linear_acceleration_mps2_ : max_angular_acceleration_radps2_;
    const double max_delta = max_acceleration * std::max(0.0, period.seconds());
    const double lowpass_target =
        filtered_command_[i] + lowpass_alpha_ * (command[i] - filtered_command_[i]);
    filtered_command_[i] += clamp(lowpass_target - filtered_command_[i], max_delta);
  }

  Eigen::Vector3d linear(
      filtered_command_[0],
      filtered_command_[1],
      filtered_command_[2]);
  Eigen::Vector3d angular(
      filtered_command_[3],
      filtered_command_[4],
      filtered_command_[5]);

  if (franka_cartesian_velocity_->setCommand(linear, angular)) {
    return controller_interface::return_type::OK;
  }

  RCLCPP_FATAL(
      get_node()->get_logger(),
      "Failed to set Franka cartesian velocity command.");
  return controller_interface::return_type::ERROR;
}

CallbackReturn VRCartesianVelocityController::on_init() {
  franka_cartesian_velocity_ =
      std::make_unique<franka_semantic_components::FrankaCartesianVelocityInterface>(false);

  try {
    auto_declare<std::string>("twist_topic", "~/twist_cmd");
    auto_declare<double>("command_timeout_sec", 0.20);
    auto_declare<double>("max_linear_velocity_mps", 0.03);
    auto_declare<double>("max_angular_velocity_radps", 0.15);
    auto_declare<double>("max_linear_acceleration_mps2", 0.06);
    auto_declare<double>("max_angular_acceleration_radps2", 0.25);
    auto_declare<double>("lowpass_alpha", 0.05);
  } catch (const std::exception& exc) {
    fprintf(stderr, "Exception during controller init: %s\n", exc.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn VRCartesianVelocityController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  twist_topic_ = get_node()->get_parameter("twist_topic").as_string();
  command_timeout_sec_ = std::max(0.02, get_node()->get_parameter("command_timeout_sec").as_double());
  max_linear_velocity_mps_ =
      std::max(0.0, get_node()->get_parameter("max_linear_velocity_mps").as_double());
  max_angular_velocity_radps_ =
      std::max(0.0, get_node()->get_parameter("max_angular_velocity_radps").as_double());
  max_linear_acceleration_mps2_ =
      std::max(0.0, get_node()->get_parameter("max_linear_acceleration_mps2").as_double());
  max_angular_acceleration_radps2_ =
      std::max(0.0, get_node()->get_parameter("max_angular_acceleration_radps2").as_double());
  lowpass_alpha_ = std::clamp(get_node()->get_parameter("lowpass_alpha").as_double(), 0.01, 1.0);

  twist_sub_ = get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      twist_topic_,
      rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        latest_command_ = {
            clamp(msg->twist.linear.x, max_linear_velocity_mps_),
            clamp(msg->twist.linear.y, max_linear_velocity_mps_),
            clamp(msg->twist.linear.z, max_linear_velocity_mps_),
            clamp(msg->twist.angular.x, max_angular_velocity_radps_),
            clamp(msg->twist.angular.y, max_angular_velocity_radps_),
            clamp(msg->twist.angular.z, max_angular_velocity_radps_),
        };
        latest_command_time_ = get_node()->now();
      });

  RCLCPP_INFO(
      get_node()->get_logger(),
      "VR Cartesian velocity controller configured. twist_topic=%s, timeout=%.3fs, "
      "max_linear=%.3fm/s, max_angular=%.3frad/s, "
      "max_linear_accel=%.3fm/s^2, max_angular_accel=%.3frad/s^2, alpha=%.2f",
      twist_topic_.c_str(),
      command_timeout_sec_,
      max_linear_velocity_mps_,
      max_angular_velocity_radps_,
      max_linear_acceleration_mps2_,
      max_angular_acceleration_radps2_,
      lowpass_alpha_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn VRCartesianVelocityController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_velocity_->assign_loaned_command_interfaces(command_interfaces_);
  latest_command_.fill(0.0);
  filtered_command_.fill(0.0);
  latest_command_time_ = rclcpp::Time(0, 0, get_node()->get_clock()->get_clock_type());
  enabled_ = true;
  return CallbackReturn::SUCCESS;
}

CallbackReturn VRCartesianVelocityController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  enabled_ = false;
  latest_command_.fill(0.0);
  filtered_command_.fill(0.0);
  franka_cartesian_velocity_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

double VRCartesianVelocityController::clamp(double value, double limit) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::clamp(value, -limit, limit);
}

}  // namespace franka_vr_teleop

PLUGINLIB_EXPORT_CLASS(
    franka_vr_teleop::VRCartesianVelocityController,
    controller_interface::ControllerInterface)
