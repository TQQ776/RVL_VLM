cmake_minimum_required(VERSION 3.8)
project(mcp)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CallTool.srv"
  "srv/ListTools.srv"
  "srv/MoveAxis.srv"
  "srv/ObjectName.srv"
  DEPENDENCIES std_msgs
)

ament_python_install_package(mcp_nodes)

install(PROGRAMS
  scripts/mcp_server
  scripts/mcp_omni_client
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
