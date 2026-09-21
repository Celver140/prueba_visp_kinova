cmake_minimum_required(VERSION 3.8)
project(kinova_control)

# Buscar dependencias de ROS 2 / Python
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Instalar el módulo Python
ament_python_install_package(${PROJECT_NAME})

# Instalar el nodo ejecutable. 
install(PROGRAMS
  IBVS_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME IBVS_node
)

install(PROGRAMS
  PBVS_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME PBVS_node
)

# Instalar el launch file
install(DIRECTORY
  urdf
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()