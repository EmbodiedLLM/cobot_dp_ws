cmake_minimum_required(VERSION 2.8.3)
project(dh_robotics_ag95_model)
find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY meshes DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
