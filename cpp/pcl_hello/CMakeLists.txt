cmake_minimum_required(VERSION 3.16)
project(pcl_hello CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)   # 测耗时一定要用 Release，Debug 会慢很多倍

find_package(PCL REQUIRED COMPONENTS common filters)

add_executable(pcl_hello main.cpp)
target_include_directories(pcl_hello PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_hello PRIVATE ${PCL_LIBRARIES})
