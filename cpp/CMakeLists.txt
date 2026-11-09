cmake_minimum_required(VERSION 3.16)
project(outdoor_perception CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # 测耗时一定用 Release
endif()

find_package(PCL REQUIRED COMPONENTS common io filters search kdtree segmentation sample_consensus)

add_executable(preprocess src/preprocess.cpp)
target_include_directories(preprocess PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(preprocess PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(preprocess PRIVATE ${PCL_LIBRARIES})
