cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM — deterministic floating-point configuration
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -mcpu=generic -mtune=generic -ffp-contract=off")
    message(STATUS "Using 64-bit ARM deterministic: -O2 -mcpu=generic -mtune=generic -ffp-contract=off")
  else()
    # 32-bit ARM
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -mcpu=generic -mtune=generic -mfpu=neon -ffp-contract=off")
    message(STATUS "Using 32-bit ARM deterministic: -O2 -mcpu=generic -mtune=generic -mfpu=neon -ffp-contract=off")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) — deterministic floating-point configuration
  # -O2: disable -O3 auto-vectorization / loop peeling / unrolling
  # -march=x86-64: use baseline x86-64 ISA, no CPU-specific SIMD variants
  # -ffp-contract=off: disable FMA contraction
  # -frounding-math: prevent optimizations that assume default rounding mode
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -march=x86-64 -mtune=generic -ffp-contract=off -frounding-math")
  message(STATUS "Using x86 deterministic: -O2 -march=x86-64 -mtune=generic -ffp-contract=off -frounding-math")
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU core count for potential multithreading optimization
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

# Set the number of cores for multithreading
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

# Find catkin and required dependencies
find_package(catkin REQUIRED COMPONENTS
  gnss_comm
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

add_message_files(
  FILES
  GnssStatus.msg
  RtkBackendStatus.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(GTSAM REQUIRED)
if(NOT GTSAM_INCLUDE_DIRS)
  set(GTSAM_INCLUDE_DIRS ${GTSAM_INCLUDE_DIR})
endif()
if(NOT GTSAM_LIBRARIES)
  set(GTSAM_LIBRARIES gtsam)
endif()

set(GTSAM_UNSTABLE_ROOT "" CACHE PATH
    "Prefix containing gtsam_unstable headers and library")
find_path(GTSAM_UNSTABLE_INCLUDE_DIR
  NAMES gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h
  HINTS
    ${GTSAM_UNSTABLE_ROOT}
    $ENV{HOME}/.local/gtsam-unstable-root/usr/local
    /usr/local
  PATH_SUFFIXES include
)
find_library(GTSAM_UNSTABLE_LIBRARY
  NAMES gtsam_unstable
  HINTS
    ${GTSAM_UNSTABLE_ROOT}
    $ENV{HOME}/.local/gtsam-unstable-root/usr/local
    /usr/local
  PATH_SUFFIXES lib
)
if(NOT GTSAM_UNSTABLE_INCLUDE_DIR OR NOT GTSAM_UNSTABLE_LIBRARY)
  message(FATAL_ERROR
    "IncrementalFixedLagSmoother requires the GTSAM unstable component. "
    "Missing gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h or "
    "libgtsam_unstable. Rebuild GTSAM with GTSAM_BUILD_UNSTABLE=ON and set "
    "GTSAM_UNSTABLE_ROOT to its install prefix.")
endif()
message(STATUS "GTSAM unstable include: ${GTSAM_UNSTABLE_INCLUDE_DIR}")
message(STATUS "GTSAM unstable library: ${GTSAM_UNSTABLE_LIBRARY}")

set(Sophus_LIBRARIES libSophus.so)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS gnss_comm geometry_msgs nav_msgs sensor_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus GTSAM
)

# Include directories for dependencies
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${GTSAM_UNSTABLE_INCLUDE_DIR}
  include
)

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(uwb src/uwb_manager.cpp)
add_library(gnss src/gnss_manager.cpp)
add_library(gnss_adapter src/gnss_adapter.cpp)
add_library(rtk_fixed_lag_backend src/rtk_fixed_lag_backend.cpp)
add_library(laser_mapping src/LIVMapper.cpp)

# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)
add_executable(gnss_manager_self_test src/gnss_manager_self_test.cpp)
add_executable(uwb_manager_self_test src/uwb_manager_self_test.cpp)
add_executable(gnss_adapter_node src/gnss_adapter_node.cpp)
add_executable(gnss_adapter_self_test src/gnss_adapter_self_test.cpp)
add_executable(rtk_fixed_lag_backend_node src/rtk_fixed_lag_backend_node.cpp)
add_executable(rtk_fixed_lag_backend_self_test src/rtk_fixed_lag_backend_self_test.cpp)
add_executable(voxel_filter_self_test src/voxel_filter_self_test.cpp)

add_dependencies(gnss_adapter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(gnss_adapter_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(gnss_adapter_self_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rtk_fixed_lag_backend ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rtk_fixed_lag_backend_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rtk_fixed_lag_backend_self_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(gnss_adapter ${catkin_LIBRARIES})

# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  gnss
  uwb
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(gnss_manager_self_test
  gnss
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(uwb_manager_self_test
  uwb
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(gnss_adapter_node
  gnss_adapter
  ${catkin_LIBRARIES}
)

target_link_libraries(gnss_adapter_self_test
  gnss_adapter
  ${catkin_LIBRARIES}
)

target_link_libraries(rtk_fixed_lag_backend
  ${catkin_LIBRARIES}
  gtsam
  ${GTSAM_UNSTABLE_LIBRARY}
)

target_link_libraries(rtk_fixed_lag_backend_node
  rtk_fixed_lag_backend
  ${catkin_LIBRARIES}
)

target_link_libraries(rtk_fixed_lag_backend_self_test
  rtk_fixed_lag_backend
  ${catkin_LIBRARIES}
  gtsam
  ${GTSAM_UNSTABLE_LIBRARY}
)

target_link_libraries(voxel_filter_self_test
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gnss ${OpenMP_CXX_FLAGS})
  target_link_libraries(gnss_manager_self_test ${OpenMP_CXX_FLAGS})
  target_link_libraries(uwb_manager_self_test ${OpenMP_CXX_FLAGS})
endif()

# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

# Link libusb-1.0 (required by PCL IO)
target_link_libraries(fastlivo_mapping usb-1.0)
