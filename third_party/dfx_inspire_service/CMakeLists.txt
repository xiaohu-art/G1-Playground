cmake_minimum_required(VERSION 3.10)
project(inspire_service)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
)

find_library(SYSTEM_FMT_LIBRARY
  NAMES fmt
  PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} /usr/lib
  NO_DEFAULT_PATH
  REQUIRED
)
message(STATUS "Using system fmt: ${SYSTEM_FMT_LIBRARY}")

link_libraries(unitree_sdk2 ddsc ddscxx rt pthread ${SYSTEM_FMT_LIBRARY})
link_libraries(libboost_program_options.a)

add_executable(inspire_g1 inspire_g1.cpp)
add_executable(inspire_h1 inspire_h1.cpp)

# example
add_executable(hand_example example/hand_example.cpp)
add_executable(bench_serial tools_bench_serial.cpp)

add_executable(probe_registers tools_probe_registers.cpp)

add_executable(probe_velocity tools_probe_velocity.cpp)

add_executable(monitor_velocity tools_monitor_velocity.cpp)
