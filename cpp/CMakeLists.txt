cmake_minimum_required(VERSION 3.10)
project(color_calibrate_cpp)

add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# CUDA
add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

# OpenCV
#set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#include_directories(SYSTEM "/usr/bin/")

# sources
FILE(GLOB SRC "*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

add_executable(main ${SRC})
target_link_libraries (main ${OpenCV_LIBS} ${EXTRA_LIBS})