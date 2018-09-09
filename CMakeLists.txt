cmake_minimum_required(VERSION 3.8)
project(IntegralImage)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(IntegralImage ${SOURCE_FILES})
target_link_libraries (IntegralImage ${OpenCV_LIBS})
