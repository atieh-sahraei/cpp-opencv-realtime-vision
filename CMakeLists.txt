cmake_minimum_required(VERSION 3.16)
project(cpp_opencv_realtime_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vision_demo src/main.cpp)

target_link_libraries(vision_demo ${OpenCV_LIBS})
target_include_directories(vision_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
