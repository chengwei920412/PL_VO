cmake_minimum_required(VERSION 2.8.3)

project(PL_VO)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(OpenCV_DIR "/home/rain/workspace/library/opencv-3.2.0/install/share/OpenCV")

find_package(OpenCV REQUIRED)
include_directories( ${Opencv_INCLUDE_DIRS} )

include_directories("/usr/include/eigen3")

include_directories(${PROJECT_SOURCE_DIR}/thirdparty/LSD/include)

add_library(${PROJECT_NAME} SHARED
    src/Camera.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/thirdparty/LSD/lib/liblsd.so
    ${OpenCV_LIBRARIES}

)

add_executable(test_lsd test/test_line_feature/test_lsd.cpp)
target_link_libraries(test_lsd ${PROJECT_NAME})