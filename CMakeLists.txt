cmake_minimum_required(VERSION 2.8.3)

project(PL_VO)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(OpenCV_DIR "/home/rain/workspace/library/opencv-3.2.0/install/share/OpenCV")

find_package(OpenCV 3 REQUIRED)
find_package(Glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

include_directories("/usr/include/eigen3/")

include_directories(
    include
    ${Opencv_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    BEFORE ${GLOG_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

include_directories(${PROJECT_SOURCE_DIR}/thirdparty/LSD/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/line_descriptor/include)

add_library(${PROJECT_NAME} SHARED
    src/Camera.cpp
    src/LineFeature.cpp
    src/Config.cpp
    src/Converter.cpp
    src/PointFeature.cpp
    src/Frame.cpp
    src/Tracking.cpp
    src/System.cpp
    src/Map.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/thirdparty/LSD/lib/liblsd.so
    ${PROJECT_SOURCE_DIR}/thirdparty/line_descriptor/lib/liblinedesc.so
)

add_executable(rgbd_tum Example/rgbd_tum.cpp)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

add_executable(test_lsd test/test_line_feature/test_lsd.cpp )
target_link_libraries(test_lsd ${PROJECT_NAME})

add_executable(test_opencv_lsd test/test_line_feature/test_opencv_lsd.cpp)
target_link_libraries(test_opencv_lsd ${PROJECT_NAME})

add_executable(test_line_match test/test_line_match/test_line_match.cpp)
target_link_libraries(test_line_match ${PROJECT_NAME})

add_executable(test_converter test/test_converter.cpp)
target_link_libraries(test_converter ${PROJECT_NAME})
