cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_CXX_FLAGS -std=c++11"")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include/)

add_library(lsd SHARED ${PROJECT_SOURCE_DIR}/src/lsd.cpp)

