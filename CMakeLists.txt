cmake_minimum_required(VERSION 3.26)
project(sst)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g")


add_executable(sst main.cpp)
