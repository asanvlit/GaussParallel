cmake_minimum_required(VERSION 3.19)
project(Gauss)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(Gauss main.cpp)
target_link_libraries(Gauss PUBLIC MPI::MPI_CXX)