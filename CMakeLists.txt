#cmake_minimum_required(VERSION 3.23)
#project(Gauss)
#
#set(CMAKE_CXX_STANDARD 14)
#
#add_executable(Gauss main.cpp)


cmake_minimum_required(VERSION 3.19)
project(Gauss)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(Gauss main7.cpp)
target_link_libraries(Gauss PUBLIC MPI::MPI_CXX)