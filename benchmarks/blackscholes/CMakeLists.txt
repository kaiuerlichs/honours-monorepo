# Set minimum versions for C++ and CMake
cmake_minimum_required(VERSION 3.20...3.24)
set(CMAKE_CXX_STANDARD 20)

# Export compile commands for LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Define project
project(HMP_Benchmark_Blackscholes LANGUAGES CXX)
include_directories(
        ${PROJECT_SOURCE_DIR}/../../libhmp/include 
        ${PROJECT_SOURCE_DIR}/../../libhmp/include/hmp
        ${PROJECT_SOURCE_DIR}/src
)


#Load dependencies
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND) # On M2 Apple chips, CMake cannot find OpenMP
  include_directories(/opt/homebrew/opt/libomp/include)
  link_directories(/opt/homebrew/opt/libomp/lib)
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


# TARGET: blackscholes_benchmark
set(BLACKSCHOLES_BENCHMARK_FILES
  src/blackscholes_benchmark.cpp
  src/parsec_blackscholes.h
)

add_executable(blackscholes_benchmark ${BLACKSCHOLES_BENCHMARK_FILES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(blackscholes_benchmark OpenMP::OpenMP_CXX)
else()
    target_link_libraries(blackscholes_benchmark "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
target_link_libraries(blackscholes_benchmark ${MPI_LIBRARIES})

