# Set minimum versions for C++ and CMake
cmake_minimum_required(VERSION 3.20...3.24)
set(CMAKE_CXX_STANDARD 20)

# Export compile commands for LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Define project
project(HybridMultiprocessingPatterns)
include_directories(
        ${PROJECT_SOURCE_DIR}/include 
        ${PROJECT_SOURCE_DIR}/include/hmp
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/hmp
)


#Load dependencies
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND) # On M2 Apple chips, CMake cannot find OpenMP
  include_directories(/opt/homebrew/opt/libomp/include)
  link_directories(/opt/homebrew/opt/libomp/lib)
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


# TARGET: debughmp (Debug executable)
set(DEBUG_FILES
  src/debughmp.cpp
)

add_executable(debughmp ${DEBUG_FILES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(debughmp OpenMP::OpenMP_CXX)
else()
    target_link_libraries(debughmp "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
target_link_libraries(debughmp ${MPI_LIBRARIES})

