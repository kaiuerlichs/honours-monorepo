cmake_minimum_required(VERSION 3.24)
project(HybridMultiprocessingPatterns)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the C++ standard you are using (e.g., C++11, C++14, C++17)
set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
  src/mpi_cluster.cpp
)

add_library(hmp SHARED ${SOURCE_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hmp OpenMP::OpenMP_CXX)
else()
    # Manual setup for OpenMP using libomp installed via Homebrew
    include_directories(/opt/homebrew/opt/libomp/include)
    link_directories(/opt/homebrew/opt/libomp/lib)
    target_link_libraries(hmp "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

# Find OpenMPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(hmp ${MPI_LIBRARIES})

# If your MPI implementation requires specific compilation flags, apply them to the library
if(MPI_COMPILE_FLAGS)
  set_target_properties(hmp PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# If your MPI implementation requires specific linking flags, apply them to the library
if(MPI_LINK_FLAGS)
  set_target_properties(hmp PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

set(DEBUG_FILES
  src/main.cpp
)

add_executable(debughmp ${SOURCE_FILES} ${DEBUG_FILES})
target_link_libraries(debughmp ${MPI_LIBRARIES})
target_link_libraries(debughmp "/opt/homebrew/opt/libomp/lib/libomp.dylib")
