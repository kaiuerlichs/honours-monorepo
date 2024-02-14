#ifndef INCLUDE_HMP_UTILS_H_
#define INCLUDE_HMP_UTILS_H_

#include "mpi.h"

namespace hmputils {

// For a given C++ type, return the corresponding MPI type
template <typename T>
MPI_Datatype get_mpi_type();

}

#endif  // INCLUDE_HMP_UTILS_H_
