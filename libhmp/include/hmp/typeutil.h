#ifndef HMP_TYPEUTIL_H_
#define HMP_TYPEUTIL_H_

#include "mpi.h"
#include <cstddef> // For std::byte

namespace hmputils {

// Returns the corresponding MPI type for a given C++ type
template <typename T>
MPI_Datatype get_mpi_type();

//IMPLEMENTATION

template<>
inline MPI_Datatype get_mpi_type<int>() {
    return MPI_INT;
}

template<>
inline MPI_Datatype get_mpi_type<float>() {
    return MPI_FLOAT;
}

template<>
inline MPI_Datatype get_mpi_type<double>() {
    return MPI_DOUBLE;
}

template<>
inline MPI_Datatype get_mpi_type<char>() {
    return MPI_CHAR;
}

template<>
inline MPI_Datatype get_mpi_type<short>() {
    return MPI_SHORT;
}

template<>
inline MPI_Datatype get_mpi_type<long>() {
    return MPI_LONG;
}

template<>
inline MPI_Datatype get_mpi_type<unsigned char>() {
    return MPI_UNSIGNED_CHAR;
}

template<>
inline MPI_Datatype get_mpi_type<unsigned short>() {
    return MPI_UNSIGNED_SHORT;
}

template<>
inline MPI_Datatype get_mpi_type<unsigned>() {
    return MPI_UNSIGNED;
}

template<>
inline MPI_Datatype get_mpi_type<unsigned long>() {
    return MPI_UNSIGNED_LONG;
}

template<>
inline MPI_Datatype get_mpi_type<long long int>() {
    return MPI_LONG_LONG_INT;
}

template<>
inline MPI_Datatype get_mpi_type<unsigned long long>() {
    return MPI_UNSIGNED_LONG_LONG;
}

template<>
inline MPI_Datatype get_mpi_type<signed char>() {
    return MPI_SIGNED_CHAR;
}

template<>
inline MPI_Datatype get_mpi_type<std::byte>() {
    return MPI_BYTE;
}

template<>
inline MPI_Datatype get_mpi_type<bool>() {
    return MPI_C_BOOL;
}

} // namespace hmputils

#endif  // HMP_TYPEUTIL_H_
