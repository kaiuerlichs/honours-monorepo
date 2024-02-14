#include "hmp/utils.h"

#include <cstddef>

namespace hmputils {

template<>
MPI_Datatype get_mpi_type<int>() {
    return MPI_INT;
}

template<>
MPI_Datatype get_mpi_type<float>() {
    return MPI_FLOAT;
}

template<>
MPI_Datatype get_mpi_type<double>() {
    return MPI_DOUBLE;
}

template<>
MPI_Datatype get_mpi_type<char>() {
    return MPI_CHAR;
}

template<>
MPI_Datatype get_mpi_type<short>() {
    return MPI_SHORT;
}

template<>
MPI_Datatype get_mpi_type<long>() {
    return MPI_LONG;
}

template<>
MPI_Datatype get_mpi_type<unsigned char>() {
    return MPI_UNSIGNED_CHAR;
}

template<>
MPI_Datatype get_mpi_type<unsigned short>() {
    return MPI_UNSIGNED_SHORT;
}

template<>
MPI_Datatype get_mpi_type<unsigned>() {
    return MPI_UNSIGNED;
}

template<>
MPI_Datatype get_mpi_type<unsigned long>() {
    return MPI_UNSIGNED_LONG;
}

template<>
MPI_Datatype get_mpi_type<long long int>() {
    return MPI_LONG_LONG_INT;
}

template<>
MPI_Datatype get_mpi_type<unsigned long long>() {
    return MPI_UNSIGNED_LONG_LONG;
}

template<>
MPI_Datatype get_mpi_type<signed char>() {
    return MPI_SIGNED_CHAR;
}

template<>
MPI_Datatype get_mpi_type<std::byte>() {
    return MPI_BYTE;
}

template<>
MPI_Datatype get_mpi_type<bool>() {
    return MPI_C_BOOL;
}

}
