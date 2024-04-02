#ifndef HMP_MPI_TYPE_TRAITS_H_
#define HMP_MPI_TYPE_TRAITS_H_

#include <type_traits>

#include "mpi.h"

namespace hmputils {

template<typename C_TYPE>
struct is_mpi_primitive : std::false_type {};

template<typename C_TYPE>
struct mpi_type_of;

#define C_MPI_TYPE_MAPPING(C_TYPE, MPI_TYPE)      \
template<>                                        \
struct mpi_type_of<C_TYPE> {                      \
    static MPI_Datatype value() {                 \
        return MPI_TYPE;                          \
    }                                             \
};                                                \
template<>                                        \
struct is_mpi_primitive<C_TYPE> : std::true_type {} 

C_MPI_TYPE_MAPPING(int, MPI_INT);
C_MPI_TYPE_MAPPING(float, MPI_FLOAT);
C_MPI_TYPE_MAPPING(double, MPI_DOUBLE);
C_MPI_TYPE_MAPPING(char, MPI_CHAR);
C_MPI_TYPE_MAPPING(short, MPI_SHORT);
C_MPI_TYPE_MAPPING(long, MPI_LONG);
C_MPI_TYPE_MAPPING(unsigned char, MPI_UNSIGNED_CHAR);
C_MPI_TYPE_MAPPING(unsigned short, MPI_UNSIGNED_SHORT);
C_MPI_TYPE_MAPPING(unsigned, MPI_UNSIGNED);
C_MPI_TYPE_MAPPING(unsigned long, MPI_UNSIGNED_LONG);
C_MPI_TYPE_MAPPING(long long int, MPI_LONG_LONG_INT);
C_MPI_TYPE_MAPPING(unsigned long long, MPI_UNSIGNED_LONG_LONG);
C_MPI_TYPE_MAPPING(signed char, MPI_SIGNED_CHAR);
C_MPI_TYPE_MAPPING(std::byte, MPI_BYTE);
C_MPI_TYPE_MAPPING(bool, MPI_C_BOOL); 

#undef C_MPI_TYPE_MAPPING

} // namespace hmp

#endif // HMP_MPI_TYPE_TRAITS_H_
