#include "mpi_cluster.h"

#include <cstddef>

#include "mpi.h"
#include "omp.h"

namespace hmp {

void NodeInfo::load_node_info() {
  load_rank();
  load_thread_count();
  load_processor_name();
}

void NodeInfo::load_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_rank = rank;
}

void NodeInfo::load_thread_count() {
  thread_count = omp_get_max_threads();
}

void NodeInfo::load_processor_name() {
  int name_length;
  MPI_Get_processor_name(processor_name, &name_length);
}

MPICluster::MPICluster() {
  MPI_Init(NULL, NULL);
  self.load_node_info();
}

MPICluster::~MPICluster() {
  MPI_Finalize();
}

} // namespace hmp
