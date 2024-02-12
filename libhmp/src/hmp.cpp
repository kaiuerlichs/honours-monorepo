#include "hmp.h"

#include <cstddef>
#include <cstdio>
#include <memory>

#include "mpi.h"
#include "omp.h"

namespace hmp {

NodeInfo::NodeInfo() {
  load_rank();
  load_thread_count();
  load_processor_name();
}

void NodeInfo::load_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_rank = rank;
}

void NodeInfo::load_thread_count() { thread_count = omp_get_max_threads(); }

void NodeInfo::load_processor_name() {
  int name_length;
  MPI_Get_processor_name(processor_name, &name_length);
}

void NodeInfo::print_info() {
  std::printf("Node information for %s: Rank %i, Threads %i", processor_name,
              process_rank, thread_count);
}

bool NodeInfo::is_master() { return process_rank == 0; }

int NodeInfo::get_rank() { return process_rank; }

void WorldInfo::add_node(std::shared_ptr<NodeInfo> node_ptr) {
  nodes.push_back(node_ptr);
}

MPICluster::MPICluster() {
  MPI_Init(NULL, NULL);
  self = std::make_shared<NodeInfo>();
  self->print_info();

  if (self->is_master()) {
    world = std::make_unique<WorldInfo>();
    world->add_node(self);
  } else {
  
  }
}

MPICluster::~MPICluster() { MPI_Finalize(); }

} // namespace hmp
