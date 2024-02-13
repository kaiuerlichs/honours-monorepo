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
  std::printf("Node information for %s: Rank %i, Threads %i\n", processor_name,
              process_rank, thread_count);
}

bool NodeInfo::is_master() { return process_rank == 0; }

int NodeInfo::get_rank() { return process_rank; }

MPI_Datatype NodeInfo::get_mpi_type() {
  int blocklengths[3] = {1, 1, MPI_MAX_PROCESSOR_NAME};

  MPI_Aint displacements[3];
  displacements[0] = offsetof(NodeInfo, process_rank);
  displacements[1] = offsetof(NodeInfo, thread_count);
  displacements[2] = offsetof(NodeInfo, processor_name);

  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_CHAR};

  MPI_Datatype mpi_node_info_type;
  MPI_Type_create_struct(3, blocklengths, displacements, types,
                         &mpi_node_info_type);
  MPI_Type_commit(&mpi_node_info_type);
  return mpi_node_info_type;
}

WorldInfo::WorldInfo() { MPI_Comm_size(MPI_COMM_WORLD, &node_count); }

void WorldInfo::add_node(std::shared_ptr<NodeInfo> node_ptr) {
  nodes.push_back(node_ptr);
}

int WorldInfo::get_node_count() { return node_count; }

void WorldInfo::print_info() {
  for (auto node_info : nodes) {
    node_info->print_info();
  }
}

MPICluster::MPICluster() {
  MPI_Init(NULL, NULL);
  self = std::make_shared<NodeInfo>();

  MPI_Datatype mpi_node_info_type = self->get_mpi_type();

  if (self->is_master()) {
    world = std::make_unique<WorldInfo>();
    world->add_node(self);
    for (int i = 1; i < world->get_node_count(); ++i) {
      NodeInfo node_info;
      MPI_Recv(&node_info, 1, mpi_node_info_type, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      world->add_node(std::make_shared<NodeInfo>(node_info));
    }
    world->print_info();
  } else {
    MPI_Send(self.get(), 1, mpi_node_info_type, 0, 0, MPI_COMM_WORLD);
  }
}

MPICluster::~MPICluster() { MPI_Finalize(); }

} // namespace hmp
