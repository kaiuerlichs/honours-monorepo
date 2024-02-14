#include "hmp.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

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

int NodeInfo::get_thread_count() { return thread_count; }

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
  thread_count += node_ptr->get_thread_count();
  nodes.push_back(node_ptr);
}

int WorldInfo::get_node_count() { return node_count; }

void WorldInfo::print_info() {
  for (auto node_info : nodes) {
    node_info->print_info();
  }
}

int WorldInfo::get_thread_count() { return thread_count; }

std::vector<std::pair<int, int>> WorldInfo::get_thread_distribution() {
  std::vector<std::pair<int, int>> distribution;
  for (auto node_info : nodes) {
    distribution.push_back(
        {node_info->get_rank(), node_info->get_thread_count()});
  }
  return distribution;
}

MPICluster::MPICluster() {
  MPI_Init(NULL, NULL);
  self = std::make_shared<NodeInfo>();
  world = std::make_shared<WorldInfo>();

  MPI_Datatype mpi_node_info_type = self->get_mpi_type();

  if (self->is_master()) {
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

bool MPICluster::on_master() { return self->is_master(); }

int MPICluster::get_node_count() { return world->get_node_count(); }

int MPICluster::get_total_thread_count() { return world->get_thread_count(); }

int MPICluster::get_local_thread_count() { return self->get_thread_count(); }

int MPICluster::get_rank() { return self->get_rank(); }

std::vector<std::pair<int, int>> MPICluster::get_thread_distribution() {
  return world->get_thread_distribution();
}

MPICluster::~MPICluster() { MPI_Finalize(); }

} // namespace hmp
