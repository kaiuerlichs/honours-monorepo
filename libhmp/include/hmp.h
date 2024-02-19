#ifndef INCLUDE_HMP_H_
#define INCLUDE_HMP_H_

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

#include "mpi.h"
#include "omp.h"

namespace hmp {

// Stores information about a given node in an MPI cluster
class NodeInfo {
private:
  int process_rank;
  int thread_count;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  void load_rank();
  void load_thread_count();
  void load_processor_name();

public:
  NodeInfo();
  ~NodeInfo(){};

  void print_info();
  bool is_master();
  int get_rank();
  int get_thread_count();

  // Returns a custom MPI type used for sending node info objects between nodes
  MPI_Datatype get_mpi_type();
};

// Stores information about the MPI cluster global context
class WorldInfo {
private:
  int node_count;
  int thread_count;
  std::vector<std::shared_ptr<NodeInfo>> nodes;

public:
  WorldInfo();
  ~WorldInfo(){};

  void add_node(std::shared_ptr<NodeInfo>);
  int get_node_count();
  int get_thread_count();

  // Returns a vector of integer pairs
  //  First: The node rank
  //  Second: The number of threads on that node
  std::vector<std::pair<int, int>> get_thread_distribution();
  
  void print_info();
};

// Top level wrapper class for the MPI functionality
class MPICluster {
private:
  std::shared_ptr<NodeInfo> self;
  std::shared_ptr<WorldInfo> world;

  void send_node_info();
  void receive_node_info();

public:
  // Constructor/destructor
  MPICluster();
  ~MPICluster();

  bool on_master();
  int get_node_count();
  int get_total_thread_count();
  int get_local_thread_count();
  std::vector<std::pair<int, int>> get_thread_distribution();
  int get_rank();
};

// IMPLEMENTATION

inline NodeInfo::NodeInfo() {
  load_rank();
  load_thread_count();
  load_processor_name();
}

inline void NodeInfo::load_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_rank = rank;
}

inline void NodeInfo::load_thread_count() { thread_count = omp_get_max_threads(); }

inline void NodeInfo::load_processor_name() {
  int name_length;
  MPI_Get_processor_name(processor_name, &name_length);
}

inline void NodeInfo::print_info() {
  std::printf("Node information for %s: Rank %i, Threads %i\n", processor_name,
              process_rank, thread_count);
}

inline bool NodeInfo::is_master() { return process_rank == 0; }

inline int NodeInfo::get_rank() { return process_rank; }

inline int NodeInfo::get_thread_count() { return thread_count; }

inline MPI_Datatype NodeInfo::get_mpi_type() {
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

inline WorldInfo::WorldInfo() { MPI_Comm_size(MPI_COMM_WORLD, &node_count); }

inline void WorldInfo::add_node(std::shared_ptr<NodeInfo> node_ptr) {
  thread_count += node_ptr->get_thread_count();
  nodes.push_back(node_ptr);
}

inline int WorldInfo::get_node_count() { return node_count; }

inline void WorldInfo::print_info() {
  for (auto node_info : nodes) {
    node_info->print_info();
  }
}

inline int WorldInfo::get_thread_count() { return thread_count; }

inline std::vector<std::pair<int, int>> WorldInfo::get_thread_distribution() {
  std::vector<std::pair<int, int>> distribution;
  for (auto node_info : nodes) {
    distribution.push_back(
        {node_info->get_rank(), node_info->get_thread_count()});
  }
  return distribution;
}

inline MPICluster::MPICluster() {
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

inline bool MPICluster::on_master() { return self->is_master(); }

inline int MPICluster::get_node_count() { return world->get_node_count(); }

inline int MPICluster::get_total_thread_count() { return world->get_thread_count(); }

inline int MPICluster::get_local_thread_count() { return self->get_thread_count(); }

inline int MPICluster::get_rank() { return self->get_rank(); }

inline std::vector<std::pair<int, int>> MPICluster::get_thread_distribution() {
  return world->get_thread_distribution();
}

inline MPICluster::~MPICluster() { MPI_Finalize(); }

} // namespace hmp

#endif // INCLUDE_HMP_H_
