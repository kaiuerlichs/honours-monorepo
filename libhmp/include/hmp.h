#ifndef INCLUDE_MPI_CLUSTER_H_
#define INCLUDE_MPI_CLUSTER_H_

#include <memory>
#include <mpi.h>

#include <vector>

namespace hmp {

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
  ~NodeInfo() {};

  void print_info();
  bool is_master();
  int get_rank();
  MPI_Datatype get_mpi_type();
};

class WorldInfo {
private:
  int node_count;
  int thread_count;
  std::vector<std::shared_ptr<NodeInfo>> nodes;

public:
  WorldInfo();
  ~WorldInfo() {};

  void add_node(std::shared_ptr<NodeInfo>);
  int get_node_count();
  void print_info();
};

class MPICluster {
private:
  std::shared_ptr<NodeInfo> self;
  std::unique_ptr<WorldInfo> world;

  void send_node_info();
  void receive_node_info();

public:
  // Constructor/destructor
  MPICluster();
  ~MPICluster();
};
} // namespace hmp

#endif // INCLUDE_MPI_CLUSTER_H_
