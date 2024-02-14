#ifndef INCLUDE_HMP_H_
#define INCLUDE_HMP_H_

#include <memory>
#include <vector>

#include "mpi.h"
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
} // namespace hmp

#endif // INCLUDE_HMP_H_
