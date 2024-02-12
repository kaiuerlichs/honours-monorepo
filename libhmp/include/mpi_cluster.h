#ifndef INCLUDE_MPI_CLUSTER_H_
#define INCLUDE_MPI_CLUSTER_H_

#include <mpi.h>

#include <vector>

namespace hmp {
struct NodeInfo {
  int process_rank;
  int thread_count;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
 
  void load_node_info();
  void load_rank();
  void load_thread_count();
  void load_processor_name();
};

class MPICluster {
private:
  NodeInfo self;
  std::vector<NodeInfo> worldInfo;

public:
  // Constructor/destructor
  MPICluster();
  ~MPICluster();
};
} // namespace hmp

#endif // INCLUDE_MPI_CLUSTER_H_
