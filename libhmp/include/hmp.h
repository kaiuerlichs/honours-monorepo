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
};

class MPICluster {
private:
  std::shared_ptr<NodeInfo> self;
  std::vector<NodeInfo> worldInfo;

public:
  // Constructor/destructor
  MPICluster();
  ~MPICluster();
};
} // namespace hmp

#endif // INCLUDE_MPI_CLUSTER_H_
