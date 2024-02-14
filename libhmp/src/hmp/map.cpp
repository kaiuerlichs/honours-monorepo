#include "hmp/map.h"

#include <cstdio>
#include <memory>
#include <vector>

#include "hmp/utils.h"

#include "mpi.h"

namespace hmp {

template <typename IN_TYPE, typename OUT_TYPE>
Map<IN_TYPE, OUT_TYPE>::Map(std::shared_ptr<MPICluster> cluster_ptr) {
  cluster = cluster_ptr;
}

template <typename IN_TYPE, typename OUT_TYPE>
std::vector<OUT_TYPE>
Map<IN_TYPE, OUT_TYPE>::execute(std::vector<IN_TYPE> &data,
                                std::function<OUT_TYPE(IN_TYPE)> map_function) {
  prepare_data(data);
  run_map_function(map_function);
  gather_data();

  return return_data;
}

// Creates buffers for local and master data, and calculates elements
// and displacements per node
template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::prepare_data(std::vector<IN_TYPE> &data) {
  elements_per_node.resize(cluster->get_node_count());

  if (cluster->on_master()) {
    std::vector<std::pair<int, int>> thread_distribution =
        cluster->get_thread_distribution();
    int total_threads = cluster->get_total_thread_count();

    int data_elements = data.size();
    int elements_per_thread = (int)data_elements / total_threads;
    int remaining_elements = data_elements % total_threads;
    
    // Calculates block wise distribution per node
    for (auto node_threads : thread_distribution) {
      auto [rank, threads] = node_threads;
      int extra_elements =
          remaining_elements < threads ? remaining_elements : threads;
      elements_per_node[rank] = elements_per_thread * threads + extra_elements;
      remaining_elements -= extra_elements;
    }

    return_data.resize(data_elements);
  }

  // Broadcast and creates local buffers
  MPI_Bcast(elements_per_node.data(), elements_per_node.size(), MPI_INT, 0,
            MPI_COMM_WORLD);
  local_data.resize(elements_per_node[cluster->get_rank()]);
  local_return_data.resize(elements_per_node[cluster->get_rank()]);
  
  // TODO: Is this more efficient to calculate on master and bcast?
  // Calculate displacement of blocks per node
  displacements_per_node.resize(elements_per_node.size());
  int cumulative_sum = 0;
  for (int i = 0; i < elements_per_node.size(); ++i) {
    displacements_per_node[i] = cumulative_sum;
    cumulative_sum += elements_per_node[i];
  }
  
  // Distribute data over nodes into local buffers
  MPI_Scatterv(data.data(), elements_per_node.data(),
               displacements_per_node.data(), hmputils::get_mpi_type<IN_TYPE>(),
               local_data.data(), elements_per_node[cluster->get_rank()],
               hmputils::get_mpi_type<IN_TYPE>(), 0, MPI_COMM_WORLD);
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::run_map_function(
    std::function<OUT_TYPE(IN_TYPE)> map_function) {
  // TODO: Implement thread-level parallelism around map function
  //  Use pthreads or OMP?
  // Runs map function through a for-loop
  for (int i = 0; i < local_data.size(); ++i) {
    local_return_data[i] = map_function(local_data[i]);
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::gather_data() {
  // Gather data to master buffer
  MPI_Gatherv(local_return_data.data(), elements_per_node[cluster->get_rank()],
              hmputils::get_mpi_type<OUT_TYPE>(), return_data.data(),
              elements_per_node.data(), displacements_per_node.data(),
              hmputils::get_mpi_type<OUT_TYPE>(), 0, MPI_COMM_WORLD);
}

template class Map<int, int>;

} // namespace hmp
