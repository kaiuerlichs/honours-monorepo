#ifndef HMP_MAP_H_
#define HMP_MAP_H_

#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "hmp/core.h"
#include "hmp/mpi_type_traits.h"
#include "hmp/distribution_util.h"

namespace hmp {

// Processes a 1-dimensional dataset using the Map pattern
// Example:
//  std::unique_ptr<Map<int, int>> parallelMap = std::make_unique<Map<int,
//  int>>(cluster_ptr); std::vector<int> results = parallelMap->execute(inputs,
//  map_function_ptr);
template <typename IN_TYPE, typename OUT_TYPE> class Map {
private:
  std::shared_ptr<MPICluster> cluster;
  std::vector<IN_TYPE> local_data;
  std::vector<OUT_TYPE> local_return_data;
  std::vector<OUT_TYPE> return_data;

  Distribution distribution_type;

  std::vector<int> items_per_node;
  std::vector<int> displacements_per_node;

  MPI_Datatype mpi_in_type = MPI_DATATYPE_NULL;
  MPI_Datatype mpi_out_type = MPI_DATATYPE_NULL;

  void prepare_data(std::vector<IN_TYPE> &data);
  void run_map_function(std::function<OUT_TYPE(IN_TYPE)> map_function);
  void gather_data();
  void load_mpi_types();

public:
  Map(std::shared_ptr<MPICluster> cluster_ptr, Distribution distribution);
  ~Map(){};

  std::vector<OUT_TYPE> execute(std::vector<IN_TYPE> &data,
                                std::function<OUT_TYPE(IN_TYPE)> map_function);
  void set_mpi_in_type(MPI_Datatype in_type);
  void set_mpi_out_type(MPI_Datatype out_type);
};

// IMPLEMENTATION

template <typename IN_TYPE, typename OUT_TYPE>
Map<IN_TYPE, OUT_TYPE>::Map(std::shared_ptr<MPICluster> cluster_ptr, Distribution distribution) {
  cluster = cluster_ptr;

  if (cluster->is_linux()) {
    distribution_type = distribution;
  } else {
    printf("Defaulting to CORE_COUNT distribution due to operating system constraints\n");
    distribution_type = Distribution::CORE_COUNT;
  }
  load_mpi_types();
}

template <typename IN_TYPE, typename OUT_TYPE>
std::vector<OUT_TYPE>
Map<IN_TYPE, OUT_TYPE>::execute(std::vector<IN_TYPE> &data,
                                std::function<OUT_TYPE(IN_TYPE)> map_function) {
  bool in_type_defined = mpi_in_type != MPI_DATATYPE_NULL;
  bool out_type_defined = mpi_out_type != MPI_DATATYPE_NULL;

  if (!in_type_defined || !out_type_defined) {
    std::string types;
    if (!in_type_defined)
      types += "IN_TYPE; ";
    if (!out_type_defined)
      types += "OUT_TYPE; ";

    throw std::invalid_argument(
        "Some template parameters have no defined MPI type. Please provide an "
        "MPI type using setter functions for: " +
        types);
  }

  prepare_data(data);
  run_map_function(map_function);
  gather_data();

  return return_data;
}

// Creates buffers for local and master data, and calculates items
// and displacements per node
template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::prepare_data(std::vector<IN_TYPE> &data) {
  items_per_node.resize(cluster->get_node_count());
  
  if (cluster->on_master()) {
    int total_items = data.size();
    items_per_node = distribute_items(total_items, distribution_type, cluster);
    return_data.resize(total_items);
  }

  // Broadcast and creates local buffers
  MPI_Bcast(items_per_node.data(), items_per_node.size(), MPI_INT, 0,
            MPI_COMM_WORLD);
  local_data.resize(items_per_node[cluster->get_rank()]);
  local_return_data.resize(items_per_node[cluster->get_rank()]);

  // TODO: Is this more efficient to calculate on master and bcast?
  // Calculate displacement of blocks per node
  displacements_per_node.resize(items_per_node.size());
  int cumulative_sum = 0;
  for (int i = 0; i < items_per_node.size(); ++i) {
    displacements_per_node[i] = cumulative_sum;
    cumulative_sum += items_per_node[i];
  }

  // Distribute data over nodes into local buffers
  MPI_Scatterv(data.data(), items_per_node.data(),
               displacements_per_node.data(), mpi_in_type, local_data.data(),
               items_per_node[cluster->get_rank()], mpi_in_type, 0,
               MPI_COMM_WORLD);
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::run_map_function(
    std::function<OUT_TYPE(IN_TYPE)> map_function) {
  int thread_count = cluster->get_local_core_count();

  #pragma omp parallel for num_threads(thread_count)
  for (int i = 0; i < local_data.size(); ++i) {
    local_return_data[i] = map_function(local_data[i]);
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::gather_data() {
  // Gather data to master buffer
  MPI_Gatherv(local_return_data.data(), items_per_node[cluster->get_rank()],
              mpi_out_type, return_data.data(), items_per_node.data(),
              displacements_per_node.data(), mpi_out_type, 0, MPI_COMM_WORLD);
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::load_mpi_types() {
  if constexpr (hmputils::is_mpi_primitive<IN_TYPE>::value) {
    mpi_in_type = hmputils::mpi_type_of<IN_TYPE>::value();
  }
  if constexpr (hmputils::is_mpi_primitive<OUT_TYPE>::value) {
    mpi_out_type = hmputils::mpi_type_of<OUT_TYPE>::value();
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::set_mpi_in_type(MPI_Datatype in_type) {
  mpi_in_type = in_type;
  MPI_Type_commit(&mpi_in_type);
}

template <typename IN_TYPE, typename OUT_TYPE>
void Map<IN_TYPE, OUT_TYPE>::set_mpi_out_type(MPI_Datatype out_type) {
  mpi_out_type = out_type;
  MPI_Type_commit(&mpi_out_type);
}

} // namespace hmp

#endif // HMP_MAP_H_
