#ifndef HMP_DISTRIBUTION_UTIL_H_
#define HMP_DISTRIBUTION_UTIL_H_

#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "hmp.h"

namespace hmp {

enum class Distribution {
  CORE_COUNT,
  CORE_FREQUENCY,
};

inline std::vector<int> distribution_with_core_count(int total_items, std::shared_ptr<MPICluster> cluster) {
  std::vector<int> distribution;
  distribution.resize(cluster->get_node_count());

  int total_cores = cluster->get_total_core_count();
  std::vector<int> cores_per_node = cluster->get_cores_per_node();

  int items_per_core = (int) total_items / total_cores;
  int remaining_items = total_items % total_cores;

  for (int rank = 0; rank < distribution.size(); ++rank) {
    int cores = cores_per_node[rank];
    int extra_items = remaining_items < cores ? remaining_items : cores;
    distribution[rank] = items_per_core * cores + extra_items;
    remaining_items -= extra_items;
  }

  return distribution;
}

inline std::vector<int> distribution_with_core_count_frequency(int total_items, std::shared_ptr<MPICluster> cluster) {
  std::vector<int> distribution;
  distribution.resize(cluster->get_node_count());

  std::vector<int> cores_per_node = cluster->get_cores_per_node();
  std::vector<int> frequency_per_node = cluster->get_frequency_per_node();

  std::vector<int> power_per_node;
  power_per_node.resize(cluster->get_node_count());
  int cluster_power = 0;

  for (int rank = 0; rank < distribution.size(); ++rank) {
    power_per_node[rank] = cores_per_node[rank] * frequency_per_node[rank];
    cluster_power += power_per_node[rank];
  }

  int allocated_items = 0;

  for (int rank = 0; rank < distribution.size(); ++rank) {
    float power_ratio = static_cast<float>(power_per_node[rank]) / cluster_power;
    distribution[rank] = (int) total_items * power_ratio;
    allocated_items += distribution[rank];
  }

  int remaining_items = total_items - allocated_items;

  for(int rank = 0; remaining_items > 0 && rank < distribution.size(); ++rank){
    distribution[rank] += 1;
    --remaining_items;
  }

  return distribution;
}

inline std::vector<int> distribution_by_type(int total_items, Distribution type, std::shared_ptr<MPICluster> cluster) {
  switch (type) {
    case Distribution::CORE_COUNT:
      return distribution_with_core_count(total_items, cluster);
      break;
    case Distribution::CORE_FREQUENCY:
      return distribution_with_core_count_frequency(total_items, cluster);
      break;
    default:
      throw std::invalid_argument("Selected distribution type is not implemented!");
      break;
  }
}

} // namespace hmp

#endif // HMP_DISTRIBUTION_UTIL_H_
