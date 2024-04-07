#ifndef HMP_DISTRIBUTION_UTIL_H_
#define HMP_DISTRIBUTION_UTIL_H_

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hmp.h"

namespace hmp {

enum class Distribution {
  CORE_COUNT,
  CORE_FREQUENCY,
};

inline std::vector<int>
distribute_items_core_count(int total_items,
                            std::shared_ptr<MPICluster> cluster) {
  std::vector<int> distribution;
  distribution.resize(cluster->get_node_count());

  int total_cores = cluster->get_total_core_count();
  std::vector<int> cores_per_node = cluster->get_cores_per_node();

  int items_per_core = (int)total_items / total_cores;
  int remaining_items = total_items % total_cores;

  for (int rank = 0; rank < distribution.size(); ++rank) {
    int cores = cores_per_node[rank];
    int extra_items = remaining_items < cores ? remaining_items : cores;
    distribution[rank] = items_per_core * cores + extra_items;
    remaining_items -= extra_items;
  }

  return distribution;
}

inline std::vector<int>
distribute_items_core_count_frequency(int total_items,
                                      std::shared_ptr<MPICluster> cluster) {
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
    float power_ratio =
        static_cast<float>(power_per_node[rank]) / cluster_power;
    distribution[rank] = (int)total_items * power_ratio;
    allocated_items += distribution[rank];
  }

  int remaining_items = total_items - allocated_items;

  for (int rank = 0; remaining_items > 0 && rank < distribution.size();
       ++rank) {
    distribution[rank] += 1;
    --remaining_items;
  }

  return distribution;
}

inline std::vector<int> distribute_items(int total_items, Distribution type,
                                         std::shared_ptr<MPICluster> cluster) {
  switch (type) {
  case Distribution::CORE_COUNT:
    return distribute_items_core_count(total_items, cluster);
    break;
  case Distribution::CORE_FREQUENCY:
    return distribute_items_core_count_frequency(total_items, cluster);
    break;
  default:
    throw std::invalid_argument(
        "Selected distribution type is not implemented");
    break;
  }
}

inline std::vector<int>
distribute_tasks_core_count(std::vector<float> task_weights,
                            std::shared_ptr<MPICluster> cluster) {
  std::vector<int> distribution;
  distribution.resize(task_weights.size());

  std::vector<std::pair<float, int>> indexed_node_weights;
  std::vector<int> cores_per_node = cluster->get_cores_per_node();
  for (int i = 0; i < cores_per_node.size(); ++i) {
    float weight =
        static_cast<float>(cores_per_node[i]) / cluster->get_total_core_count();
    indexed_node_weights.push_back(std::make_pair(weight, i));
  }

  std::vector<std::pair<float, int>> indexed_task_weights;
  for (int i = 0; i <= task_weights.size(); ++i) {
    indexed_task_weights.push_back(std::make_pair(task_weights[i], i));
  }

  distribution[0] = 0;
  indexed_node_weights.erase(indexed_node_weights.begin());
  indexed_task_weights.erase(indexed_task_weights.begin());

  std::sort(indexed_node_weights.begin(), indexed_node_weights.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first > b.first;
            });

  std::sort(indexed_task_weights.begin(), indexed_task_weights.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first > b.first;
            });

  for (int i = 0; i < indexed_task_weights.size(); ++i) {
    distribution[indexed_task_weights[i].second] =
        indexed_node_weights[i].second;
  }

  return distribution;
}

inline std::vector<int>
distribute_tasks_core_count_frequency(std::vector<float> task_weights,
                                      std::shared_ptr<MPICluster> cluster) {
  std::vector<int> distribution;
  distribution.resize(task_weights.size());

  std::vector<int> cores_per_node = cluster->get_cores_per_node();
  std::vector<int> frequency_per_node = cluster->get_frequency_per_node();

  int cluster_power = 0;
  for (int rank = 0; rank < frequency_per_node.size(); ++rank) {
    int power = cores_per_node[rank] * frequency_per_node[rank];
    cluster_power += power;
  }

  std::vector<std::pair<float, int>> indexed_node_weights;
  for (int rank = 0; rank < cores_per_node.size(); ++rank) {
    float weight =
        static_cast<float>(cores_per_node[rank] * frequency_per_node[rank]) /
        static_cast<float>(cluster_power);
    indexed_node_weights.push_back(std::make_pair(weight, rank));
  }

  std::vector<std::pair<float, int>> indexed_task_weights;
  for (int i = 0; i <= task_weights.size(); ++i) {
    indexed_task_weights.push_back(std::make_pair(task_weights[i], i));
  }

  distribution[0] = 0;
  indexed_node_weights.erase(indexed_node_weights.begin());
  indexed_task_weights.erase(indexed_task_weights.begin());

  std::sort(indexed_node_weights.begin(), indexed_node_weights.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first > b.first;
            });

  std::sort(indexed_task_weights.begin(), indexed_task_weights.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first > b.first;
            });

  for (int i = 0; i < indexed_task_weights.size(); ++i) {
    distribution[indexed_task_weights[i].second] =
        indexed_node_weights[i].second;
  }

  return distribution;
}

inline std::vector<int>
distribute_tasks(std::vector<float> task_weights, Distribution type,
                 std::shared_ptr<MPICluster> cluster) {
  switch (type) {
  case Distribution::CORE_COUNT:
    return distribute_tasks_core_count(task_weights, cluster);
    break;
  case Distribution::CORE_FREQUENCY:
    return distribute_tasks_core_count_frequency(task_weights, cluster);
    break;
  default:
    throw std::invalid_argument(
        "Selected distribution type is not implemented");
    break;
  }
}

} // namespace hmp

#endif // HMP_DISTRIBUTION_UTIL_H_
