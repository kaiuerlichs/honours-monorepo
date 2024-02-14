#ifndef INCLUDE_HMP_MAP_H_
#define INCLUDE_HMP_MAP_H_

#include <functional>
#include <memory>
#include <sys/wait.h>
#include <vector>

#include "hmp.h"

namespace hmp {

// Processes a 1-dimensional dataset using the Map pattern
// Example:
//  std::unique_ptr<Map<int, int>> parallelMap = std::make_unique<Map<int, int>>(cluster_ptr);
//  std::vector<int> results = parallelMap->execute(inputs, map_function_ptr);
template <typename IN_TYPE, typename OUT_TYPE> class Map {
private:
  std::shared_ptr<MPICluster> cluster; 
  std::vector<IN_TYPE> local_data;
  std::vector<OUT_TYPE> local_return_data;
  std::vector<OUT_TYPE> return_data;
  
  std::vector<int> elements_per_node;
  std::vector<int> displacements_per_node;

  void prepare_data(std::vector<IN_TYPE> &data);
  void run_map_function(std::function<OUT_TYPE(IN_TYPE)> map_function);
  void gather_data();

public:
  Map(std::shared_ptr<MPICluster> cluster_ptr);
  ~Map(){};

  std::vector<OUT_TYPE> execute(std::vector<IN_TYPE> &data,
                                std::function<OUT_TYPE(IN_TYPE)> map_function);
};

} // namespace hmp

#endif // INCLUDE_HMP_MAP_H_
