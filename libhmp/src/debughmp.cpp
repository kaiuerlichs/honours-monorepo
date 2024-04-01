#include "distribution_util.h"
#include "hmp.h"
#include "hmp/map.h"
#include "mpi.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

int test(int i) { return i * i; }

int main() {
  std::shared_ptr<hmp::MPICluster> cluster =
      std::make_shared<hmp::MPICluster>();
  std::unique_ptr<hmp::Map<int, int>> map =
      std::make_unique<hmp::Map<int, int>>(cluster, hmp::Distribution::CORE_FREQUENCY);
  std::vector<int> vector;

  if (cluster->on_master()) {
    for (int i = 0; i < 64; ++i) {
      vector.push_back(i);
    }
    printf("Input data: ");
    for (auto i : vector) {
      printf("%i ", i);
    }
    printf("\n");
  }

  std::vector<int> ret = map->execute(vector, test);

  if (cluster->on_master()) {
    printf("Output data: ");
    for (auto i : ret) {
      printf("%i ", i);
    }
    printf("\n");
  }

  return 0;
}
