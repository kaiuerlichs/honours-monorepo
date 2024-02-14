#include "hmp.h"
#include "hmp/map.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

int test(int i) { return i*i; }

int main() {
  std::shared_ptr<hmp::MPICluster> cluster =
      std::make_shared<hmp::MPICluster>();
  std::unique_ptr<hmp::Map<int, int>> map =
      std::make_unique<hmp::Map<int, int>>(cluster);
  std::vector<int> vector;

  if (cluster->on_master()) {
    for (int i = 0; i < 64; ++i) {
      vector.push_back(i);
    }
  }

  std::vector<int> ret = map->execute(vector, test);

  if (cluster->on_master()) {
    for (auto i : ret) {
      printf("%i ", i);
    }
    printf("\n");
  }

  return 0;
}
