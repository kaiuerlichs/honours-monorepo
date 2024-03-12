#include "hmp.h"
#include "hmp/map.h"
#include "mpi.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

struct TestData {
  int i;
  int j;
};

int test(TestData i) { return i.i * i.j; }

int main() {
  std::shared_ptr<hmp::MPICluster> cluster =
      std::make_shared<hmp::MPICluster>();
  std::unique_ptr<hmp::Map<TestData, int>> map =
      std::make_unique<hmp::Map<TestData, int>>(cluster);
  std::vector<TestData> vector;

  if (cluster->on_master()) {
    for (int i = 0; i < 64; ++i) {
      vector.push_back(TestData{i, i});
    }
  }

  MPI_Datatype type;
  int blocks[2] = {1, 1};
  MPI_Aint disp[2] = {offsetof(TestData, i), offsetof(TestData, j)};
  MPI_Datatype types[2] = {MPI_INT, MPI_INT};
  MPI_Type_create_struct(2, blocks, disp, types, &type);

  map->set_mpi_in_type(type);
  std::vector<int> ret = map->execute(vector, test);

  if (cluster->on_master()) {
    for (auto i : ret) {
      printf("%i ", i);
    }
    printf("\n");
  }

  return 0;
}
