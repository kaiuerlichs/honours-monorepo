#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "hmp.h"

int test(int i) { return i * i; }

std::vector<int> generate_test_data() {
  std::vector<int> vector;
  for (int i = 0; i < 64; ++i) {
    vector.push_back(i);
  }
  printf("Input data: ");
  for (auto i : vector) {
    printf("%i ", i);
  }
  printf("\n");
  return vector;
}

void handle_output_data(std::vector<int> data) {
  printf("Output data: ");
  for (auto i : data) {
    printf("%i ", i);
  }
  printf("\n");
}

// Executes a simple map on an MPI cluster
void test_map() {
  auto cluster = std::make_shared<hmp::MPICluster>();
  std::vector<int> data;

  if (cluster->on_master())
    data = generate_test_data();

  auto map = std::make_unique<hmp::Map<int, int>>(
      cluster, hmp::Distribution::CORE_FREQUENCY);
  std::vector<int> return_data = map->execute(data, test);

  if (cluster->on_master())
    handle_output_data(return_data);
}

// Executes a simple pipeline with two stages on an MPI cluster
void test_pipeline() {
  auto cluster = std::make_shared<hmp::MPICluster>();
  std::vector<int> data;

  if (cluster->on_master()) 
    data = generate_test_data();

  auto pipeline = std::make_unique<hmp::Pipeline<int, int>>(cluster, hmp::Distribution::CORE_FREQUENCY);

  pipeline->add_stage<int, int>(test, 1);
  pipeline->add_stage<int, int>(test, 1);

  std::vector<int> out = pipeline->execute(data);

  if(cluster->on_master()) {
    handle_output_data(out);
  }
}

int main() {
  test_map();
  test_pipeline();
  return 0;
}
