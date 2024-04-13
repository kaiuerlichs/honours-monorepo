#include "distribution_util.h"
#include "hmp.h"
#include "hmp/map.h"
#include "hmp/pipeline.h"
#include "mpi.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

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

struct TestData {
  int a;
  int b;
};

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

void test_pipeline() {
  auto cluster = std::make_shared<hmp::MPICluster>();
  std::vector<int> data = {1, 2, 3};

  auto pipeline = std::make_unique<hmp::Pipeline<int, int>>(cluster, hmp::Distribution::CORE_FREQUENCY);

  pipeline->add_stage<int, int>([](int x) { printf("Stage 1"); return x * x; }, 1);
  pipeline->add_stage<int, int>([](int x) { printf("Stage 2"); return x * x; }, 1);

  std::vector<int> out = pipeline->execute(data);
  if(cluster->on_master()) {
    handle_output_data(out);
  }
}

int main() {
  test_pipeline();
  return 0;
}
