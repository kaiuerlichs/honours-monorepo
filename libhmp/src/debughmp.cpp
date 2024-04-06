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

  auto pipeline = std::make_unique<hmp::Pipeline<int, double>>(cluster, hmp::Distribution::CORE_FREQUENCY);

  int blocklengths[2] = {1, 1};

  // Displacements of each block
  MPI_Aint displacements[2];
  TestData temp;
  MPI_Get_address(&temp.a, &displacements[0]);
  MPI_Get_address(&temp.b, &displacements[1]);

  // Types of each block
  MPI_Datatype types[2] = {MPI_INT, MPI_INT};

  // Adjust displacements to be relative to the stat of the struct
  displacements[1] -= displacements[0];
  displacements[0] = 0;

  // Create the MPI datatype
  MPI_Datatype testDataMPIType;
  MPI_Type_create_struct(2, blocklengths, displacements, types,
                         &testDataMPIType);

  pipeline->add_mpi_type<TestData>(testDataMPIType);

  pipeline->add_stage<int, double>([](int x) { std::this_thread::sleep_for(std::chrono::seconds(1)); return x; }, 0);
  pipeline->add_stage<double, TestData>([](double x) { std::this_thread::sleep_for(std::chrono::seconds(2)); return TestData(); }, 0);
  pipeline->add_stage<TestData, double>([](TestData x) { std::this_thread::sleep_for(std::chrono::seconds(3)); return 0; }, TestData());

  pipeline->execute(data);
}

int main() {
  test_pipeline();
  return 0;
}
