#include "distribution_util.h"
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

#include <hmp.h>
#include <hmp/map.h>
#include <mpi.h>
#include <omp.h>

#define MANDELBROT_MAX_ITERATIONS 1000

double x_min, x_max, y_min, y_max;
int width, height;

std::vector<std::complex<double>> input;
std::vector<int> output;

std::string version;
int runs;

std::vector<int> run_duration;

int mandelbrot(std::complex<double> c) {
  std::complex<double> z = 0;
  int n = 0;
  while (abs(z) <= 2 && n < MANDELBROT_MAX_ITERATIONS) {
    z = z * z + c;
    ++n;
  }
  return n;
}

void generate_mandelbrot_input() {
  std::cout << "Generating mandelbrot input on master." << std::endl;
  int threads = omp_get_max_threads();

#pragma omp parallel for collapse(2) num_threads(threads)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      double real = x_min + j * (x_max - x_min) / width;
      double imag = y_min + i * (y_max - y_min) / height;
      input[i * width + j] = std::complex<double>(real, imag);
    }
  }
  std::cout << "Generation complete.\n" << std::endl;
}

void write_mandelbrot_output() {
  std::cout << "Writing output to file on master." << std::endl;
  std::ofstream file("mandelbrot.txt");
  if (!file.is_open()) {
    std::cerr << "Error opening file" << std::endl;
    return;
  }
  file << x_min << " " << x_max << " " << y_min << " " << y_max << " " << width
       << " " << height << "\n";
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      file << output[i * width + j];
      if (j != width - 1)
        file << " ";
    }
    file << "\n";
  }
  file.close();
  std::cout << "Write complete.\n" << std::endl;
}

MPI_Datatype get_mpi_complex_type() {
  MPI_Datatype mpi_complex_type;
  MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_complex_type);
  MPI_Type_commit(&mpi_complex_type);
  return mpi_complex_type;
}

void run_hmp(hmp::Distribution d) {
  auto cluster = std::make_shared<hmp::MPICluster>();

  if (cluster->on_master()) {
    generate_mandelbrot_input();
  }

  auto map = std::make_unique<hmp::Map<std::complex<double>, int>>(cluster, d);
  map->set_mpi_in_type(get_mpi_complex_type());

  for (int run = 0; run < runs; ++run) {
    char hostname[256];
    gethostname(hostname, 256);
    int thread_count = omp_get_max_threads();

    std::cout << "Hello from node " << hostname << "! Running mandelbrot with " << thread_count << "threads." << std::endl; 

    auto start = std::chrono::high_resolution_clock::now();
    map->set_map_function(mandelbrot);
    output = map->execute(input);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (cluster->on_master()) {
      std::cout << "Finished in " << duration << "ms.\n" << std::endl;
    }
  }

  if (cluster->on_master()) {
    write_mandelbrot_output();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 9) {
    std::cerr << "Usage: <VERSION: serial|openmp|openmpi|hmpcore|hmpfreq> "
                 "<RUNS> <XMIN> <XMAX> <YMIN> <YMAX> <WIDTH> <HEIGHT>"
              << std::endl;
    return 1;
  }

  version = argv[1];
  runs = std::stoi(argv[2]);
  x_min = std::atof(argv[3]);
  x_max = std::atof(argv[4]);
  y_min = std::atof(argv[5]);
  y_max = std::atof(argv[6]);
  width = std::atoi(argv[7]);
  height = std::atoi(argv[8]);

  input.resize(width * height);
  output.resize(width * height);

  run_hmp(hmp::Distribution::CORE_COUNT);
  
  return 0;
}
