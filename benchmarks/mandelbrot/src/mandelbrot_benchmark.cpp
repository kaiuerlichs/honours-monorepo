#include "distribution_util.h"
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
  std::cout << "Generating mandelbrot input" << std::endl;
  int threads = omp_get_max_threads();

#pragma omp parallel for collapse(2) num_threads(threads)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      double real = x_min + j * (x_max - x_min) / width;
      double imag = y_min + i * (y_max - y_min) / height;
      input[i * width + j] = std::complex<double>(real, imag);
    }
  }
  std::cout << "Generation complete\n" << std::endl;
}

void write_mandelbrot_output() {
  std::cout << "Writing output to file" << std::endl;
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
  std::cout << "Write complete\n" << std::endl;
}

MPI_Datatype get_mpi_complex_type() {
  MPI_Datatype mpi_complex_type;
  MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_complex_type);
  MPI_Type_commit(&mpi_complex_type);
  return mpi_complex_type;
}

void run_serial() {
  generate_mandelbrot_input();

  for (int run = 0; run < runs; ++run) {
    std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        std::cout << "hey" << std::endl;
        output[i * width + j] = mandelbrot(input[i * width + j]);
      }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);
    std::cout << "Finished in " << duration << "ms\n" << std::endl;
  }

  int total_ms = 0;
  for (int run = 0; run < runs; ++run) {
    total_ms += run_duration[run];
  }
  double average_ms = static_cast<double>(total_ms) / static_cast<double>(runs);
  std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

  write_mandelbrot_output();
}

void run_openmp() {
  generate_mandelbrot_input();
  int threads = omp_get_max_threads();

  for (int run = 0; run < runs; ++run) {
    std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2) num_threads(threads)
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        output[i * width + j] = mandelbrot(input[i * width + j]);
      }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);
    std::cout << "Finished in " << duration << "ms\n" << std::endl;
  }

  int total_ms = 0;
  for (int run = 0; run < runs; ++run) {
    total_ms += run_duration[run];
  }
  double average_ms = static_cast<double>(total_ms) / static_cast<double>(runs);
  std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

  write_mandelbrot_output();
}

void run_openmpi() {
  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    generate_mandelbrot_input();
  }

  MPI_Datatype mpi_complex = get_mpi_complex_type();
  MPI_Type_commit(&mpi_complex);

  int threads = omp_get_max_threads();

  for (int run = 0; run < runs; ++run) {
    if (rank == 0) {
      std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();

    int points_total = width * height;
    int points_per_node = points_total / size;
    int remaining_points = points_total % size;

    std::vector<int> sendcounts(size, points_per_node);
    std::vector<int> displacements(size, 0);

    for (int i = 0; i < remaining_points; ++i) {
      sendcounts[i]++;
    }

    for (int i = 1; i < size; ++i) {
      displacements[i] = displacements[i - 1] + sendcounts[i - 1];
    }

    int recvcount = sendcounts[rank];

    std::vector<std::complex<double>> local_input;
    local_input.resize(recvcount);
    std::vector<int> local_output;
    local_output.resize(recvcount);

    MPI_Scatterv(input.data(), sendcounts.data(), displacements.data(),
                 mpi_complex, local_input.data(), recvcount, mpi_complex, 0,
                 MPI_COMM_WORLD);

#pragma omp parallel for collapse(2) num_threads(threads)
    for (int i = 0; i < local_input.size(); ++i) {
      local_output[i] = mandelbrot(local_input[i]);
    }

    MPI_Gatherv(local_output.data(), recvcount, MPI_INT, output.data(),
                sendcounts.data(), displacements.data(), MPI_INT, 0,
                MPI_COMM_WORLD);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (rank == 0) {
      std::cout << "Finished in " << duration << "ms\n" << std::endl;
    }
  }

  if (rank == 0) {
    int total_ms = 0;
    for (int run = 0; run < runs; ++run) {
      total_ms += run_duration[run];
    }
    double average_ms =
        static_cast<double>(total_ms) / static_cast<double>(runs);
    std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

    write_mandelbrot_output();
  }

  MPI_Type_free(&mpi_complex);
  MPI_Finalize();
}

void run_hmp(hmp::Distribution d) {
  auto cluster = std::make_shared<hmp::MPICluster>();

  if (cluster->on_master()) {
    generate_mandelbrot_input();
  }

  auto map = std::make_unique<hmp::Map<std::complex<double>, int>>(cluster, d);
  map->set_mpi_in_type(get_mpi_complex_type());

  for (int run = 0; run < runs; ++run) {
    if (cluster->on_master()) {
      std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    output = map->execute(input, mandelbrot);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (cluster->on_master()) {
      std::cout << "Finished in " << duration << "ms\n" << std::endl;
    }
  }

  if (cluster->on_master()) {
    int total_ms = 0;
    for (int run = 0; run < runs; ++run) {
      total_ms += run_duration[run];
    }
    double average_ms =
        static_cast<double>(total_ms) / static_cast<double>(runs);
    std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

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

  if (version == "serial") {
    run_serial();
  } else if (version == "openmp") {
    run_openmp();
  } else if (version == "openmpi") {
    run_openmpi();
  } else if (version == "hmpcore") {
    run_hmp(hmp::Distribution::CORE_COUNT);
  } else if (version == "hmpfreq") {
    run_hmp(hmp::Distribution::CORE_FREQUENCY);
  }

  return 0;
}
