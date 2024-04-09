#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hmp.h>
#include <hmp/map.h>
#include <mpi.h>
#include <omp.h>

#include "parsec_blackscholes.h"

std::string version;
int runs;
std::string input_file;
std::string output_file;

std::vector<int> run_duration;

std::vector<OptionData> options;
int options_count;
std::vector<float> prices;

void read_options_from_file() {
  std::cout << "Reading options from \"" << input_file << "\"" << std::endl;
  std::ifstream file(input_file);
  std::string line;
  int i = 0;

  if (!file.is_open()) {
    std::cerr << "Unable to open input file" << std::endl;
    exit(1);
  }

  std::getline(file, line);
  options_count = std::stoi(line);

  while (std::getline(file, line)) {
    OptionData opt;
    std::istringstream iss(line);

    if (!(iss >> opt.s >> opt.strike >> opt.r >> opt.divq >> opt.v >> opt.t >>
          opt.OptionType >> opt.divs >> opt.DGrefval)) {
      std::cerr << "Error processing line: " << line << std::endl;
      exit(1);
    }

    options.push_back(opt);
  }

  file.close();
  std::cout << "Read complete\n" << std::endl;
}

void write_prices_to_file() {
  std::cout << "Writing prices to \"" << output_file << "\"" << std::endl;
  std::ofstream file(output_file);

  if (!file.is_open()) {
    std::cerr << "Unable to open output file" << std::endl;
    exit(1);
  }

  file << options_count << "\n";
  if (file.fail()) {
    std::cerr << "Cannot write to output file" << std::endl;
    file.close();
    exit(1);
  }

  for (int i = 0; i < options_count; ++i) {
    file << std::fixed << std::setprecision(18) << prices[i] << "\n";
  }

  file.close();
  std::cout << "Write complete\n" << std::endl;
}

void run_serial() {
  read_options_from_file();
  prices.resize(options_count);

  for (int run = 0; run < runs; ++run) {
    std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < options_count; ++i) {
      prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                      options[i].r, options[i].v, options[i].t,
                                      options[i].OptionType, 0);
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
  write_prices_to_file();
}

void run_openmp() {
  read_options_from_file();
  prices.resize(options_count);
  int threads = omp_get_max_threads();

  for (int run = 0; run < runs; ++run) {
    std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < options_count; ++i) {
      prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                      options[i].r, options[i].v, options[i].t,
                                      options[i].OptionType, 0);
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

  write_prices_to_file();
}

void run_openmpi() {

  int init_flag;
  MPI_Initialized(&init_flag);

  if (!init_flag)
    MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    read_options_from_file();
  }

  int threads = omp_get_max_threads();
  MPI_Bcast(&options_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int nitems = 9;
  int blocklengths[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Datatype types[9] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_CHAR,  MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype mpi_option_type;
  MPI_Aint offsets[9];

  offsets[0] = offsetof(OptionData, s);
  offsets[1] = offsetof(OptionData, strike);
  offsets[2] = offsetof(OptionData, r);
  offsets[3] = offsetof(OptionData, divq);
  offsets[4] = offsetof(OptionData, v);
  offsets[5] = offsetof(OptionData, t);
  offsets[6] = offsetof(OptionData, OptionType);
  offsets[7] = offsetof(OptionData, divs);
  offsets[8] = offsetof(OptionData, DGrefval);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_option_type);
  MPI_Type_commit(&mpi_option_type);

  for (int run = 0; run < runs; ++run) {
    if (rank == 0)
      std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int options_per_process = options_count / size;
    int remainder = options_count % size;

    std::vector<int> sendcounts(size, options_per_process);
    std::vector<int> displs(size, 0);

    for (int i = 0; i < remainder; ++i) {
      sendcounts[i]++;
    }

    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    int recvcount = sendcounts[rank];

    std::vector<OptionData> local_options;
    local_options.resize(recvcount);
    std::vector<float> local_prices;
    local_prices.resize(recvcount);

    MPI_Scatterv(options.data(), sendcounts.data(), displs.data(),
                 mpi_option_type, local_options.data(), recvcount,
                 mpi_option_type, 0, MPI_COMM_WORLD);

    for (int i = 0; i < recvcount; ++i) {
      local_prices[i] = BlkSchlsEqEuroNoDiv(
          local_options[i].s, local_options[i].strike, local_options[i].r,
          local_options[i].v, local_options[i].t,
          local_options[i].OptionType == 'C' ? 0 : 1, 0);
    }

    prices.resize(options_count);

    MPI_Gatherv(local_prices.data(), recvcount, MPI_FLOAT, prices.data(),
                sendcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (rank == 0)
      std::cout << "Finished in " << duration << "ms\n" << std::endl;
  }

  if (rank == 0) {
    int total_ms = 0;
    for (int run = 0; run < runs; ++run) {
      total_ms += run_duration[run];
    }
    double average_ms =
        static_cast<double>(total_ms) / static_cast<double>(runs);
    std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

    write_prices_to_file();
  }

  MPI_Type_free(&mpi_option_type);
  MPI_Finalize();
}

void run_hmpcore() {
  auto cluster = std::make_shared<hmp::MPICluster>();

  const int nitems = 9;
  int blocklengths[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Datatype types[9] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_CHAR,  MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype mpi_option_type;
  MPI_Aint offsets[9];

  offsets[0] = offsetof(OptionData, s);
  offsets[1] = offsetof(OptionData, strike);
  offsets[2] = offsetof(OptionData, r);
  offsets[3] = offsetof(OptionData, divq);
  offsets[4] = offsetof(OptionData, v);
  offsets[5] = offsetof(OptionData, t);
  offsets[6] = offsetof(OptionData, OptionType);
  offsets[7] = offsetof(OptionData, divs);
  offsets[8] = offsetof(OptionData, DGrefval);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_option_type);

  if (cluster->on_master())
    read_options_from_file();

  for (int run = 0; run < runs; ++run) {
    if (cluster->on_master())
      std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;

    auto map = std::make_unique<hmp::Map<OptionData, float>>(
        cluster, hmp::Distribution::CORE_COUNT);
    map->set_mpi_in_type(mpi_option_type);

    auto start = std::chrono::high_resolution_clock::now();
    prices = map->execute(options, [](OptionData opt) {
      return BlkSchlsEqEuroNoDiv(opt.s, opt.strike, opt.r, opt.v, opt.t,
                                 opt.OptionType == 'C' ? 0 : 1, 0);
    });
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (cluster->on_master())
      std::cout << "Finished in " << duration << "ms\n" << std::endl;
  }

  if (cluster->on_master()) {
    int total_ms = 0;
    for (int run = 0; run < runs; ++run) {
      total_ms += run_duration[run];
    }
    double average_ms =
        static_cast<double>(total_ms) / static_cast<double>(runs);
    std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

    write_prices_to_file();
  }
}
void run_hmpfreq() {
  auto cluster = std::make_shared<hmp::MPICluster>();

  const int nitems = 9;
  int blocklengths[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Datatype types[9] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                           MPI_CHAR,  MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype mpi_option_type;
  MPI_Aint offsets[9];

  offsets[0] = offsetof(OptionData, s);
  offsets[1] = offsetof(OptionData, strike);
  offsets[2] = offsetof(OptionData, r);
  offsets[3] = offsetof(OptionData, divq);
  offsets[4] = offsetof(OptionData, v);
  offsets[5] = offsetof(OptionData, t);
  offsets[6] = offsetof(OptionData, OptionType);
  offsets[7] = offsetof(OptionData, divs);
  offsets[8] = offsetof(OptionData, DGrefval);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_option_type);

  if (cluster->on_master())
    read_options_from_file();

  for (int run = 0; run < runs; ++run) {
    if (cluster->on_master())
      std::cout << "Starting run " << run + 1 << "/" << runs << std::endl;

    auto map = std::make_unique<hmp::Map<OptionData, float>>(
        cluster, hmp::Distribution::CORE_FREQUENCY);
    map->set_mpi_in_type(mpi_option_type);

    auto start = std::chrono::high_resolution_clock::now();
    prices = map->execute(options, [](OptionData opt) {
      return BlkSchlsEqEuroNoDiv(opt.s, opt.strike, opt.r, opt.v, opt.t,
                                 opt.OptionType == 'C' ? 0 : 1, 0);
    });
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    run_duration.push_back(duration);

    if (cluster->on_master())
      std::cout << "Finished in " << duration << "ms\n" << std::endl;
  }

  if (cluster->on_master()) {
    int total_ms = 0;
    for (int run = 0; run < runs; ++run) {
      total_ms += run_duration[run];
    }
    double average_ms =
        static_cast<double>(total_ms) / static_cast<double>(runs);
    std::cout << "Average run duration: " << average_ms << "ms\n" << std::endl;

    write_prices_to_file();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: <VERSION: serial|openmp|openmpi|hmpcore|hmpfreq> "
                 "<RUNS> <INPUT FILE> <OUTPUT FILE>"
              << std::endl;
    return 1;
  }

  version = argv[1];
  runs = std::stoi(argv[2]);
  input_file = argv[3];
  output_file = argv[4];

  if (version == "serial") {
    run_serial();
  } else if (version == "openmp") {
    run_openmp();
  } else if (version == "openmpi") {
    run_openmpi();
  } else if (version == "hmpcore") {
    run_hmpcore();
  } else if (version == "hmpfreq") {
    run_hmpfreq();
  }

  return 0;
}
