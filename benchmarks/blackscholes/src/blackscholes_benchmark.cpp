#include "distribution_util.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <hmp.h>
#include <hmp/map.h>
#include <mpi.h>
#include <omp.h>

enum class Version {
  SERIAL,
  OPEN_MP,
  OPENMPI,
  HMPCORE,
  HMPFREQ,
};

struct OptionData {
  float s;         // spot price
  float strike;    // strike price
  float r;         // risk-free interest rate
  float divq;      // dividend rate
  float v;         // volatility
  float t;         // time to maturity or option expiration in years
                   //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
  char OptionType; // Option type.  "P"=PUT, "C"=CALL
  float divs;      // dividend vals (not used in this test)
  float DGrefval;  // DerivaGem Reference Value
};

Version version;

std::vector<OptionData> options;
int options_count;
std::vector<float> prices;

std::string input_file;
std::string output_file;

#define inv_sqrt_2xPI 0.39894228040143270286
float CNDF(float InputX) {
  int sign;

  float OutputX;
  float xInput;
  float xNPrimeofX;
  float expValues;
  float xK2;
  float xK2_2, xK2_3;
  float xK2_4, xK2_5;
  float xLocal, xLocal_1;
  float xLocal_2, xLocal_3;

  // Check for negative value of InputX
  if (InputX < 0.0) {
    InputX = -InputX;
    sign = 1;
  } else
    sign = 0;

  xInput = InputX;

  // Compute NPrimeX term common to both four & six decimal accuracy calcs
  expValues = exp(-0.5f * InputX * InputX);
  xNPrimeofX = expValues;
  xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

  xK2 = 0.2316419 * xInput;
  xK2 = 1.0 + xK2;
  xK2 = 1.0 / xK2;
  xK2_2 = xK2 * xK2;
  xK2_3 = xK2_2 * xK2;
  xK2_4 = xK2_3 * xK2;
  xK2_5 = xK2_4 * xK2;

  xLocal_1 = xK2 * 0.319381530;
  xLocal_2 = xK2_2 * (-0.356563782);
  xLocal_3 = xK2_3 * 1.781477937;
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_4 * (-1.821255978);
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_5 * 1.330274429;
  xLocal_2 = xLocal_2 + xLocal_3;

  xLocal_1 = xLocal_2 + xLocal_1;
  xLocal = xLocal_1 * xNPrimeofX;
  xLocal = 1.0 - xLocal;

  OutputX = xLocal;

  if (sign) {
    OutputX = 1.0 - OutputX;
  }

  return OutputX;
}

float BlkSchlsEqEuroNoDiv(float sptprice, float strike, float rate,
                          float volatility, float time, int otype,
                          float timet) {
  float OptionPrice;

  // local private working variables for the calculation
  float xStockPrice;
  float xStrikePrice;
  float xRiskFreeRate;
  float xVolatility;
  float xTime;
  float xSqrtTime;

  float logValues;
  float xLogTerm;
  float xD1;
  float xD2;
  float xPowerTerm;
  float xDen;
  float d1;
  float d2;
  float FutureValueX;
  float NofXd1;
  float NofXd2;
  float NegNofXd1;
  float NegNofXd2;

  xStockPrice = sptprice;
  xStrikePrice = strike;
  xRiskFreeRate = rate;
  xVolatility = volatility;

  xTime = time;
  xSqrtTime = sqrt(xTime);

  logValues = log(sptprice / strike);

  xLogTerm = logValues;

  xPowerTerm = xVolatility * xVolatility;
  xPowerTerm = xPowerTerm * 0.5;

  xD1 = xRiskFreeRate + xPowerTerm;
  xD1 = xD1 * xTime;
  xD1 = xD1 + xLogTerm;

  xDen = xVolatility * xSqrtTime;
  xD1 = xD1 / xDen;
  xD2 = xD1 - xDen;

  d1 = xD1;
  d2 = xD2;

  NofXd1 = CNDF(d1);
  NofXd2 = CNDF(d2);

  FutureValueX = strike * (exp(-(rate) * (time)));
  if (otype == 0) {
    OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
  } else {
    NegNofXd1 = (1.0 - NofXd1);
    NegNofXd2 = (1.0 - NofXd2);
    OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
  }

  return OptionPrice;
}

void read_options_from_file(std::string input_file) {
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
}

void write_prices_to_file(std::string output_file) {
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
}

void run_serial() {
  read_options_from_file(input_file);
  prices.resize(options_count);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < options_count; ++i) {
    prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                    options[i].r, options[i].v, options[i].t,
                                    options[i].OptionType, 0);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  std::cout << duration << std::endl;

  write_prices_to_file(output_file);
}

void run_openmp() {
  read_options_from_file(input_file);
  prices.resize(options_count);
  int threads = omp_get_max_threads();

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads)
  for (int i = 0; i < options_count; ++i) {
    prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                    options[i].r, options[i].v, options[i].t,
                                    options[i].OptionType, 0);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  std::cout << duration << std::endl;

  write_prices_to_file(output_file);
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
    read_options_from_file(input_file);
  }

  MPI_Bcast(&options_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int threads = omp_get_max_threads();

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

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < recvcount; ++i) {
    local_prices[i] = BlkSchlsEqEuroNoDiv(
        local_options[i].s, local_options[i].strike, local_options[i].r,
        local_options[i].v, local_options[i].t,
        local_options[i].OptionType == 'C' ? 0 : 1, 0);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);


  prices.resize(options_count);

  MPI_Gatherv(local_prices.data(), recvcount, MPI_FLOAT, prices.data(),
              sendcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << duration << std::endl;
    write_prices_to_file(output_file);
  }

  MPI_Type_free(&mpi_option_type);
  MPI_Finalize();
}

void run_hmp_core() {
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
    read_options_from_file(input_file);

  auto map = std::make_unique<hmp::Map<OptionData, float>>(
      cluster, hmp::Distribution::CORE_FREQUENCY);
  map->set_mpi_in_type(mpi_option_type);

  auto start = std::chrono::high_resolution_clock::now();
  prices = map->execute(options, [](OptionData opt) {
    return BlkSchlsEqEuroNoDiv(opt.s, opt.strike, opt.r, opt.v, opt.t,
                               opt.OptionType == 'C' ? 0 : 1, 0);
  });
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  if (cluster->on_master()) {
    std::cout << duration << std::endl;
    write_prices_to_file(output_file);
  }
}

void run_hmp_freq() {
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
    read_options_from_file(input_file);

  auto map = std::make_unique<hmp::Map<OptionData, float>>(
      cluster, hmp::Distribution::CORE_FREQUENCY);
  map->set_mpi_in_type(mpi_option_type);

  auto start = std::chrono::high_resolution_clock::now();
  prices = map->execute(options, [](OptionData opt) {
    return BlkSchlsEqEuroNoDiv(opt.s, opt.strike, opt.r, opt.v, opt.t,
                               opt.OptionType == 'C' ? 0 : 1, 0);
  });
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  if (cluster->on_master()) {
    std::cout << duration << std::endl;
    write_prices_to_file(output_file);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: <VERSION NAME: serial|openmp|openmpi|hmp> "
                 "<INPUT_FILE> <OUTPUT_FILE>"
              << std::endl;
    return 1;
  }

  char *version_string = argv[1];

  if (strcmp(version_string, "serial") == 0) {
    version = Version::SERIAL;
  } else if (strcmp(version_string, "openmp") == 0) {
    version = Version::OPEN_MP;
  } else if (strcmp(version_string, "openmpi") == 0) {
    version = Version::OPENMPI;
  } else if (strcmp(version_string, "hmpcore") == 0) {
    version = Version::HMPCORE;
  } else if (strcmp(version_string, "hmpfreq") == 0) {
    version = Version::HMPFREQ;
  } else {
    std::cerr << "Version must be one of: serial|openmp|openmpi|hmpcore|hmpfreq"
              << std::endl;
    return 1;
  }

  input_file = argv[2];
  output_file = argv[3];

  switch (version) {
  case Version::SERIAL:
    run_serial();
    break;
  case Version::OPEN_MP:
    run_openmp();
    break;
  case Version::OPENMPI:
    run_openmpi();
    break;
  case Version::HMPCORE:
    run_hmp_core();
    break;
  case Version::HMPFREQ:
    run_hmp_freq();
    break;
  default:
    break;
  }
}
