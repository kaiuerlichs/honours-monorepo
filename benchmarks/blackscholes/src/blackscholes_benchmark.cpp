#include <__chrono/duration.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>
#include <mpi.h>

enum class Version {
  SERIAL,
  OPEN_MP,
  OPENMPI,
  HMP,
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

std::vector<OptionData> options;
int options_count;
std::vector<float> prices;

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
}

void run_serial() {
  for (int i = 0; i < options_count; ++i) {
    prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                    options[i].r, options[i].v, options[i].t,
                                    options[i].OptionType, 0);
  }
}

void run_openmp() {
  int threads = omp_get_max_threads();
  std::cout << threads << std::endl;
  #pragma omp parallel for num_threads(threads)
  for (int i = 0; i < options_count; ++i) {
    std::cout << omp_get_thread_num() << std::endl;
    prices[i] = BlkSchlsEqEuroNoDiv(options[i].s, options[i].strike,
                                    options[i].r, options[i].v, options[i].t,
                                    options[i].OptionType, 0);
  }
}

void run_openmpi() {
  MPI_Init(NULL, NULL);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: <VERSION NAME: serial|openmp|openmpi|hmp> "
                 "<NODE COUNT> <RUNS> <INPUT_FILE> <OUTPUT_FILE>"
              << std::endl;
    return 1;
  }

  char *version_string = argv[1];
  Version version;

  if (strcmp(version_string, "serial") == 0) {
    version = Version::SERIAL;
  } else if (strcmp(version_string, "openmp") == 0) {
    version = Version::OPEN_MP;
  } else if (strcmp(version_string, "openmpi") == 0) {
    version = Version::OPENMPI;
  } else if (strcmp(version_string, "hmp") == 0) {
    version = Version::HMP;
  } else {
    std::cerr << "Version must be one of: serial|openmp|openmpi|hmp"
              << std::endl;
    return 1;
  }

  int node_count = std::atoi(argv[2]);
  if (node_count <= 0) {
    std::cerr << "Node count must be positive." << std::endl;
    return 1;
  }

  int runs = std::atoi(argv[3]);
  if (runs <= 0) {
    std::cerr << "Runs must be positive." << std::endl;
    return 1;
  }

  std::string input_file = argv[4];
  std::string output_file = argv[5];

  read_options_from_file(input_file);
  prices.resize(options_count);
  
  auto start = std::chrono::high_resolution_clock::now();

  switch(version){
    case Version::SERIAL:
      run_serial();
      break;
    case Version::OPEN_MP:
      run_openmp();
      break;
    default:
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration(end-start);

  std::cout << duration.count() << std::endl;

  write_prices_to_file(output_file);
}
