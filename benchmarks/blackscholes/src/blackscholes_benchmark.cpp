#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

enum class Version {
  SERIAL,
  OPEN_MP,
  OPENMPI,
  HMP,
};

struct OptionData {
  float s;        // spot price
  float strike;   // strike price
  float r;        // risk-free interest rate
  float divq;     // dividend rate
  float v;        // volatility
  float t;        // time to maturity or option expiration in years
                   //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
  char OptionType; // Option type.  "P"=PUT, "C"=CALL
  float divs;     // dividend vals (not used in this test)
  float DGrefval; // DerivaGem Reference Value
};

std::vector<OptionData> options;
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


int read_options_from_file()

int main(int argc, char *argv[]) {
  if (argc < 2) {
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

  int node_count = 1;
  if (argc > 2) {
    int node_count = std::atoi(argv[2]);
    if (node_count <= 0) {
      std::cerr << "Node count must be positive." << std::endl;
      return 1;
    }
  }

  int runs = 1;
  if (argc > 2) {
    int runs = std::atoi(argv[3]);
    if (runs <= 0) {
      std::cerr << "Runs must be positive." << std::endl;
      return 1;
    }
  }
}
