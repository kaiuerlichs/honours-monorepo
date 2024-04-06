#ifndef HMP_PIPELINE_H_
#define HMP_PIPELINE_H_

#include "mpi.h"
#include <any>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "hmp.h"
#include "hmp/distribution_util.h"
#include "hmp/mpi_type_traits.h"

namespace hmp {

struct IStage {
  virtual ~IStage() = default;
  virtual const std::type_info &input_type() const = 0;
  virtual const std::type_info &output_type() const = 0;
  virtual int profile() = 0;

  int profiling_runtime;
};

template <typename IN_TYPE, typename OUT_TYPE> struct Stage : IStage {
  std::function<OUT_TYPE(IN_TYPE)> stage_function;

  MPI_Datatype mpi_input_type;
  MPI_Datatype mpi_output_type;

  IN_TYPE profiling_input;

  Stage(std::function<OUT_TYPE(IN_TYPE)> f, MPI_Datatype &mpi_in_type,
        MPI_Datatype &mpi_out_type, IN_TYPE profiling_in);

  int profile() override;

  const std::type_info &input_type() const override { return typeid(IN_TYPE); }

  const std::type_info &output_type() const override {
    return typeid(OUT_TYPE);
  }
};

template <typename IN_TYPE, typename OUT_TYPE> class Pipeline {
private:
  std::shared_ptr<MPICluster> cluster;
  std::vector<std::unique_ptr<IStage>> stages;
  std::unordered_map<std::type_index, MPI_Datatype> mpi_type_table;
  int stage_count = 0;
  int total_profiling_runtime;
  std::vector<std::vector<int>> stages_per_node;

  Distribution distribution_type;

  void profile_stages();
  void allocate_stages();

public:
  Pipeline(std::shared_ptr<MPICluster> cluster_ptr, Distribution distribution);
  ~Pipeline(){};

  template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
  void add_stage(std::function<STAGE_OUT_TYPE(STAGE_IN_TYPE)> f,
                 STAGE_IN_TYPE profiling_input);

  template <typename C_TYPE> void add_mpi_type(MPI_Datatype mpi_type);

  std::vector<OUT_TYPE> execute(std::vector<IN_TYPE> &data);
};

// IMPLEMENTATION

template <typename IN_TYPE, typename OUT_TYPE>
Stage<IN_TYPE, OUT_TYPE>::Stage(std::function<OUT_TYPE(IN_TYPE)> f,
                                MPI_Datatype &mpi_in_type,
                                MPI_Datatype &mpi_out_type,
                                IN_TYPE profiling_in) {
  stage_function = std::move(f);
  mpi_input_type = mpi_in_type;
  mpi_output_type = mpi_out_type;
  profiling_input = profiling_in;
}

template <typename IN_TYPE, typename OUT_TYPE>
int Stage<IN_TYPE, OUT_TYPE>::profile() {
  auto start = std::chrono::high_resolution_clock::now();
  stage_function(profiling_input);
  auto end = std::chrono::high_resolution_clock::now();
  profiling_runtime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return profiling_runtime;
}

template <typename IN_TYPE, typename OUT_TYPE>
Pipeline<IN_TYPE, OUT_TYPE>::Pipeline(std::shared_ptr<MPICluster> cluster_ptr,
                                      Distribution distribution) {
  cluster = cluster_ptr;

  if (cluster->is_linux()) {
    distribution_type = distribution;
  } else {
    printf("Defaulting to CORE_COUNT distribution due to operating system "
           "constraints\n");
    distribution_type = Distribution::CORE_COUNT;
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::add_stage(
    std::function<STAGE_OUT_TYPE(STAGE_IN_TYPE)> f,
    STAGE_IN_TYPE profiling_input) {
  if (stage_count == 0 && typeid(IN_TYPE) != typeid(STAGE_IN_TYPE)) {
    throw std::invalid_argument(
        "Input type of first stage does not match pipeline input type");
  } else if (stage_count > 0 &&
             stages.back()->output_type() != typeid(STAGE_IN_TYPE)) {
    std::stringstream exception_message;
    exception_message << "Input type of stage " << stage_count + 1
                      << " does not match output type of stage " << stage_count;
    throw std::invalid_argument(exception_message.str());
  }

  std::type_index stage_in_index = std::type_index(typeid(STAGE_IN_TYPE));
  std::type_index stage_out_index = std::type_index(typeid(STAGE_OUT_TYPE));

  MPI_Datatype mpi_in_type, mpi_out_type;
  if constexpr (hmputils::is_mpi_primitive<STAGE_IN_TYPE>::value) {
    mpi_in_type = hmputils::mpi_type_of<STAGE_IN_TYPE>::value();
  } else if (mpi_type_table.find(stage_in_index) != mpi_type_table.end()) {
    mpi_in_type = mpi_type_table.at(stage_in_index);
  } else {
    throw std::invalid_argument(
        "No existing MPI type for stage function IN_TYPE");
  }

  if constexpr (hmputils::is_mpi_primitive<STAGE_OUT_TYPE>::value) {
    mpi_out_type = hmputils::mpi_type_of<STAGE_OUT_TYPE>::value();
  } else if (mpi_type_table.find(stage_out_index) != mpi_type_table.end()) {
    mpi_out_type = mpi_type_table.at(stage_out_index);
  } else {
    throw std::invalid_argument(
        "No existing MPI type for stage function OUT_TYPE");
  }

  stages.push_back(std::make_unique<Stage<STAGE_IN_TYPE, STAGE_OUT_TYPE>>(
      std::move(f), mpi_in_type, mpi_out_type, profiling_input));
  ++stage_count;
}

template <typename IN_TYPE, typename OUT_TYPE>
template <typename C_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::add_mpi_type(MPI_Datatype mpi_type) {
  const auto c_type_index = std::type_index(typeid(C_TYPE));

  if constexpr (hmputils::is_mpi_primitive<C_TYPE>::value) {
    throw std::invalid_argument(
        "Attempting to override MPI primitive type definition");
  }

  auto [iter, inserted] = mpi_type_table.insert({c_type_index, mpi_type});
  if (!inserted) {
    throw std::invalid_argument(
        "Duplicate definition of MPI non-primitive type");
  }

  MPI_Type_commit(&(iter->second));
}

template <typename IN_TYPE, typename OUT_TYPE>
std::vector<OUT_TYPE>
Pipeline<IN_TYPE, OUT_TYPE>::execute(std::vector<IN_TYPE> &data) {
  if (stage_count == 0) {
    throw std::invalid_argument("Pipeline has no defined stages");
  }

  if (stages.back()->output_type() != typeid(OUT_TYPE)) {
    throw std::invalid_argument(
        "Output type of final stage does not match pipeline output type");
  }

  if (stage_count > cluster->get_total_core_count()) {
    throw std::invalid_argument(
        "Too many stages: Stage count exceeds cluster core count");
  }

  profile_stages();
  allocate_stages();

  for (size_t i = 0; i < stages_per_node.size(); ++i) {
    printf("Node %zu: ", i);
    for (size_t j = 0; j < stages_per_node[i].size(); ++j) {
      printf("%d", stages_per_node[i][j]);
      if (j < stages_per_node[i].size() - 1) {
        printf(", "); // Add a comma between numbers, but not after the last one
      }
    }
    printf("\n"); // Newline after each node's stages are printed
  }

  std::vector<OUT_TYPE> return_data;
  return return_data;
}

template <typename IN_TYPE, typename OUT_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::profile_stages() {
  for (auto &stage : stages) {
    total_profiling_runtime += stage->profile();
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::allocate_stages() {
  std::vector<float> stage_weights;
  for (auto &stage : stages) {
    stage_weights.push_back(static_cast<float>(stage->profiling_runtime) /
                            static_cast<float>(total_profiling_runtime));
  }
  stages_per_node = distribute_tasks(stage_weights, distribution_type, cluster);
}

} // namespace hmp

#endif // HMP_PIPELINE_H_
