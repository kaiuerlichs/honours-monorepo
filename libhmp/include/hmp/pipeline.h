#ifndef HMP_PIPELINE_H_
#define HMP_PIPELINE_H_

#include "mpi.h"
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "hmp.h"
#include "hmp/mpi_type_traits.h"

namespace hmp {

struct IStage {
  virtual ~IStage() = default;
  virtual const std::type_info &input_type() const = 0;
  virtual const std::type_info &output_type() const = 0;

  MPI_Datatype mpi_input_type;
  MPI_Datatype mpi_output_type;
};

template <typename IN_TYPE, typename OUT_TYPE> struct Stage : IStage {
  std::function<OUT_TYPE(IN_TYPE)> stage_function;

  Stage(std::function<OUT_TYPE(IN_TYPE)> f, MPI_Datatype &mpi_in_type,
        MPI_Datatype &mpi_out_type)
      : stage_function(std::move(f)) {}

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

public:
  Pipeline(std::shared_ptr<MPICluster> cluster_ptr);
  ~Pipeline(){};

  template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
  void add_stage(std::function<STAGE_OUT_TYPE(STAGE_IN_TYPE)> f);

  template <typename C_TYPE> void add_mpi_type(MPI_Datatype mpi_type);

  std::vector<OUT_TYPE> execute(std::vector<IN_TYPE> &data);
};

// IMPLEMENTATION

template <typename IN_TYPE, typename OUT_TYPE>
Pipeline<IN_TYPE, OUT_TYPE>::Pipeline(std::shared_ptr<MPICluster> cluster_ptr) {
  cluster = cluster_ptr;
}

template <typename IN_TYPE, typename OUT_TYPE>
template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::add_stage(
    std::function<STAGE_OUT_TYPE(STAGE_IN_TYPE)> f) {
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

  stages.push_back(
      std::make_unique<Stage<STAGE_IN_TYPE, STAGE_OUT_TYPE>>(std::move(f), mpi_in_type, mpi_out_type));
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

  std::vector<OUT_TYPE> return_data;
  return return_data;
}

} // namespace hmp

#endif // HMP_PIPELINE_H_
