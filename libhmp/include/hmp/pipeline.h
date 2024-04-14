#ifndef HMP_PIPELINE_H_
#define HMP_PIPELINE_H_

#include "mpi.h"
#include <alloca.h>
#include <any>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
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

struct StageAllocation {
  int self = -1;
  std::vector<int> node_per_stage;
  bool participant = false;
};

struct IStage {
  virtual ~IStage() = default;
  virtual const std::type_info &input_type() const = 0;
  virtual const std::type_info &output_type() const = 0;

  virtual int profile() = 0;
  virtual void run_self(std::shared_ptr<MPICluster> cluster,
                        StageAllocation allocation, std::any data) = 0;

  MPI_Datatype input_mpi_type = MPI_DATATYPE_NULL;
  MPI_Datatype output_mpi_type = MPI_DATATYPE_NULL;

  int profiling_runtime;
};

template <typename IN_TYPE, typename OUT_TYPE> struct Stage : IStage {
  std::function<OUT_TYPE(IN_TYPE)> stage_function;

  IN_TYPE profiling_input;

  Stage(std::function<OUT_TYPE(IN_TYPE)> f, MPI_Datatype mpi_in_type,
        MPI_Datatype mpi_out_type, IN_TYPE profiling_in);

  int profile() override;
  virtual void run_self(std::shared_ptr<MPICluster> cluster,
                        StageAllocation allocation, std::any data) override;

  const std::type_info &input_type() const override { return typeid(IN_TYPE); }
  const std::type_info &output_type() const override {
    return typeid(OUT_TYPE);
  }
};

template <typename IN_TYPE, typename OUT_TYPE> class Pipeline {
private:
  std::shared_ptr<MPICluster> cluster;
  std::unordered_map<std::type_index, MPI_Datatype> mpi_type_table;

  std::vector<std::unique_ptr<IStage>> stages;
  int stage_count = 0;

  int total_profiling_runtime;

  Distribution distribution_type;

  StageAllocation allocation;

  void profile_stages();
  void allocate_stages();
  void run_stages(std::vector<IN_TYPE> &data);
  std::vector<OUT_TYPE> collect_data(int item_count);

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
                                MPI_Datatype mpi_in_type,
                                MPI_Datatype mpi_out_type,
                                IN_TYPE profiling_in) {
  stage_function = std::move(f);
  profiling_input = profiling_in;
  input_mpi_type = mpi_in_type;
  output_mpi_type = mpi_out_type;
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

  if (stage_count > cluster->get_node_count()) {
    throw std::invalid_argument(
        "Too many stages: Stage count exceeds node count");
  }

  profile_stages();

  allocate_stages();

  run_stages(data);

  std::vector<OUT_TYPE> return_data;
  if (cluster->on_master()) {
    return_data = collect_data(data.size());
  }
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
  if (cluster->on_master()) {
    std::vector<float> stage_weights;
    for (auto &stage : stages) {
      stage_weights.push_back(static_cast<float>(stage->profiling_runtime) /
                              static_cast<float>(total_profiling_runtime));
    }
    allocation.node_per_stage =
        distribute_tasks(stage_weights, distribution_type, cluster);

    allocation.self = 0;

    for (int rank = 0; rank < cluster->get_node_count(); ++rank) {
      MPI_Send(allocation.node_per_stage.data(), stage_count, MPI_INT, rank, 0,
               MPI_COMM_WORLD);
    }
  } else {
    allocation.node_per_stage.resize(stage_count);
    MPI_Recv(allocation.node_per_stage.data(), stage_count, MPI_INT, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < stage_count; ++i) {
    if (allocation.node_per_stage[i] == cluster->get_rank()) {
      allocation.self = i;
      break;
    }
  }
}

template <typename IN_TYPE, typename OUT_TYPE>
void Pipeline<IN_TYPE, OUT_TYPE>::run_stages(std::vector<IN_TYPE> &data) {
  std::cout << allocation.self << std::endl;
  std::cout << data.size() << std::endl;
  std::any any_data = std::any(data);
  stages[allocation.self]->run_self(cluster, allocation, any_data);
}

template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
void Stage<STAGE_IN_TYPE, STAGE_OUT_TYPE>::run_self(
    std::shared_ptr<MPICluster> cluster, StageAllocation allocation,
    std::any data) {
  std::cout << "HEYHEYHEHY" << std::endl;
  if (allocation.self == -1) {
    return;
  }

  std::vector<STAGE_IN_TYPE> input_data;
  int item_count;

  int threads = omp_get_max_threads();

  std::vector<std::vector<MPI_Request>> send_requests(threads);

  if (cluster->on_master()) {
    input_data = std::any_cast<std::vector<STAGE_IN_TYPE>>(data);
    item_count = input_data.size();
  }
  MPI_Bcast(&item_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int self_rank = allocation.self;
  int prev_rank = self_rank > 0 ? allocation.node_per_stage[self_rank - 1] : 0;
  int next_rank = self_rank < allocation.node_per_stage.size() - 1
                      ? allocation.node_per_stage[self_rank + 1]
                      : 0;

  if (cluster->on_master()) {
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < item_count; ++i) {
      STAGE_OUT_TYPE output = stage_function(input_data[i]);
      MPI_Request request;
      MPI_Isend(&output, 1, output_mpi_type, next_rank, i, MPI_COMM_WORLD,
                &request);
      send_requests[omp_get_thread_num()].push_back(request);
    }
  } else {
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < item_count; ++i) {
      STAGE_IN_TYPE input;
      MPI_Status status;
      MPI_Recv(&input, 1, input_mpi_type, prev_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;

      STAGE_OUT_TYPE output = stage_function(input);

      MPI_Request request;
      MPI_Isend(&output, 1, output_mpi_type, next_rank, tag, MPI_COMM_WORLD,
                &request);
      send_requests[omp_get_thread_num()].push_back(request);
    }
  }

  std::vector<MPI_Request> merged_requests;
  for (auto &reqs : send_requests) {
    merged_requests.insert(merged_requests.end(), reqs.begin(), reqs.end());
  }
  MPI_Waitall(merged_requests.size(), merged_requests.data(),
              MPI_STATUSES_IGNORE);
}

template <typename IN_TYPE, typename OUT_TYPE>
std::vector<OUT_TYPE>
Pipeline<IN_TYPE, OUT_TYPE>::collect_data(int item_count) {
  std::vector<OUT_TYPE> output_data;
  if (cluster->on_master()) {
    MPI_Datatype out_type;
    std::type_index out_index = std::type_index(typeid(OUT_TYPE));

    if constexpr (hmputils::is_mpi_primitive<OUT_TYPE>::value) {
      out_type = hmputils::mpi_type_of<OUT_TYPE>::value();
    } else if (mpi_type_table.find(out_index) != mpi_type_table.end()) {
      out_type = mpi_type_table.at(out_index);
    }

    output_data.resize(item_count);
    int source_rank = allocation.node_per_stage[stage_count - 1];

    for (int i = 0; i < output_data.size(); ++i) {
      MPI_Status status;
      OUT_TYPE output;
      MPI_Recv(&output, 1, out_type, source_rank, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status);
      int index = status.MPI_TAG;
      output_data[index] = output;
    }
  }
  return output_data;
}

} // namespace hmp

#endif // HMP_PIPELINE_H_
