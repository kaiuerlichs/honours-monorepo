#ifndef HMP_PIPELINE_H_
#define HMP_PIPELINE_H_

#include "mpi.h"
#include <any>
#include <chrono>
#include <cmath>
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
  virtual std::any run_self(int stage_number, int threads, int rank,
                            int prev_rank, int next_rank, std::any &data) = 0;

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

  virtual std::any run_self(int stage_number, int threads, int rank,
                            int prev_rank, int next_rank,
                            std::any &data) override;

  std::vector<MPI_Request> send_requests;

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
  int local_stage;
  std::vector<int> node_per_stage;

  void profile_stages();
  void allocate_stages();
  std::vector<OUT_TYPE> run_stages(std::vector<IN_TYPE> &data);

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
  printf("STAGE COUNT %d", stage_count);
  profile_stages();

  allocate_stages();

  // std::vector<OUT_TYPE> return_data = run_stages(data);
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
  if (cluster->on_master()) {
    std::vector<float> stage_weights;
    for (auto &stage : stages) {
      stage_weights.push_back(static_cast<float>(stage->profiling_runtime) /
                              static_cast<float>(total_profiling_runtime));
    }
    node_per_stage =
        distribute_tasks(stage_weights, distribution_type, cluster);

    local_stage = 0;

    for (int stage = 1; stage < node_per_stage.size(); ++stage) {
      MPI_Send(&stage, 1, MPI_INT, node_per_stage[stage], 0, MPI_COMM_WORLD);
      MPI_Send(node_per_stage.data(), stage_count, MPI_INT,
               node_per_stage[stage], 0, MPI_COMM_WORLD);
    }
  } else {
    node_per_stage.resize(stage_count);
    MPI_Recv(&local_stage, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(node_per_stage.data(), stage_count, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  printf("Test");
}

template <typename IN_TYPE, typename OUT_TYPE>
std::vector<OUT_TYPE>
Pipeline<IN_TYPE, OUT_TYPE>::run_stages(std::vector<IN_TYPE> &data) {
  int threads = cluster->get_local_core_count();
  int rank = cluster->get_rank();
  int prev_rank = (rank == 0) ? 0 : node_per_stage[local_stage - 1];
  int next_rank =
      (local_stage == stage_count - 1) ? 0 : node_per_stage[local_stage + 1];
  std::any any_input = std::any(data);
  std::any output_data = stages[local_stage]->run_self(
      local_stage, threads, rank, prev_rank, next_rank, any_input);

  return std::any_cast<std::vector<OUT_TYPE>>(output_data);
}

template <typename STAGE_IN_TYPE, typename STAGE_OUT_TYPE>
std::any
Stage<STAGE_IN_TYPE, STAGE_OUT_TYPE>::run_self(int stage_number, int threads,
                                               int rank, int prev_rank,
                                               int next_rank, std::any &data) {
  int expected_seq = 0;
  bool terminate = false;

  std::vector<STAGE_IN_TYPE> input_data;
  std::vector<STAGE_OUT_TYPE> output_data;

  if (rank == 0) {
    input_data = std::any_cast<const std::vector<STAGE_IN_TYPE>>(data);
    output_data.resize(input_data.size());
  }

#pragma omp parallel num_threads(threads)
  {
    std::vector<MPI_Request> send_requests;
    STAGE_IN_TYPE item_buffer;
    while (!terminate) {
#pragma omp critical
      {
        int current_seq = expected_seq;
        ++expected_seq;

        bool local_terminate = false;
        if (rank == 0) {
          if (current_seq >= input_data.size()) {
            local_terminate = true;
            MPI_Send(&profiling_input, 0, output_mpi_type, next_rank,
                     MPI_TAG_UB, MPI_COMM_WORLD);
          } else {
            item_buffer = input_data[current_seq];
          }
        } else {
          bool message_received = false;
          while (!message_received) {
            int flag = 0;
            MPI_Status status;

            MPI_Iprobe(prev_rank, current_seq, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
              message_received = true;
              continue;
            }

            MPI_Iprobe(prev_rank, -1, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
              local_terminate = true;
              MPI_Recv(&item_buffer, 0, input_mpi_type, 0, MPI_TAG_UB,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Send(&profiling_input, 0, output_mpi_type, next_rank,
                       MPI_TAG_UB, MPI_COMM_WORLD);
              message_received = true;
              continue;
            }
          }
        }

        if (local_terminate) {
          terminate = true;
        } else {
          if (rank != 0) {
            MPI_Recv(&item_buffer, 1, input_mpi_type, prev_rank, current_seq,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          STAGE_OUT_TYPE return_value = stage_function(item_buffer);

          MPI_Request send_request;
          MPI_Isend(&return_value, 1, output_mpi_type, next_rank, current_seq,
                    MPI_COMM_WORLD, &send_request);
          send_requests.push_back(send_request);
        }
      }
    }

    if (rank == 0) {
      STAGE_OUT_TYPE received_item;
      MPI_Status status;
      int num_items_received = 0;
      while (num_items_received < input_data.size()) {
        MPI_Recv(&received_item, 1, output_mpi_type, MPI_ANY_SOURCE,
                 MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG >= 0) {
          output_data[status.MPI_TAG] = received_item;
          num_items_received++;
        } else {
          break;
        }
      }
    }

    MPI_Waitall(send_requests.size(), send_requests.data(),
                MPI_STATUSES_IGNORE);
  }

  return std::any(output_data);
}

} // namespace hmp

#endif // HMP_PIPELINE_H_
