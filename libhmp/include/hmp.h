#ifndef INCLUDE_HMP_H_
#define INCLUDE_HMP_H_

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <vector>

#include "mpi.h"
#include "omp.h"

namespace hmp {

// Stores information about a given node in an MPI cluster
class Node {
private:
  int process_rank;
  int processor_frequency;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int core_count;
  bool os_linux;

  void load_rank();
  void load_os_type();
  void load_processor_info();
  std::string read_cpu_info();
  int get_processor_frequency();

public:
  Node();
  ~Node(){};

  void initialise();

  void print_info();
  bool is_master();
  bool is_linux();
  int get_rank();
  int get_frequency();
  int get_core_count();

  // Returns a custom MPI type used for sending node info objects between nodes
  MPI_Datatype get_mpi_type();
};

// Stores information about the MPI cluster global context
class MPICluster {
private:
  std::shared_ptr<Node> self;
  std::vector<std::shared_ptr<Node>> nodes;

  int node_count;
  int core_count;
  bool os_linux = true;

  void add_node(std::shared_ptr<Node>);

  void send_node();
  void receive_node();

public:
  // Constructor/destructor
  MPICluster();
  ~MPICluster();

  bool on_master();
  bool is_linux();
  int get_node_count();
  int get_total_core_count();
  int get_local_core_count();
  std::vector<int> get_cores_per_node();
  std::vector<int> get_frequency_per_node();
  int get_rank();
  void print_info();
};

// IMPLEMENTATION

inline Node::Node() {}

inline void Node::initialise() {
  load_rank();
  load_os_type();
  load_processor_info();
}

inline void Node::load_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_rank = rank;
}

inline void Node::load_processor_info() {
  int name_length;
  MPI_Get_processor_name(processor_name, &name_length);

  core_count = omp_get_max_threads();

  if (is_linux()) {
    processor_frequency = get_processor_frequency();
  } else {
    processor_frequency = 0;
  }
}

inline void Node::load_os_type() {
// TODO: This implementation evaluates OS type on compile time, which might need
// to be changed to runtime
#ifdef __linux__
  os_linux = true;
#else
  os_linux = false;
#endif
}

inline std::string Node::read_cpu_info() {
  std::ifstream proc_cpuinfo("/proc/cpuinfo");

  if (!proc_cpuinfo.is_open()) {
    // TODO: Handle file open error
  }

  std::stringstream buffer;
  buffer << proc_cpuinfo.rdbuf();

  return buffer.str();
}

inline int Node::get_processor_frequency() {
  std::string proc_cpuinfo = read_cpu_info();
  std::regex regex_pattern("cpu MHz\\s*:\\s*([\\d.]+)");
  std::smatch regex_match;

  if (std::regex_search(proc_cpuinfo, regex_match, regex_pattern)) {
    float frequency_raw = std::stof(regex_match[1].str());
    return static_cast<int>(std::round(frequency_raw));
  }

  return 0;
}

inline void Node::print_info() {
  std::printf("Node information for %s: Rank %i, Cores %i\n", processor_name,
              process_rank, core_count);
}

inline bool Node::is_master() { return process_rank == 0; }

inline bool Node::is_linux() { return os_linux; }

inline int Node::get_rank() { return process_rank; }

inline int Node::get_frequency() { return processor_frequency; }

inline int Node::get_core_count() { return core_count; }

inline MPI_Datatype Node::get_mpi_type() {
  const int nitems = 5; // Adjusted for the five items we're sending
  int blocklengths[5] = {1, 1, MPI_MAX_PROCESSOR_NAME, 1,
                         1}; // Added core_count and os_linux (as int)

  MPI_Aint displacements[5];
  displacements[0] = offsetof(Node, process_rank);
  displacements[1] = offsetof(
      Node, processor_frequency); // Adjusted to include processor_frequency
  displacements[2] = offsetof(Node, processor_name);
  displacements[3] = offsetof(Node, core_count);
  displacements[4] = offsetof(Node, os_linux); // Adjusted to include os_linux

  MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_CHAR, MPI_INT,
                           MPI_INT}; // os_linux treated as MPI_INT

  MPI_Datatype mpi_node_type;
  MPI_Type_create_struct(nitems, blocklengths, displacements, types,
                         &mpi_node_type);
  MPI_Type_commit(&mpi_node_type);
  return mpi_node_type;
}

inline MPICluster::MPICluster() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &node_count);

  self = std::make_shared<Node>();
  self->initialise();

  MPI_Datatype mpi_node_type = self->get_mpi_type();

  if (self->is_master()) {
    add_node(self);
    for (int i = 1; i < node_count; ++i) {
      auto node = std::make_shared<Node>();
      MPI_Recv(node.get(), 1, mpi_node_type, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      add_node(node);
    }
    print_info();
  } else {
    MPI_Send(self.get(), 1, mpi_node_type, 0, 0, MPI_COMM_WORLD);
  }
}

inline bool MPICluster::on_master() { return self->is_master(); }

inline int MPICluster::get_node_count() { return node_count; }

inline int MPICluster::get_total_core_count() { return core_count; }

inline int MPICluster::get_local_core_count() { return self->get_core_count(); }

inline int MPICluster::get_rank() { return self->get_rank(); }

inline void MPICluster::add_node(std::shared_ptr<Node> node_ptr) {
  core_count += node_ptr->get_core_count();
  nodes.push_back(node_ptr);

  if (!node_ptr->is_linux()) {
    os_linux = false;
  }
}

inline std::vector<int> MPICluster::get_cores_per_node() {
  std::vector<int> cores;
  cores.resize(node_count);
  for (auto node : nodes) {
    cores[node->get_rank()] = node->get_core_count();
  }
  return cores;
}

inline std::vector<int> MPICluster::get_frequency_per_node() {
  std::vector<int> frequencies;
  frequencies.resize(node_count);
  for (auto node : nodes) {
    frequencies[node->get_rank()] = node->get_frequency();
  }
  return frequencies;
}

inline bool MPICluster::is_linux() { return os_linux; }

inline void MPICluster::print_info() {
  for (auto node_info : nodes) {
    node_info->print_info();
  }
}

inline MPICluster::~MPICluster() { MPI_Finalize(); }

} // namespace hmp

#endif // INCLUDE_HMP_H_