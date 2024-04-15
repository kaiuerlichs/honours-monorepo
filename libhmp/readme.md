# libhmp: A hybrid mulitprocessing pattern library

HMP is a high-perforance computing (HPC) library implemented in C++20 targeting hybrid memory architecture systems. It provides high-level parallel pattern implementations to support HPC tasks.

HMP is built to run on MPI clusters and optimises the use of compute resources on cluster nodes by employing both distributed and shared memory parallel techniques.

The library implements multiple parallel patterns, such as Map and Pipeline, which provide ready-to-use parallel structures. Developers simply develop their algorithms and application source code, and HMP handles the parallelisation for them!


## Authors

This library was developed by Kai Uerlichs as part of their final year Honours project at the University of Dundee, Scotland.


## Technologies

HMP is built using modern C++20, making use of its powerful template metaprogramming feautures. 

For distributed memory parallelism, MPI is used to facilitate communication between cluster nodes. On a shared memory level, the work is further parellelised using OpenMP.

## Installation

HMP is a header-only library and therefore does not need to be installed to your system. All you need to do is add the header files from the include directory into your project structure and include them in your source code or header files.

To use HMP in your application, simply add the libhmp folder (/or the includes directory) to your project structure. If you follow the [Pitchfork layout](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs) for C++ projects, you may wish to place the folder in your `external` directory.

You will also need to make sure that you have the required dependencies installed on ALL of your cluster nodes:

- [OpenMP / compatible compiler](https://www.openmp.org/resources/openmp-compilers-tools/)
- An MPI implementation, such as [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)

HMP is built to be used on MPI clusters. If you need help setting up an MPI cluster, you might want to consult [this tutorial](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/) (external link).
## Hello cluster!

To get started, simply copy the following code into your main function and compile the program using an MPI-enabled compiler (such as mpicc).

```cpp
auto cluster = std::make_shared<hmp::MPICluster>();
cluster->print_info()
```

You can then run this program on your cluster using mpiexec:

```bash
mpiexec -n <number of nodes> -hostfile /path/to/hostfile /path/to/program/binary
```

## Parallel patterns

Currently, the following parallel patterns are available in HMP:

- Map
- Pipeline (of Farms/Maps)

The code snippets provide a quick start guide for both patterns:

### Map
```cpp
auto cluster = std::make_shared<hmp::MPICluster>();
std::vector<int> data;

if (cluster->on_master())
        data = generate_test_data();

auto map = std::make_unique<hmp::Map<int, int>>(
    cluster, 
    hmp::Distribution::CORE_FREQUENCY);

map->set_map_function(map_function);

std::vector<int> return_data = map->execute(data);

if (cluster->on_master())
    handle_output_data(return_data);
```

### Pipeline
```cpp
auto cluster = std::make_shared<hmp::MPICluster>();
std::vector<int> data;

if (cluster->on_master()) 
    data = generate_test_data();

auto pipeline = std::make_unique<hmp::Pipeline<int, int>>(
    cluster, 
    hmp::Distribution::CORE_FREQUENCY);

pipeline->add_stage<int, int>(a_stage_function, sample_data);
pipeline->add_stage<int, int>(another_function, more_sample_data);

std::vector<int> out = pipeline->execute(data);

if(cluster->on_master())
    handle_output_data(out);
```
## API Documentation

### Class: `MPICluster`

**Description**: Manages the overall MPI cluster environment, facilitating node management, communication, and synchronization among the cluster nodes.

| Name                               | Description                                                              |
|------------------------------------|--------------------------------------------------------------------------|
| `MPICluster()`                     | Initializes the cluster and MPI environment with multithreading support.    |
| `~MPICluster()`                    | Cleans up the MPI environment when the cluster object is destroyed.      |
| `void print_info()`                | Prints detailed information about all nodes within the MPI cluster.      |
| `bool on_master()`                 | Checks if the current node is the master node.                           |
| `bool is_linux()`                  | Checks if all nodes in the cluster are running Linux.                    |
| `int get_node_count()`             | Returns the total number of nodes in the cluster.                        |
| `int get_total_core_count()`       | Returns the sum of cores across all nodes.                               |
| `int get_local_core_count()`       | Returns the number of cores on the local node.                           |
| `std::vector<int> get_cores_per_node()` | Returns a vector listing the number of cores for each node.        |
| `std::vector<int> get_frequency_per_node()` | Returns a vector listing the processor frequency for each node. |

### Template Class: `Map<IN_TYPE, OUT_TYPE>`

**Description**: Processes a 1-dimensional dataset using the Map pattern by applying a specified function to each element in parallel across the MPI cluster.

| Name                                        | Description                                                           |
|---------------------------------------------|-----------------------------------------------------------------------|
| `Map(std::shared_ptr<MPICluster> cluster, Distribution distribution)` | Constructor initializes the Map with a given MPI cluster and a distribution strategy. |
| `void set_map_function(std::function<OUT_TYPE(IN_TYPE)> f)` | Sets the function that will map each input to an output.            |
| `std::vector<OUT_TYPE> execute(std::vector<IN_TYPE>& data)` | Distributes the data across the cluster and executes the map function over it. |
| `void set_mpi_in_type(MPI_Datatype in_type)` | Sets the MPI datatype for the input type. Only required if MPI type is not primitive.                            |
| `void set_mpi_out_type(MPI_Datatype out_type)` | Sets the MPI datatype for the output type. Only required if MPI type is not primitive.                        |

### Template Class: `Pipeline<PIPE_IN_TYPE, PIPE_OUT_TYPE>`

**Description**: Processes a 1-dimensional dataset by applying a series of data transformations using the Pipeline pattern, running each stage in parallel across the MPI cluster.

**Notes**: Each node will take exactly one stage, until all stages have been assigned. The initial stage will always run on the master node. 

| Name                                     | Description                                                     |
|------------------------------------------|-----------------------------------------------------------------|
| `Pipeline(std::shared_ptr<MPICluster> cluster, Distribution distribution)` | Initializes a pipeline with the specified cluster and distribution strategy. |
| `void add_stage<IN, OUT>(std::function<OUT(IN)> f, IN profiling_input)` | Adds a stage to the pipeline with a function and profiling input. |
| `void add_mpi_type<C_TYPE>(MPI_Datatype mpi_type)` | Registers an MPI datatype for non-primitive types used in the pipeline stages. |
| `std::vector<PIPE_OUT_TYPE> execute(std::vector<PIPE_IN_TYPE>& data)` | Profiles pipeline stages for runtime, then executes them in parallel across the cluster. |


### Enum: `Distribution`

**Description**: Defines possible methods for distributing workloads across the MPI cluster.

| Name             | Description                                              |
|------------------|----------------------------------------------------------|
| `CORE_COUNT`     | Distributes items based on the number of cores in each node.   |
| `CORE_FREQUENCY` | Distributes items based on the number of cores scaled by the processor frequency of each node. |
