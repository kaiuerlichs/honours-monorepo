#include <omp.h>
#include <iostream>
#include <string>
#include <unistd.h>


int main() {
    int thread_cound = omp_get_max_threads();

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    int pid = getpid();

    std::printf("Hello from node %s with pid: %d, node thread count: %d\n", hostname, pid, thread_cound);

    return 0;
}