import sys
import time
import subprocess

def process_command_line_args():
    if len(sys.argv) < 6:
        print("Usage: python blackscholes_runner.py <version_string> <nodes> <runs> <input file> <output file> <host file>")
        sys.exit(1)

    version = sys.argv[1]
    node_count = int(sys.argv[2])
    runs = int(sys.argv[3]) 
    input_file = sys.argv[4]
    output_file = sys.argv[5]

    host_file = ""
    if len(sys.argv) == 7:
        host_file = sys.argv[6]

    return version, node_count, runs, input_file, output_file, host_file

if __name__ == "__main__":
    version, node_count, runs, input_file, output_file, host_file = process_command_line_args()

    executable_prefix = ""
    if version in ["openmpi", "hmpcore", "hmpfreq"]:
        executable_prefix = f'mpiexec -n {node_count} -hosts {host_file}'
   
    executable_command = f'{executable_prefix}./blackscholes_benchmark {version} {input_file} {output_file}'

    print(f'Running Blackscholes benchmark ({version}) using command:')
    print(f'{executable_command} \n')

    time_in_ms = []
    
    for iter in range(int(runs)): 
        start_time = time.perf_counter()
        subprocess.run(executable_command, shell=True)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        time_in_ms.append(execution_time_ms)
        print(f"Run {iter+1}: Execution time: {execution_time_ms:.2f} ms")

    average_execution_time = sum(time_in_ms) / len(time_in_ms)
    print(f"\nAverage execution time over {runs} runs: {average_execution_time:.2f} ms")

