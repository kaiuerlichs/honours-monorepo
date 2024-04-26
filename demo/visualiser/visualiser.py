import sys
import paramiko
from scp import SCPClient
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def create_ssh_client(server, port, user, key_file):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, username=user, key_filename=key_file)
    return client

def ssh_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    for line in iter(stdout.readline, ""):
        print(f'[CLUSTER] {line}', end="")

def download_file(ssh_client, remote_path, local_path):
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.get(remote_path, local_path)

if __name__ == "__main__":
    hostname = sys.argv[1]
    key_file = sys.argv[2]
    
    print("[LOCAL] Establishing SSH connection to cluster head node.\n")
    client = create_ssh_client(
        hostname, 
        22,
        "ubuntu",
        key_file
    )
    print("[LOCAL] Established.\n")
    
    mpiexec_command = "mpiexec -n 8 -hostfile /mnt/nfs/hosts /mnt/nfs/honours-monorepo/demo/build/mandelbrot_benchmark hmpcore 1 -2.0 0.5 -1.25 1.25 5000 5000"

    try:
        print("[LOCAL] Running mpiexec command on cluster master node.\n")
        ssh_command(client, mpiexec_command)
        print("\n[LOCAL] Remote command execution complete.\n")

        print("[LOCAL] Downloading Mandelbrot set from cluster.\n")
        download_file(client, "mandelbrot.txt", "./mandelbrot.txt")
        print("[LOCAL] Download complete.")

    finally:
        client.close

    with open("mandelbrot.txt", 'r') as file:
        first_line = file.readline().split()
        x_min, x_max = float(first_line[0]), float(first_line[1])
        y_min, y_max = float(first_line[2]), float(first_line[3])
        width, height = int(first_line[4]), int(first_line[5])
        data = np.array([list(map(int, line.split())) for line in file], dtype=int)

    log_data = np.log(np.log2(data + 1))

    norm = Normalize(vmin=np.min(data), vmax=np.max(data))

    plt.figure(num="Mandelbrot Set", figsize=(6, 6))
    plt.imshow(norm(log_data), cmap='magma_r', extent=(x_min, x_max, y_min, y_max), interpolation='nearest')
    plt.colorbar()
    plt.title('Mandelbrot Set (generated on HMP cluster)')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.show()

