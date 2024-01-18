import os

import pyamdgpuinfo

from colorama import Fore
from colorama import init as colorama_init

colorama_init()


def main():
    n_devices = pyamdgpuinfo.detect_gpus()
    if n_devices > 0:
        utilizations = get_utilization()
        pids, users = get_pid_user(n_devices)
        for gpu_id in range(n_devices):
            gpu_device = pyamdgpuinfo.get_gpu(gpu_id)
            temperature = gpu_device.query_temperature()
            memory_usage = int(gpu_device.query_vram_usage() / 1024 / 1024)
            memory_max = int(gpu_device.memory_info["vram_size"] / 1024 / 1024)
            print(
                gpu_info_format(
                    gpu_id,
                    memory_usage=memory_usage,
                    memory_max=memory_max,
                    temperature=temperature,
                    utilization=utilizations[gpu_id],
                    pid=pids[gpu_id],
                    user=users[gpu_id]
                )
            )
    else:
        print("There is no AMD GPUs")


def get_utilization():
    os.system("rocm-smi |grep W | grep c >tmp")
    utilization = [int(x.split()[9][:-1]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    return utilization

def get_pid_user(n_devices):
    os.system("rocm-smi --showpidgpus>tmp")
    info = [x for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    pids = ["" for _ in range(n_devices)]
    users = ["" for _ in range(n_devices)]
    pid = 0
    user = ""
    for line in info:
        if pid != 0:
            for i in line.split():
                pids[int(i)] = pid
                users[int(i)] = user
        pid = 0
        user = ""
        if "PID" in line and "DRM" in line:
            pid = int(line.split()[1])
            os.system(f"ps -u -p {pid} | grep {pid}>tmp")
            user = open("tmp", "r").readline().split()[0]
            os.system("rm tmp")
    return pids, users

def gpu_info_format(gpu_id, memory_usage, memory_max, temperature, utilization, pid, user):
    return f"{Fore.GREEN}[{gpu_id}]{Fore.RESET} | \
{Fore.LIGHTRED_EX}{temperature:2}°C{Fore.RESET} \
{Fore.LIGHTBLUE_EX}{utilization:3} %{Fore.RESET} | \
{Fore.YELLOW}{memory_usage:5}{Fore.RESET} / \
{Fore.LIGHTYELLOW_EX}{memory_max:5}{Fore.RESET} MB | \
{Fore.LIGHTBLACK_EX}{user}{Fore.RESET}/{pid}"


# [0] Tesla V100-SXM2-32GB | 31°C,   0 % |  6969 / 32768 MB | xiandong/3167205(6966M)


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    # TODO; if no gpu, return None
    try:
        visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        memory_visible = []
        for i in visible_gpu.split(","):
            memory_visible.append(memory_available[int(i)])
        return np.argmax(memory_visible)
    except KeyError:
        return np.argmax(memory_available)



if __name__ == "__main__":
    main()
