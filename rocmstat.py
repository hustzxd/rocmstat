import os

from colorama import Fore
from colorama import init as colorama_init
from datetime import datetime

colorama_init()


def main():
    utilizations, temperatures = get_utilization_temp()
    n_devices = len(utilizations)
    pid_user_list = get_pid_user(n_devices)
    memory_total_list, memory_used_list = get_memory()
    hostname = open("/proc/sys/kernel/hostname", "r").readline().strip()
    print(f"{hostname}\t {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \t version: {get_version()}")
    for gpu_id in range(n_devices):
        memory_used = int(memory_used_list[gpu_id] / 1024 / 1024)
        memory_total = int(memory_total_list[gpu_id] / 1024 / 1024)
        
        print(
            gpu_info_format(
                gpu_id,
                memory_used=memory_used,
                memory_total=memory_total,
                temperature=temperatures[gpu_id],
                utilization=utilizations[gpu_id],
                pid_user=pid_user_list[gpu_id],
            )
        )

def get_version():
    os.system("apt-cache show rocm-libs | grep Version > tmp")
    version = open("tmp", "r").readline().split()[-1]
    os.system("rm tmp")
    return version
    

def get_utilization_temp():
    os.system("rocm-smi |grep W | grep c >tmp")
    utilization = [int(x.split()[9][:-1]) for x in open("tmp", "r").readlines()]
    temp = [float(x.split()[1][:-1]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    return utilization, temp

def get_pid_user(n_devices):
    os.system("rocm-smi --showpidgpus>tmp")
    info = [x for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    pid_user_list = ["" for _ in range(n_devices)]
    pid = 0
    user = ""
    for line in info:
        if pid != 0:
            for i in line.split():
                pid_user_list[int(i)] += f" {user}/{pid}"
        pid = 0
        user = ""
        if "PID" in line and "DRM" in line:
            pid = int(line.split()[1])
            os.system(f"ps -u -p {pid} | grep {pid}>tmp")
            user = open("tmp", "r").readline().split()[0]
            os.system("rm tmp")
    return pid_user_list

def get_memory():
    os.system("rocm-smi -u --showmeminfo vram | grep VRAM>tmp")
    memory = [int(x.split()[-1]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    memory_used = memory[1::2] 
    memory_total = memory[0::2]
    return memory_total, memory_used

def gpu_info_format(gpu_id, memory_used, memory_total, temperature, utilization, pid_user):
    return f"{Fore.GREEN}[{gpu_id}]{Fore.RESET} | \
{Fore.LIGHTRED_EX}{temperature:2}°C{Fore.RESET} \
{Fore.LIGHTBLUE_EX}{utilization:3} %{Fore.RESET} | \
{Fore.YELLOW}{memory_used:5}{Fore.RESET} / \
{Fore.LIGHTYELLOW_EX}{memory_total:5}{Fore.RESET} MB | \
{Fore.LIGHTBLACK_EX}{pid_user}{Fore.RESET}"


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
