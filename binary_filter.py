import argparse
import math
import os
import time

import torch
import torch.distributed as dist

DEVICES_PER_NODE = 8
DEVICES_STEP = 8


def get_master_node():
    import subprocess

    if os.getenv("SLURM_JOB_ID") is None:
        raise RuntimeError("get_master_node can only used in Slurm launch!")
    result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
    result = result.decode("utf8").strip()
    return result


class ProcessGroupForBinaryFilter:
    def __init__(self, launcher) -> None:
        self.initialize_distributed_env(launcher)
        self.binary_filter_group = []
        self.binary_filter_group_ranks = []
        self.init_binary_filter()
        self.buffer = torch.ones((256 * 1024 * 1024), dtype=torch.float32).cuda(device=f"cuda:{self.rank % 8}")

    def init_binary_filter(self):
        binary_group_num = int(math.log(self.world_size // DEVICES_STEP, 2))
        ranks_num_in_group = DEVICES_STEP
        if self.rank == 0:
            print(f"binary_group_num : {binary_group_num}", flush=True)
        for _ in range(binary_group_num):
            single_binary_group_num = self.world_size // ranks_num_in_group
            for gid in range(single_binary_group_num):
                ranks = [gid * ranks_num_in_group + j for j in range(ranks_num_in_group)]
                new_group = dist.new_group(ranks)
                if gid == self.rank // ranks_num_in_group:
                    self.binary_filter_group.append(new_group)
                    self.binary_filter_group_ranks.append(ranks)
            ranks_num_in_group *= 2
            # if self.rank == 0:
        print(f"RANK: {self.rank} {self.binary_filter_group_ranks}", flush=True)

    def initialize_distributed_env(self, launcher):
        if launcher == "torch":
            self.launch_from_torch()
        elif launcher == "slurm":
            self.launch_from_slurm()
        else:
            assert launcher in ["slurm", "torch"], "launcher only support slurm or torch"

    def do_all_reduce(self):
        count = 0
        for group in self.binary_filter_group:
            # dist.barrier(group=group)
            dist.barrier(group=group)
            s = time.time()
            dist.all_reduce(self.buffer, group=group)
            torch.cuda.synchronize()
            a_s = time.time() - s
            if self.rank % DEVICES_STEP:
                print(f"No:{count}, rank: {self.rank} all-reduce use time : {a_s:.3f} s")
            count += 1
            # dist.barrier(group=group)
            dist.barrier(group=group)

    def launch(self):
        init_method = f"tcp://[{self.host}]:{self.port}"
        dist.init_process_group(rank=self.rank, world_size=self.world_size, backend="nccl", init_method=init_method)
        # set cuda device
        if torch.cuda.is_available():
            # if local rank is not given, calculate automatically
            torch.cuda.set_device(self.rank % DEVICES_PER_NODE)  #

    def launch_from_slurm(self):
        self.host = get_master_node()
        self.port = 9988

        try:
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ["SLURM_NPROCS"])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the SLURM environment")

        self.launch()

    def launch_from_torch(self):
        try:
            self.rank = int(os.environ["RANK"])
            # local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.host = os.environ["MASTER_ADDR"]
            self.port = int(os.environ["MASTER_PORT"])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the torch environment")

        self.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List the content of a folder")  # 2）创建parser
    parser.add_argument("--launcher", type=str, help="the path to list")  # 3）向parse添加位置变量和可选变量
    args = parser.parse_args()  # 4）运行 parser.parse_args()，获得Namespace object
    filter = ProcessGroupForBinaryFilter(args.launcher)
    filter.do_all_reduce()
