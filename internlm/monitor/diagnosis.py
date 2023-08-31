import torch
import socket
import functools
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.utils.writer import Writer
from internlm.utils.megatron_timers import megatron_timer
from internlm.utils.megatron_timers import megatron_timer as timer
from typing import Callable, Iterable, Union
from internlm.utils.logger import get_logger
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import DummyProfile
import torch.distributed as dist


logger = get_logger(__file__)


class LLMProfiler:
    def __init__(
        self,
        timer: megatron_timer,
        train_state,
        log_time_step: int = 1,
        launch_time: str = None,
        active_count: int = 1,
        do_trace_profiling: bool = False,
        trace_profiling_range: Callable = None,
        do_diagnosis: bool = False,
        writer: Writer = None,
    ) -> None:
        """
        TODO: support PP diagnosis mode.

        Args:
            timer (megatron_timer): 
            train_state (_type_): 
            log_time_step (int, optional): . Defaults to 1.
            trace_profiling_range (Callable, optional): A user-defined function with a return value of true or false, 
                used to control whether trace profiling is enabled. Defaults to None.
            launch_time (str, optional): . Defaults to None.
            active_count (int, optional): trace profiling interval. Defaults to 1.
            do_trace_profiling (bool): Whether to enable trace profiling. Defaults to False.
            do_diagnosis (bool, optional): Whether to enable runtime diagnostics. Defaults to False.
            writer (Writer, optional): . Defaults to None.
        """
        from datetime import datetime

        self.trace_profiling = do_trace_profiling
        self.train_state = train_state
        self.active_count = active_count
        self.in_diagnosis = do_diagnosis
        self.log_time_step = log_time_step
        self.writer = writer
        self.timer: megatron_timer = timer
        self.torch_profile = DummyProfile()
        # runtime time metrics.
        self.time_ckpts = [
            "batch-gen",
            "fwd",
            "bwd",
            "fwd-bwd",
            "dp_sync",
            "post_fn",
            "cal_loss",
            "sync_grad",
            "cal_norm",
            "step",
            "one-batch",
        ]

        if trace_profiling_range is None:
            trace_profiling_range = (
                lambda: gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            )

        if launch_time is None:
            launch_time = datetime.now().strftime("%H:%M:%S")

        if self.trace_profiling:
            if trace_profiling_range():
                self.torch_profile = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(skip_first=30, wait=1, warmup=1, active=1, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f"{gpc.config.JOB_NAME}/{launch_time}/traces/",
                    ),
                    with_stack=True,
                    with_modules=True,
                )
            else:
                self.torch_profile = DummyProfile()
        else:
            self.torch_profile = DummyProfile()

    def get_rank_uid(self):
        return (
            f"{socket.gethostname()}_rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"pp{gpc.get_local_rank(ParallelMode.PIPELINE)}"
        )

    def __enter__(self):
        return self

    def step(self):
        # Try increase torch trace profiler counter
        if self.train_state.step_count % self.active_count == 0:
            self.torch_profile.step()

        # Try dump timer to rb or log.
        if (self.train_state.step_count + 1) % self.log_time_step == 0:
            if self.writer:
                self.timer.write(
                    self.time_ckpts,
                    self.writer,
                    self.train_state.step_count,
                    normalizer=self.log_time_step,
                    reset=False,
                )
            if gpc.is_rank_for_log():
                self.timer.log(
                    names=self.time_ckpts, logger=logger, normalizer=self.log_time_step, reset=False
                )

        # If we are in diagnosis mode, rank 0 will gahter all rank runtime time info.
        if self.in_diagnosis:
            time_list = (self.get_rank_uid(), timer.get_all_timer_results(reset=False))
            if gpc.get_global_rank() == 0:
                all_rank_time_list = [None for _ in range(gpc.get_world_size(ParallelMode.GLOBAL))]
            else:
                all_rank_time_list = None

            dist.gather_object(time_list, all_rank_time_list, dst=0, group=gpc.get_group(ParallelMode.GLOBAL))
            if gpc.get_global_rank() == 0:
                all_times = {}
                for rank_time_info in all_rank_time_list:
                    ruid, info = rank_time_info
                    for time_tuple in info:
                        name, value = time_tuple
                        if name not in all_times:
                            all_times[name] = [(value, ruid)]
                        else:
                            all_times[name].append((value, ruid))

                for key, values in all_times.items():
                    all_times[key] = sorted(values, key=lambda x: x[0], reverse=True)
                    print(f"key: {key}, {all_times[key]}", flush=True)

                # TODO: format ouput, maybe can use BeautifulTable.
                # from beautifultable import BeautifulTable
                # table = BeautifulTable()
                # table.rows.append(["Jacob", 1, "boy"])
                

        self.timer.reset()

        # Do cuda burn test


        # Do nccl-test benchmark


    def __exit__(self, a, b, c):
        pass
