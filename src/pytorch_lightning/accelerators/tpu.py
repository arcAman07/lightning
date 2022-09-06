# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import queue as q
import traceback
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from lightning_utilities.core.imports import RequirementCache

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import device_parser


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(f"Cannot instantiate the `{type(self).__name__}`: {_XLA_AVAILABLE!s}")
        super().__init__(*args, **kwargs)

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for the given TPU device.

        Args:
            device: TPU device for which to get stats

        Returns:
            A dictionary mapping the metrics (free memory and peak memory) to their values.
        """
        import torch_xla.core.xla_model as xm

        memory_info = xm.get_memory_info(device)
        free_memory = memory_info["kb_free"]
        peak_memory = memory_info["kb_total"] - free_memory
        device_stats = {
            "avg. free memory (MB)": free_memory,
            "avg. peak memory (MB)": peak_memory,
        }
        return device_stats

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[Union[int, List[int]]]:
        """Accelerator device parsing logic."""
        return device_parser.parse_tpu_cores(devices)

    @staticmethod
    def get_parallel_devices(devices: Union[int, List[int]]) -> List[int]:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, int):
            return list(range(devices))
        return devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 8

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        # check `xla_available` again to avoid launching processes
        return xla_available() and _is_device_tpu()

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "tpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )


# define TPU availability timeout in seconds
TPU_CHECK_TIMEOUT = 60


def _inner_f(queue: Queue, func: Callable, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
    try:
        queue.put(func(*args, **kwargs))
    except Exception:
        traceback.print_exc()
        queue.put(None)


def _multi_process(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[bool, Any]:
        queue: Queue = Queue()
        proc = Process(target=_inner_f, args=(queue, func, *args), kwargs=kwargs)
        proc.start()
        proc.join(TPU_CHECK_TIMEOUT)
        try:
            return queue.get_nowait()
        except q.Empty:
            traceback.print_exc()
            return False

    return wrapper


@_multi_process
def _is_device_tpu() -> bool:
    """Check if TPU devices are available. Runs XLA device check within a separate process.

    Return:
        A boolean value indicating if TPU devices are available
    """
    if not xla_available():
        return False
    import torch_xla.core.xla_model as xm

    # For the TPU Pod training process, for example, if we have
    # TPU v3-32 with 4 VMs, the world size would be 4 and as
    # we would have to use `torch_xla.distributed.xla_dist` for
    # multiple VMs and TPU_CONFIG won't be available, running
    # `xm.get_xla_supported_devices("TPU")` won't be possible.
    return (xm.xrt_world_size() > 1) or bool(xm.get_xla_supported_devices("TPU"))


_XLA_AVAILABLE = RequirementCache("torch_xla")


def xla_available() -> bool:
    """Check if XLA library is installed.

    Return:
        A boolean value indicating if a XLA is installed
    """
    return bool(_XLA_AVAILABLE)
