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

from multiprocessing import Queue
from typing import Any, Callable

from pytorch_lightning.accelerators.tpu import _inner_f as new_inner_f
from pytorch_lightning.accelerators.tpu import _multi_process as new_pl_multi_process
from pytorch_lightning.accelerators.tpu import xla_available as new_xla_available
from pytorch_lightning.utilities import rank_zero_deprecation  # TODO(lite): update to lightning_lite.utilities


def inner_f(queue: Queue, func: Callable, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.xla_device.inner_f` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. This class is internal but you can copy over its implementation."
    )
    return new_inner_f(queue, func, *args, **kwargs)


def pl_multi_process(func: Callable) -> Callable:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.xla_device.pl_multi_process` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. This class is internal but you can copy over its implementation."
    )
    return new_pl_multi_process(func)


class XLADeviceUtils:
    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.xla_device.XLADeviceUtils` has been deprecated in v1.8.0 and will be"
            " removed in v1.10.0. This class is internal but you can copy over its implementation."
        )

    @staticmethod
    def xla_available() -> bool:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.xla_device.XLADeviceUtils.xla_available` has been deprecated in v1.8.0 and"
            " will be removed in v1.10.0. Please use `pytorch_lightning.accelerators.tpu.xla_available()`"
            " instead."
        )
        return new_xla_available()

    @staticmethod
    def tpu_device_exists() -> bool:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.xla_device.XLADeviceUtils.tpu_device_exists` has been deprecated in v1.8.0"
            " and will be removed in v1.10.0. Please use `pytorch_lightning.accelerators.TPUAccelerator.is_available()`"
            " instead."
        )
        from pytorch_lightning.accelerators.tpu import TPUAccelerator

        return TPUAccelerator.is_available()
