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
from abc import ABC
from typing import Any

from lightning_lite.plugins.io.torch_plugin import TorchCheckpointIO as NewTorchCheckpointIO
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class TorchCheckpointIO(NewTorchCheckpointIO, ABC):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints
    respectively, common for most use cases.

    .. deprecated:: v1.8.0     This plugin has been deprecated in v1.8.0 and will be removed in v1.10.0.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "TorchCheckpointIO":
        rank_zero_deprecation(
            "`pytorch_lightning.plugins.io.TorchCheckpointIO` has been deprecated in v1.8.0 and will be"
            " removed in v1.10.0. Please use `lightning_lite.plugins.io.TorchCheckpointIO` instead."
        )
        return super().__new__(cls, *args, **kwargs)
