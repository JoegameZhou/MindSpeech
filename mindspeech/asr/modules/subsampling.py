#  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Subsampling layer definition."""
from typing import Tuple

import torch

from mindspore import nn
from mindspore import ops
from mindspore.common import Tensor

from mindspeech.asr.modules.embedding import PositionalEncoding


class BaseSubsampling(nn.Cell):
    def __init__(self, pos_enc_class: nn.Cell = PositionalEncoding):
        super().__init__()
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Cell = PositionalEncoding):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.SequentialCell([
            nn.Dense(idim, odim),
            nn.LayerNorm(odim, epsilon=1e-5),
            nn.Dropout(keep_prob=1 - dropout_rate)])
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def construct(self, x: Tensor, x_mask: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Cell = PositionalEncoding):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(1, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU()])
        self.out = nn.SequentialCell([
            nn.Dense(odim * (((idim - 1) // 2 - 1) // 2), odim)])
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def construct(self, x: Tensor, x_mask: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """Subsample x.
        Args:
            x (Tensor): Input tensor (#batch, time, idim).
            x_mask (Tensor): Input mask (#batch, 1, time).
            offset (int): offset.
        Returns:
            Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            Tensor: positional encoding
        """
        x = ops.ExpandDims()(x, 1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(
            ops.Reshape()(
                ops.Transpose(x, (0, 2, 1, 3)), (b, t, c * f))
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Cell = PositionalEncoding):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(1, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, 3, pad_mode="valid", has_bias=True),
            nn.ReLU()])
        self.out = nn.SequentialCell([
            nn.Dense(odim * (((idim - 1) // 2 - 2) // 3), odim)])
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.right_context = 10

    def construct(self, x: Tensor, x_mask: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """Subsample x.
        Args:
            x (Tensor): Input tensor (#batch, time, idim).
            x_mask (Tensor): Input mask (#batch, 1, time).
            offset (int): offset.
        Returns:
            Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            Tensor: positional encoding
        """
        x = ops.ExpandDims()(x, 1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(
            ops.Reshape()(
                ops.Transpose(x, (0, 2, 1, 3)), (b, t, c * f))
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: nn.Cell = PositionalEncoding):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(1, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2, pad_mode="valid", has_bias=True),
            nn.ReLU()])
        self.out = nn.SequentialCell([
            nn.Dense(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)])
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def construct(self, x: Tensor, x_mask: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """Subsample x.
        Args:
            x (Tensor): Input tensor (#batch, time, idim).
            x_mask (Tensor): Input mask (#batch, 1, time).
            offset (int): offset.
        Returns:
            Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            Tensor: positional encoding
        """
        x = ops.ExpandDims()(x, 1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(
            ops.Reshape()(
                ops.Transpose(x, (0, 2, 1, 3)), (b, t, c * f))
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
