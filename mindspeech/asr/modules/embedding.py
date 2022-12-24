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
"""Positonal Encoding Module."""
import math
from typing import Tuple

import numpy as np

from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common import Tensor


class PositionalEncoding(nn.Cell):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(keep_prob=1-dropout_rate)
        self.max_len = max_len

        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32).reshape((-1, 1))
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = Tensor(pe.reshape((1, max_len, d_model)), dtype=mstype.float32)

    def construct(self, x: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        pos_emb = self.position_encoding(offset, x.shape[1], False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, apply_dropout: bool = True) -> Tensor:
        """ For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
            apply_dropout(bool): whether apply dropout on pos_emb
        Returns:
            Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        pos_emb = self.pe[:, offset:offset + size]
        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def construct(self, x: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Compute positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
            offset(int): offset
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
            Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.shape[1], False)
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(nn.Cell):
    """ No position encoding
    """
    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)

    def construct(self, x: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = ops.Zeros()((1, x.shape[1], self.d_model))
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: int, size: int) -> Tensor:
        return ops.Zeros()((1, size, self.d_model))
