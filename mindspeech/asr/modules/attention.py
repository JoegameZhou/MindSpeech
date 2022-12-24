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
"""Multi-Head Attention layer definition."""
import math
from typing import Tuple

import mindspore as ms
import torch

from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops import Ones


class MultiHeadedAttention(nn.Cell):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Dense(n_feat, n_feat)
        self.linear_k = nn.Dense(n_feat, n_feat)
        self.linear_v = nn.Dense(n_feat, n_feat)
        self.linear_out = nn.Dense(n_feat, n_feat)
        self.dropout = nn.Dropout(keep_prob=1-dropout_rate)

    def forward_qkv(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Transform query, key and value.
        Args:
            query (Tensor): Query tensor (#batch, time1, size).
            key (Tensor): Key tensor (#batch, time2, size).
            value (Tensor): Value tensor (#batch, time2, size).
        Returns:
            Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]
        q = ops.Reshape()(self.linear_q(query), (n_batch, -1, self.h, self.d_k))
        k = ops.Reshape()(self.linear_k(key), (n_batch, -1, self.h, self.d_k))
        v = ops.Reshape()(self.linear_v(value), (n_batch, -1, self.h, self.d_k))
        q = ops.Transpose(q, (0, 2, 1, 3))
        k = ops.Transpose(k, (0, 2, 1, 3))
        v = ops.Transpose(v, (0, 2, 1, 3))

        return q, k, v

    def forward_attention(
            self, value: Tensor, scores: Tensor, mask: Tensor = Ones()((0, 0, 0), mstype.bool_)) -> Tensor:
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """