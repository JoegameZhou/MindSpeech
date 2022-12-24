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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch

import numpy as np
import mindspore as ms


def main():
    d_model = 10
    max_len = 80

    # pe = torch.zeros(max_len, d_model)
    # position = torch.arange(0, max_len,
    #                         dtype=torch.float32).unsqueeze(1)
    # div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    # pe[:, 0::2] = torch.sin(position * div_term)
    # pe[:, 1::2] = torch.cos(position * div_term)
    # pe = pe.unsqueeze(0)

    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    print(pe.shape, flush=True)
    print(pe, flush=True)


if __name__ == "__main__":
    main()
