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
"""U2 ASR Model
Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition
(https://arxiv.org/pdf/2012.05481.pdf)
"""
import time
from typing import Dict
from typing import Optional
from typing import Tuple

import mindspore as ms

from mindspore import nn


class U2STBaseModel(nn.Cell):
    """CTC-Attention hybrid Encoder-Decoder model"""

    def __init__(self,
                 vocab_size: int,
                 encoder: TransformerEncoder,
                 st_decoder: TransformerDecoder,
                 decoder: TransformerDecoder=None,
                 ctc: CTCDecoderBase=None,
                 ctc_weight: float=0.0,
                 asr_weight: float=0.0,
                 ignore_id: int=IGNORE_ID,
                 lsm_weight: float=0.0,
                 length_normalized_loss: bool=False,
                 **kwargs):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.asr_weight = asr_weight

        self.encoder = encoder
        self.st_decoder = st_decoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss, )

    def construct(self, ):
        pass
