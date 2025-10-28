# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Training Rectified Flow on AFHQ-CAT."""

from configs.default_lsun_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'rectified_flow'
    training.continuous = False
    training.reduce_mean = True
    training.n_iters = 100000  # Total training steps (default: 2400001)
    training.snapshot_freq = 10000  # Save checkpoint every 10000 steps
    training.log_freq = 50  # Log every 50 steps
    training.eval_freq = 100  # Evaluate every 100 steps
    training.data_dir = './data/afhq/train'
    training.batch_size = 64  # Reduced to 4 to fit in 10GB GPU memory

    # sampling
    sampling = config.sampling
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'rk45'

    # data
    data = config.data
    data.dataset = 'AFHQ-CAT-Pytorch'
    data.centered = True

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 64  # Must be divisible by GroupNorm groups
    model.ch_mult = (1, 2, 2, 2, 2, 2, 2)  # For 256x256: 256->128->64->32->16->8->4
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    return config
