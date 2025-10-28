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
    training.n_iters = 300000  # Total training steps 
    training.snapshot_freq = 20000  # Save checkpoint every 10000 steps
    training.log_freq = 50  # Log every 50 steps
    training.eval_freq = 10000  # Evaluate every 100 steps
    training.data_dir = './data/afhq/train'
    training.batch_size = 32  # Reduced to 4 to fit in 10GB GPU memory

    # sampling
    sampling = config.sampling
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'rk45'

    # data
    data = config.data
    data.image_size = 512
    data.downsample_size = 64
    data.dataset = 'AFHQ-CAT-Pytorch'
    data.centered = True

    # model - for 64x64 images: 64->32->16->8 (4 levels max)
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = False  # Important for smaller images
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128  # Increased capacity
    model.ch_mult = (1, 2, 2, 2)  # 4 levels for 64x64: 64->32->16->8->4
    model.num_res_blocks = 4  # More blocks for better quality
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False  # Disable FIR for smaller images
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'  # Must be 'none' for 64x64
    model.progressive_input = 'none'  # Must be 'none' for 64x64
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.embedding_type = 'positional'  # Better for smaller images
    model.fourier_scale = 16
    model.conv_size = 3
    model.dropout = 0.15

    return config