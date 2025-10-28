# configs/flow_matching_mnist.py
import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # data configuration
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'MNIST'
    data.image_size = 32
    data.num_channels = 1
    data.centered = True
    data.random_flip = False
    data.uniform_dequantization = False
    
    # model configuration
    config.model = model = ml_collections.ConfigDict()
    model.name = 'ncsnpp'
    model.ema_rate = 0.9999
    
    # NCSN++ architecture parameters
    model.scale_by_sigma = False
    model.sigma_max = 1.0
    model.sigma_min = 0.01
    model.num_scales = 1000
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.dropout = 0.1
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.embedding_type = 'fourier'
    model.fourier_scale = 16.0
    model.conv_size = 3
    
    model.nonlinearity = 'swish'
    model.normalization = 'GroupNorm'
    model.num_groups = 32
    
    # other parameters
    model.continuous = True
    model.reduce_mean = False
    model.centered = True
    
    # optimization configuration
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 0.0
    optim.grad_clip = 1.0
    optim.warmup = 5000
    
    # training configuration
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128
    training.n_epochs = 50
    training.snapshot_freq = 10
    training.log_freq = 100
    training.eval_freq = 5
    
    # these are the key parameters for NCSN++
    training.continuous = True  # 
    training.sde = 'rectified_flow'  # or set to other SDE types
    training.reduce_mean = False
    training.likelihood_weighting = False
    
    # sampling configuration
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps = 100
    sampling.method = 'euler'
    sampling.denoise = True
    
    # device and seed
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    return config