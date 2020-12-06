import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear, InstanceNorm, SpectralNorm
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn

#################################################################################
#这一部分十分感谢百度飞桨团队对github中issue问题的解答，对各种函数定义给出的建议与方案##
#################################################################################

'''
重新定义的类（或称论文中的函数）有：BCEWithLogitsLoss、Spectralnorm、ReflectionPad2d、
Upsample、ReLU、LeakyReLU；
需要注意的是ReLU、LeakyReLU中的inplace函数需要设为false，这一点百度飞桨团队在issue中给出了解答
'''
class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'
        
    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out
    
class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")

    
# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        out = fluid.layers.resize_nearest(input=inputs, scale=self.scale)

        return out


# 定义ReLU函数
class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y=fluid.layers.relu(x)
            return y
             


# 定义Leaky_ReLU函数
class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha=0.02, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha=alpha
        self.inplace=inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.leaky_relu(x, alpha=self.alpha))
            return x
        else:
            y = fluid.layers.leaky_relu(x, alpha=self.alpha)
        return y


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU(False)]
        
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU(False)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU(False)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu')]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            # 设置相关属性的值
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False,act='relu'),
                         ILN(int(ngf * mult / 2))]

        UpBlock2 += [ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False,act='tanh')]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input) #[1, 256, 64, 64]
        gap = fluid.layers.adaptive_pool2d(x, 1, pool_type='avg', require_index=False, name=None) #[1, 256, 1, 1]
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) #torch.Size([1, 1])
        gap_logit = self.gap_fc(gap) #[1, 1]
        gap_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], [1, 0]) #[1, 256]这一步出现维度问题，paddle为[256, 1]，因此需要transpose
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3])
        gap = x * gap_weight #[1, 256, 64, 64]

        gmp = fluid.layers.adaptive_pool2d(x, 1, pool_type='max', require_index=False, name=None)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], [1, 0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3])  
        gmp = x * gmp_weight  #torch.Size([1, 256, 64, 64])
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)   #torch.Size([1, 512, 64, 64])
        x = self.relu(self.conv1x1(x))
        
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
            x_ = fluid.layers.reshape(x_, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        else:
            x_ = fluid.layers.reshape(x, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU(False)]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(False)
        
        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out= self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):

    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.9))

    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        rho_expand = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])
        out = rho_expand * out_in + (1 - rho_expand) * out_ln

        out=out * fluid.layers.unsqueeze(input=gamma, axes=[2, 3])+fluid.layers.unsqueeze(beta, [2, 3])

        return out


class ILN(fluid.dygraph.Layer):

    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self.gamma = self.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        
    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2, 3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1, 2, 3])
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        rho_expand = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])
        out = rho_expand * out_in + (1 - rho_expand)*out_ln
        gamma_expand = fluid.layers.expand(x=self.gamma, expand_times=[input.shape[0], 1, 1, 1])
        beta_expand = fluid.layers.expand(x=self.beta, expand_times=[input.shape[0], 1, 1, 1])
        out = out * gamma_expand + beta_expand
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 LeakyReLU(0.2)]        

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      LeakyReLU(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                  Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)),
                  LeakyReLU(0.2)]        

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 =Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True) 
        self.leaky_relu = LeakyReLU(0.2)

        self.pad=ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))   

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input) #[1, 2048, 2, 2]
        gap =fluid.layers.adaptive_pool2d(x, 1, pool_type='avg', require_index=False, name=None)
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1])
        gap_logit = self.gap_fc(gap)#torch.Size([1, 1])
        gap_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], [1, 0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3])
        gap = x * gap_weight #[1, 2048, 2, 2]

        gmp =fluid.layers.adaptive_pool2d(x, 1, pool_type='max', require_index=False, name=None)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = fluid.layers.transpose(list(self.gmp_fc.parameters())[0], [1, 0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3])
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
