import torch.nn as nn
from models.building_bloacks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
import torch
from models.utils import get_class, number_of_features_per_level
from einops import rearrange
use_prompt = True


class ECA_3D(nn.Module):
    def __init__(self, k_size=3):
        super(ECA_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, 
                              kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      
        b, c, d, h, w = x.size()
        

        y = self.avg_pool(x)
        
 
        y = y.view(b, c)
        
       
        y = y.unsqueeze(1)
        
        y = self.conv(y)  
        
        
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        
        return x * y.expand_as(x)



    
class Mask_Conv(nn.Module):
    def __init__(self, in_channel ,out_channel):
        super().__init__()
        self.eca = ECA_3D(k_size=3)
        self.conv = nn.Conv3d(in_channels=in_channel,out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
    def forward(self,x_with_prompt):
        x = self.eca(x_with_prompt)
        x = self.conv(x)
        x = self.act(x)
        return x

class PromptEncoder(nn.Module):
    def __init__(self,embed_dim=64):
        super(PromptEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=14, embedding_dim=embed_dim) 
        self.conv1 = nn.Conv3d(in_channels=embed_dim,out_channels=128,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
    def forward(self,tooth_number,x):
        b, _,d, h, w = x.size()
        tooth_embedding = self.embedding(tooth_number)
        tooth_embedding = rearrange(tooth_embedding,'b c -> b c 1 1 1').expand(b,64,d,h,w)
        x = self.conv1(tooth_embedding)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    
class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d)
        # use prompt encoder
        if use_prompt:
            self.mask_conv = Mask_Conv(in_channel=1024+128,out_channel=1024)
            self.prompt_encoder = PromptEncoder(embed_dim=64)
        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x,prompt =None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        ## Using Prompt
        if use_prompt:
            prompt = self.prompt_encoder(prompt,x)
           # prompt = self.prompt_encoder(prompt)
            x = torch.cat([x,prompt],dim=1)
            x = self.mask_conv(x)
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        final_feature = x
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x,final_feature




class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=True)




def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)
