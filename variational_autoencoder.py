from typing import Any, Callable, List, Union, Tuple

import torch
import torch.nn as nn

_size_2_t = Union[int, Tuple[int, int]]
_list_size_2_t = List[_size_2_t]
_tensor_size_3_t = Tuple[torch.TensorType, torch.TensorType, torch.TensorType]

class VariationalAutoEncoder(nn.Module):
    # def __init__(self,latent_size=32,num_classes=10):
    def __init__(self,
                 enc_in_channels: List[int],
                 enc_out_channels: List[int],
                 enc_kernel_sizes: _list_size_2_t,
                 enc_strides: _list_size_2_t,
                 enc_paddings: _list_size_2_t,
                 dec_in_channels: List[int],
                 dec_out_channels: List[int],
                 dec_kernel_sizes: _list_size_2_t,
                 dec_strides: _list_size_2_t,
                 dec_paddings: _list_size_2_t,
                 dec_output_paddings: _list_size_2_t,
                 latent_dim: int = 2,
                 data_dim: int = 512,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25,
                 latent_size=32,
                 num_classes=10) -> None:
        super(VariationalAutoEncoder,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        # Try Encoder Decoder class method
        self.encoder = Encoder(in_channels=enc_in_channels,
                               out_channels=enc_out_channels,
                               kernel_sizes=enc_kernel_sizes,
                               strides=enc_strides,
                               paddings=enc_paddings,
                               use_batchnorm=use_batchnorm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)
        # to calculate the hidden dim the size of the last dimension
        # of the encoder must be known, hence a sample forward pass is
        # done on the encoder to determine this.
        sample_input = torch.randn((1, enc_in_channels[0], data_dim, data_dim))
        hidden_dim = self.encoder(sample_input).size(-1)

        # initialize the bottleneck layer
        self.bottle_neck = BottleNeck(latent_dim, hidden_dim)
        
        self.decoder = Decoder(in_channels=dec_in_channels,
                               out_channels=dec_out_channels,
                               kernel_sizes=dec_kernel_sizes,
                               strides=dec_strides,
                               paddings=dec_paddings,
                               output_paddings=dec_output_paddings,
                               latent_dim=latent_dim,
                               hidden_dim=hidden_dim,
                               use_batchnorm=use_batchnorm,
                               use_dropout=use_dropout,
                               dropout_rate=dropout_rate)
        
        # For encode
        self.conv1 = nn.Conv2d(3+1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(5*5*32,300)
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        # For decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,4*4*32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        # self.conv5 = nn.ConvTranspose2d(1, 1, kernel_size=8)
        self.conv5 = nn.ConvTranspose2d(1, 3, kernel_size=8)

    # def encoder(self,x,y):
    #     y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
    #     y = y.expand(-1, -1, x.size(2), x.size(3))
    #     t = torch.cat((x,y),dim=1)
        
    #     t = F.relu(self.conv1(t))
    #     t = F.relu(self.conv2(t))
    #     t = t.reshape((x.shape[0], -1))
    #     print(t.shape)
        
    #     t = F.relu(self.linear1(t))
    #     mu = self.mu(t)
    #     logvar = self.logvar(t)
    #     return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 32, 4, 4))

    # def decoder(self, z):
    #     t = F.relu(self.linear2(z))
    #     t = F.relu(self.linear3(t))
    #     t = self.unFlatten(t)
    #     t = F.relu(self.conv3(t))
    #     t = F.relu(self.conv4(t))
    #     t = F.relu(self.conv5(t))
    #     return t


    # def forward(self, x, y):
    #     mu, logvar = self.encoder(x,y)
    #     z = self.reparameterize(mu,logvar)

    #     # Class conditioning
    #     z = torch.cat((z, y.float()), dim=1)
    #     pred = self.decoder(z)
    #     return pred, mu, logvar
    
    def forward(self, x: torch.TensorType) -> _tensor_size_3_t:
        # extract meaningful features from the data
        # by  performing a forward pass through the encoder
        encoder_output = self.encoder(x)

        # using features extracted from the data obtain mean and log_variance
        # that will be used to define the distribution of the latent space.
        # perform a forward pass through the bottle neck
        z, mu, log_variance = self.bottle_neck(encoder_output)

        # reconstruct the data points from by using the sampled points
        # by performing a forward pass through the decoder.
        x_hat = self.decoder(z)

        return z, mu, log_variance, x_hat
        
        
        
class Encoder(nn.Module):
    """
    The Encoder takes a data point and applies convolutional 
    operations to it.
    Implements a sequential stack of ConvBlock(s).

    Consists of:

    - `ConvBlock`(s):
        carries out the convolutional operations
    - `Flatten`:
        reshapes the 4d output tensor of the convolutional layers 
        into 2d tensor
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(Encoder, self).__init__()

        # initialize conv blocks
        conv_blocks = nn.ModuleList()
        # append ConvBlock(s) whith their configuration
        for i in range(len(kernel_sizes)):
            conv_blocks.append(
                ConvBlock(in_channels=in_channels[i],
                          out_channels=out_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=paddings[i],
                          use_batchnorm=use_batchnorm,
                          use_dropout=use_dropout,
                          dropout_rate=dropout_rate))

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.flatten = Flatten()

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # perform a forward pass through the convolutional blocks
        # (N, C, W, H) -> (N, C, W, H)
        conv_blocks_output = self.conv_blocks(x)

        # flatten the output from the conv_blocks
        # (N, C, W, H) -> (N, C * W * H)
        return self.flatten(conv_blocks_output)

class Flatten(nn.Module):
    """
    Implements a keras like Flatten layer,
    where only the batch dimension is maintained 
    and all the other dimensions are flattened. 
    i.e reshapes a tensor from (N, *,...) -> (N, product_of_other_dimensions)
    """
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # maintain the batch dimension
        # and concatenate all other dimensions
        return torch.flatten(x, start_dim=1)

class UnFlatten(nn.Module):
    """
    Implements a reshape operation that transforms a 2d
    tensor into a 4d tensor. 
    i.e reshapes a tensor from (N, channels * width * height) -> (N, channels, width, height)
    """
    def __init__(self, num_channels: int) -> None:
        super(UnFlatten, self).__init__()

        self.num_channels = num_channels

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        batch_size = x.size(0)
        width_height_dim = int((x.size(1) // self.num_channels)**0.5)

        return torch.reshape(x, (batch_size, self.num_channels,
                                 width_height_dim, width_height_dim))
                                 
class Lambda(nn.Module):
    """
    Implements a keras like Lambda layer 
    that wraps a function into a module making
    it composable.
    """
    def __init__(self, function: Callable = lambda x: x) -> None:
        super(Lambda, self).__init__()

        self.function = function

    def forward(self, x: torch.TensorType) -> Any:
        return self.function(x)
        
class BottleNeck(nn.Module):
    """
    Implements the layers that output parameters defining the 
    latent space as well as the sampling points from the latent space.

    The BottleNeck consists of:

    - `Linear`(Mean Layer):
        takes in features extracted by the encoder and outputs the mean of the
        latent space distribution.
    - `Linear`(Log Variance Layer):
        takes in features extracted by the encoder and outputs the logarithmic variance
        of the latent space distribution
    - `Lambda`(Reparametarization/Sampling Layer):
        takes the mean and logarithmic variance of the distribution and 
        samples points from this using the formula:
        `sample = mean + standard_deviation * epsilon`
        where epsilon is drawn from a standard normal distribution i.e `epsilon ~ N(0, I)`
    """
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super(BottleNeck, self).__init__()

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_variance = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Lambda(self.sample)

    def forward(self, x: torch.TensorType) -> _tensor_size_3_t:
        # outputs the mean of the distribution
        # mu: (N, latent_dim)
        mu = self.fc_mu(x)
        # outputs the log_variace of the distribution
        # log_variance: (N, latent_dim)
        log_variance = self.fc_log_variance(x)

        # sample z from a distribution
        z = self.sampling([mu, log_variance])

        return z, mu, log_variance

    def sample(self, args):
        mu, log_variance = args

        std = torch.exp(log_variance / 2)
        # define a distribution q with the parameters mu and std
        q = torch.distributions.Normal(mu, std)
        # sample z from q
        z = q.rsample()

        return z


class Decoder(nn.Module):
    """
    The Decoder takes a sampled point from the latent space and reconstructs 
    a data point from the sampled point. 

    The Decoder consists of: 

    - `Linear`(Decoder Input): 
        takes the sampled point and applies a linear transformation to it. 
        This is necessary for reshape operation that is carried out after this layer
        to make sure that the input shape to the ConvTranspose2d layer of the Decoder matches
        the shape of the last Conv2d layer in the Encoder.
    - `UnFlatten`:
        takes a 2d tensor and reshapes it into a 4d tensor. 
        In this case the sampled points are reshaped into a shape that
        can be passed into the ConvTransposeBlock(s).
    - `ConvTransposeBlock`(s):
        carries out the transposed convolutional operation as well as 
        regularization by using batch normalization and dropout layers.
    - OutputBlock: 

        - `ConvTranspose2d`:
            carries out the transposed convolutional operation.
        - `Sigmoid`:
            activation for the output layer. 
            Maps the tensors to the range [0, 1].
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: _list_size_2_t,
                 strides: _list_size_2_t,
                 paddings: _list_size_2_t,
                 output_paddings: _list_size_2_t,
                 latent_dim: int,
                 hidden_dim: int,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(Decoder, self).__init__()

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # number of channels should match the number of
        # channels of the first ConvTranspose2d layer
        self.unflatten = UnFlatten(in_channels[0])

        # initialize conv_transpose blocks
        conv_transpose_blocks = nn.ModuleList()
        # append ConvTransposeBlock(s) with their configuration
        # the last configuration is used for the output layer and thus
        # excluded from the list.
        for i in range(len(kernel_sizes) - 1):
            conv_transpose_blocks.append(
                ConvTransposeBlock(in_channels=in_channels[i],
                                   out_channels=out_channels[i],
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   use_batchnorm=use_batchnorm,
                                   use_dropout=use_dropout,
                                   dropout_rate=dropout_rate))
        self.conv_transpose_blocks = nn.Sequential(*conv_transpose_blocks)

        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels[-1],
                               out_channels=out_channels[-1],
                               kernel_size=kernel_sizes[-1],
                               stride=strides[-1],
                               padding=paddings[-1],
                               output_padding=output_paddings[-1]),
            nn.Sigmoid())

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # perform a linear transformation on the input
        # (N, latent_dim) -> (N, hidden_dim)
        x = self.decoder_input(x)

        # reshape the decoders input
        # (N, hidden_dim) -> (N, C, W, H)
        x = self.unflatten(x)

        # perform a forward pass through the conv_transpose_blocks
        # (N, C_in, W_in, H_in) -> (N, C_out, W_out, H_out)
        conv_transpose_output = self.conv_transpose_blocks(x)

        # perform a forward pass through the output block
        # (N, C_in, W_in, H_in) -> (N, C_out, W_out, H_out)
        return self.output_block(conv_transpose_output)

class ConvBlock(nn.Module):
    """
    Implements a Sequential stack of layers.

    Consists of:

    - `Conv2d`: 
        functions as a convolutional layer.
    - `LeakyReLU`: 
        functions as an activation layer.
    - `BatchNorm2d`(optional): 
        functions as a regularizing layer.
    - `Dropout`(optional): 
        functions as a regularizing layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t,
                 padding: _size_2_t = 1,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(ConvBlock, self).__init__()

        # the convolutional layer and activation layers are
        # always part of the module list.
        # Other layers; dropout and batch normalization
        # are added dynamically based on the arguments.
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.LeakyReLU()
        ])

        # add an optional batch normalization layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        # add an optional dropout layer
        if use_dropout:
            # Dropout2d randomly zeroes out entire channels as
            # opposed to regular Dropout which zeroes out neurons
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.conv_block(x)


class ConvTransposeBlock(nn.Module):
    """
    Implements a Sequential stack of layers.

    Consists of: 

    - `ConvTranspose2d`: 
        functions as a transposed convolutional layers. 
    - `LeakyReLU`:
        functions as an activation layer.
    - `BatchNorm2d`(optional):
        functions as a regularization layer.
    - `Dropout2d`(optional):
        functions as a regularization layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t,
                 padding: _size_2_t = 1,
                 output_padding: _size_2_t = 0,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.25) -> None:
        super(ConvTransposeBlock, self).__init__()

        # the transposed convolutional layer and activation layers are
        # always part of the module list.
        # Other layers; dropout and batch normalization
        # are added dynamically based on the arguments.
        layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding),
            nn.LeakyReLU()
        ])

        # optionally add a batch normalization layer
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        # optionally add a dropout layer
        if use_dropout:
            # Dropout2d randomly zeroes out entire channels
            # as opposed to regular Dropout that zeroes out neurons
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv_transpose_block = nn.Sequential(*layers)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.conv_transpose_block(x)
        