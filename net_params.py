from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]

convgru_sposs_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [5, 16, 3, 1, 1],
                     'conv2_leaky_1': [16, 32, 3, 2, 1],
                     'conv3_leaky_1': [32, 64, 3, 2, 1],
                     'conv4_leaky_1': [64, 128, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(5,32), input_channels=128, filter_size=5, num_features=128)
    ]
]

convgru_sposs_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 64, 4, 2, 1],
                     'deconv2_leaky_1': [64, 32, 4, 2, 1],
                     'deconv3_leaky_1': [32, 16, 4, 2, 1],
                     'conv4_leaky_1': [16, 2, 3, 1, 1],
                     'conv5_1': [2, 2, 1, 1, 0]})
    ],

    [
        CGRU_cell(shape=(5,32), input_channels=128, filter_size=5, num_features=128)
    ]
]

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),   # --> B*Seq*16*64*64
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),  # --> B*Seq*64*32*32
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),  # --> B*Seq*96*16*16
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),   # --> B*Seq*64*64*64
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),   # --> B*Seq*96*32*32
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)    # --> B*Seq*96*16*16
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CGRU_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]