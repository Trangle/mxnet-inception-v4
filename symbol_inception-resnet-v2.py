"""

Inception V4, suitable for images with around 299 x 299

Reference:

Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and the impact of residual connections on learning[J]. arXiv preprint arXiv:1602.07261, 2016.

"""

import find_mxnet
import mxnet as mx
def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s%s_conv2d' %(name, suffix))
    act = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def ConvLiner(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s%s_conv2d' %(name, suffix))
    return conv

# Input Shape is 3*299*299 (th)
def InceptionResnetStem(data,
                        num_1_1, num_1_2, num_1_3,
                        num_2_1,
                        num_3_1, num_3_2,
                        num_4_1, num_4_2, num_4_3, num_4_4,
                        num_5_1,
                        name):
    stem_3x3 = Conv(data=data, num_filter=num_1_1, kernel=(3, 3), stride=(2, 2), name=('%s_conv' % name))
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_2, kernel=(3, 3), name=('%s_stem' % name), suffix='_conv')
    stem_3x3 = Conv(data=stem_3x3, num_filter=num_1_3, kernel=(3, 3), pad=(1, 1), name=('%s_stem' % name), suffix='_conv_1')

    pool1 = mx.sym.Pooling(data=stem_3x3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))
    stem_1_3x3 = Conv(data=stem_3x3, num_filter=num_2_1, kernel=(3, 3), stride=(2, 2), name=('%s_stem_1' % name), suffix='_conv_1')

    concat1 = mx.sym.Concat(*[pool1, stem_1_3x3], name=('%s_concat_1' % name))

    stem_1_1x1 = Conv(data=concat1, num_filter=num_3_1, name=('%s_stem_1' % name), suffix='_conv_2')
    stem_1_3x3 = Conv(data=stem_1_1x1, num_filter=num_3_2, kernel=(3, 3), name=('%s_stem_1' % name), suffix='_conv_3')

    stem_2_1x1 = Conv(data=concat1, num_filter=num_4_1, name=('%s_stem_2' % name), suffix='_conv_1')
    stem_2_7x1 = Conv(data=stem_2_1x1, num_filter=num_4_2, kernel=(7, 1), pad=(3, 0), name=('%s_stem_2' % name), suffix='_conv_2')
    stem_2_1x7 = Conv(data=stem_2_7x1, num_filter=num_4_3, kernel=(1, 7), pad=(0, 3), name=('%s_stem_2' % name), suffix='_conv_3')
    stem_2_3x3 = Conv(data=stem_2_1x7, num_filter=num_4_4, kernel=(3, 3), name=('%s_stem_2' % name), suffix='_conv_4')

    concat2 = mx.sym.Concat(*[stem_1_3x3, stem_2_3x3], name=('%s_concat_2' % name))

    pool2 = mx.sym.Pooling(data=concat2, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool2' % ('max', name)))
    stem_3_3x3 = Conv(data=concat2, num_filter=num_5_1, kernel=(3, 3), stride=(2, 2), name=('%s_stem_3' % name), suffix='_conv_1')

    concat3 = mx.sym.Concat(*[pool2, stem_3_3x3], name=('%s_concat_3' % name))
    bn1 = mx.sym.BatchNorm(data=concat3, name=('%s_bn1' % name))
    # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_relu1' % name))

    return bn1


def InceptionResnetV2A(data,
                       num_1_1,
                       num_2_1, num_2_2,
                       num_3_1, num_3_2, num_3_3,
                       proj,
                       name,
                       scaleResidual=True):
    init = data

    a1 = Conv(data=data, num_filter=num_1_1, name=('%s_a_1' % name), suffix='_conv')

    a2 = Conv(data=data, num_filter=num_2_1, name=('%s_a_2' % name), suffix='_conv_1')
    a2 = Conv(data=a2, num_filter=num_2_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_2' % name), suffix='_conv_2')

    a3 = Conv(data=data, num_filter=num_3_1, name=('%s_a_3' % name), suffix='_conv_1')
    a3 = Conv(data=a3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2')
    a3 = Conv(data=a3, num_filter=num_3_3, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[a1, a2, a3], name=('%s_a_concat1' % name))

    conv = ConvLiner(data=merge, num_filter=proj, name=('%s_a_liner_conv' % name))
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_a_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_a_relu1' % name))

    return act

def InceptionResnetV2B(data,
                       num_1_1,
                       num_2_1, num_2_2, num_2_3,
                       proj,
                       name,
                       scaleResidual=True):

    init = data

    b1 = Conv(data=data, num_filter=num_1_1, name=('%s_b_1' % name), suffix='_conv')

    b2 = Conv(data=data, num_filter=num_2_1, name=('%s_b_2' % name), suffix='_conv_1')
    b2 = Conv(data=b2, num_filter=num_2_2, kernel=(7, 1), pad=(3, 0), name=('%s_b_2' % name), suffix='_conv_2')
    b2 = Conv(data=b2, num_filter=num_2_3, kernel=(1, 7), pad=(0, 3), name=('%s_b_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[b1, b2], name=('%s_b_concat1' % name))

    conv = ConvLiner(data=merge, num_filter=proj, name=('%s_b_liner_conv' % name))
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_b_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_b_relu1' % name))

    return act

def InceptionResnetV2C(data,
                       num_1_1,
                       num_2_1, num_2_2, num_2_3,
                       proj,
                       name,
                       scaleResidual=True):

    init = data

    c1 = Conv(data=data, num_filter=num_1_1, name=('%s_c_1' % name), suffix='_conv')

    c2 = Conv(data=data, num_filter=num_2_1, name=('%s_c_2' % name), suffix='_conv_1')
    c2 = Conv(data=c2, num_filter=num_2_2, kernel=(3, 1), pad=(1, 0), name=('%s_c_2' % name), suffix='_conv_2')
    c2 = Conv(data=c2, num_filter=num_2_3, kernel=(1, 3), pad=(0, 1), name=('%s_c_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[c1, c2], name=('%s_c_concat1' % name))

    conv = ConvLiner(data=merge, num_filter=proj, name=('%s_b_liner_conv' % name))
    if scaleResidual:
        conv *= 0.1

    out = init + conv
    bn = mx.sym.BatchNorm(data=out, name=('%s_c_bn1' % name))
    act = mx.sym.Activation(data=bn, act_type='relu', name=('%s_c_relu1' % name))

    return act

def ReductionResnetV2A(data,
                       num_2_1,
                       num_3_1, num_3_2, num_3_3,
                       name):
    ra1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    ra2 = Conv(data=data, num_filter=num_2_1, kernel=(3, 3), stride=(2, 2), name=('%s_ra_2' % name), suffix='_conv')

    ra3 = Conv(data=data, num_filter=num_3_1, name=('%s_ra_3' % name), suffix='_conv_1')
    ra3 = Conv(data=ra3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_ra_3' % name), suffix='_conv_2')
    ra3 = Conv(data=ra3, num_filter=num_3_3, kernel=(3, 3), stride=(2, 2), name=('%s_ra_3' % name), suffix='_conv_3')

    m = mx.sym.Concat(*[ra1, ra2, ra3], name=('%s_ra_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_ra_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_ra_relu1' % name))

    return m

def ReductionResnetV2B(data,
                     num_2_1, num_2_2,
                     num_3_1, num_3_2,
                     num_4_1, num_4_2, num_4_3,
                     name):
    rb1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    rb2 = Conv(data=data, num_filter=num_2_1, name=('%s_rb_2' % name), suffix='_conv_1')
    rb2 = Conv(data=rb2, num_filter=num_2_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2')

    rb3 = Conv(data=data, num_filter=num_3_1, name=('%s_rb_3' % name), suffix='_conv_1')
    rb3 = Conv(data=rb3, num_filter=num_3_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_3' % name), suffix='_conv_2')

    rb4 = Conv(data=data, num_filter=num_4_1, name=('%s_rb_4' % name), suffix='_conv_1')
    rb4 = Conv(data=rb4, num_filter=num_4_2, kernel=(3, 3), pad=(1, 1), name=('%s_rb_4' % name), suffix='_conv_2')
    rb4 = Conv(data=rb4, num_filter=num_4_3, kernel=(3, 3), stride=(2, 2), name=('%s_rb_4' % name), suffix='_conv_3')

    m = mx.sym.Concat(*[rb1, rb2, rb3, rb4], name=('%s_rb_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_rb_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_rb_relu1' % name))

    return m

# create inception_v4
def get_symbol(num_classes=1000, scale=True):

    # input shape 3*229*229
    data = mx.symbol.Variable(name="data")

    # stage stem
    in_stem = InceptionResnetStem(data,
                                  32, 32, 64,
                                  96,
                                  64, 96,
                                  64, 64, 64, 96,
                                  192,
                                  'stem_stage')

    # stage 5 x Inception Resnet A
    in3a = InceptionResnetV2A(in_stem,
                              32,
                              32, 32,
                              32, 48, 64,
                              384,
                              'in3a_1',
                              scaleResidual=scale)
    in3a = InceptionResnetV2A(in3a,
                              32,
                              32, 32,
                              32, 48, 64,
                              384,
                              'in3a_2',
                              scaleResidual=scale)
    in3a = InceptionResnetV2A(in3a,
                              32,
                              32, 32,
                              32, 48, 64,
                              384,
                              'in3a_3',
                              scaleResidual=scale)
    in3a = InceptionResnetV2A(in3a,
                              32,
                              32, 32,
                              32, 48, 64,
                              384,
                              'in3a_4',
                              scaleResidual=scale)
    in3a = InceptionResnetV2A(in3a,
                              32,
                              32, 32,
                              32, 48, 64,
                              384,
                              'in3a_5',
                              scaleResidual=scale)

    # stage Reduction Resnet A
    re3a = ReductionResnetV2A(in3a,
                              384,
                              256, 256, 384,
                              're3a')

    # stage 10 x Inception Resnet B
    in2b = InceptionResnetV2B(re3a,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_1',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_2',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_3',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_4',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_5',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_6',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_7',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_8',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_9',
                              scaleResidual=scale)
    in2b = InceptionResnetV2B(in2b,
                              192,
                              128, 160, 192,
                              1154,
                              'in2b_10',
                              scaleResidual=scale)
    # stage ReductionB
    re4b = ReductionResnetV2B(in2b,
                              256, 384,
                              256, 288,
                              256, 288, 320,
                              're4b')

    # stage 5 x Inception Resnet C
    in2c = InceptionResnetV2C(re4b,
                              192,
                              192, 224, 256,
                              2048,
                              'in2c_1',
                              scaleResidual=scale)
    in2c = InceptionResnetV2C(in2c,
                              192,
                              192, 224, 256,
                              2048,
                              'in2c_2',
                              scaleResidual=scale)
    in2c = InceptionResnetV2C(in2c,
                              192,
                              192, 224, 256,
                              2048,
                              'in2c_3',
                              scaleResidual=scale)
    in2c = InceptionResnetV2C(in2c,
                              192,
                              192, 224, 256,
                              2048,
                              'in2c_4',
                              scaleResidual=scale)

    in2c = InceptionResnetV2C(in2c,
                              192,
                              192, 224, 256,
                              2048,
                              'in2c_5',
                              scaleResidual=scale)
    # stage Average Pooling
    pool = mx.sym.Pooling(data=in2c, kernel=(8, 8), stride=(1, 1), pool_type="avg", name="global_pool")

    # stage Dropout
    dropout = mx.sym.Dropout(data=pool, p=0.2)
    # dropout =  mx.sym.Dropout(data=pool, p=0.8)
    flatten = mx.sym.Flatten(data=dropout, name="flatten")

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

if __name__ == '__main__':
    net = get_symbol(1000, scale=True)
    mx.viz.plot_network(net).render('inceptionbn-resnet-v2')
