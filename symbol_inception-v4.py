"""

Inception V3, suitable for images with around 299 x 299

Reference:

Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision." arXiv preprint arXiv:1512.00567 (2015).

"""

import find_mxnet
import mxnet as mx
def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s%s_conv2d' %(name, suffix))
    act = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

# Input Shape is 3*299*299 (th)
def inception_stem(data,
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


def InceptionA(data,
               num_1_1,
               num_2_1,
               num_3_1, num_3_2,
               num_4_1, num_4_2, num_4_3,
               name):
    a1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg', name=('%s_%s_pool1' %('avg', name)))
    a1 = Conv(data=a1, num_filter=num_1_1, name=('%s_a_1' % name), suffix='_conv')

    a2 = Conv(data=data, num_filter=num_2_1, name=('%s_a_2' % name), suffix='_conv')

    a3 = Conv(data=data, num_filter=num_3_1, name=('%s_a_3' % name), suffix='_conv_1')
    a3 = Conv(data=a3, num_filter=num_3_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2')

    a4 = Conv(data=data, num_filter=num_4_1, name=('%s_a_4' % name), suffix='_conv_1')
    a4 = Conv(data=a4, num_filter=num_4_2, kernel=(3, 3), pad=(1, 1), name=('%s_a_4' % name), suffix='_conv_2')
    a4 = Conv(data=a4, num_filter=num_4_3, kernel=(3, 3), pad=(1, 1), name=('%s_a_4' % name), suffix='_conv_3')

    m = mx.sym.Concat(*[a1, a2, a3, a4], name=('%s_a_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_a_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_a_relu1' % name))

    return m

def InceptionB(data,
               num_1_1,
               num_2_1,
               num_3_1, num_3_2, num_3_3,
               num_4_1, num_4_2, num_4_3, num_4_4, num_4_5,
               name):
    b1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg', name=('%s_%s_pool1' %('avg', name)))
    b1 = Conv(data=b1, num_filter=num_1_1, name=('%s_b_1' % name), suffix='_conv')

    b2 = Conv(data=data, num_filter=num_2_1, name=('%s_b_2' % name), suffix='_conv')

    b3 = Conv(data=data, num_filter=num_3_1, name=('%s_b_3' % name), suffix='_conv_1')
    b3 = Conv(data=b3, num_filter=num_3_2, kernel=(7, 1), pad=(3, 0), name=('%s_b_3' % name), suffix='_conv_2')
    b3 = Conv(data=b3, num_filter=num_3_3, kernel=(1, 7), pad=(0, 3), name=('%s_b_3' % name), suffix='_conv_3')

    b4 = Conv(data=data, num_filter=num_4_1, name=('%s_b_4' % name), suffix='_conv_1')
    b4 = Conv(data=b4, num_filter=num_4_2, kernel=(7, 1), pad=(3, 0), name=('%s_b_4' % name), suffix='_conv_2')
    b4 = Conv(data=b4, num_filter=num_4_3, kernel=(1, 7), pad=(0, 3), name=('%s_b_4' % name), suffix='_conv_3')
    b4 = Conv(data=b4, num_filter=num_4_4, kernel=(7, 1), pad=(3, 0), name=('%s_b_4' % name), suffix='_conv_4')
    b4 = Conv(data=b4, num_filter=num_4_5, kernel=(1, 7), pad=(0, 3), name=('%s_b_4' % name), suffix='_conv_5')

    m = mx.sym.Concat(*[b1, b2, b3, b4], name=('%s_b_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_b_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_b_relu1' % name))

    return m

def InceptionC(data,
               num_1_1,
               num_2_1,
               num_3_1, num_3_2, num_3_3,
               num_4_1, num_4_2, num_4_3, num_4_4, num_4_5,
               name):
    c1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg', name=('%s_%s_pool1' %('avg', name)))
    c1 = Conv(data=c1, num_filter=num_1_1, name=('%s_c_1' % name), suffix='_conv')

    c2 = Conv(data=data, num_filter=num_2_1, name=('%s_c_2' % name), suffix='_conv')

    c3 = Conv(data=data, num_filter=num_3_1, name=('%s_c_3' % name), suffix='_conv_1')
    c3_1 = Conv(data=c3, num_filter=num_3_2, kernel=(3, 1), pad=(1, 0), name=('%s_c_3' % name), suffix='_conv_2')
    c3_2 = Conv(data=c3, num_filter=num_3_3, kernel=(1, 3), pad=(0, 1), name=('%s_c_3' % name), suffix='_conv_3')

    c4 = Conv(data=data, num_filter=num_4_1, name=('%s_c_4' % name), suffix='_conv_1')
    c4 = Conv(data=c4, num_filter=num_4_2, kernel=(3, 1), pad=(1, 0), name=('%s_c_4' % name), suffix='_conv_2')
    c4 = Conv(data=c4, num_filter=num_4_3, kernel=(1, 3), pad=(0, 1), name=('%s_c_4' % name), suffix='_conv_3')
    c4_1 = Conv(data=c4, num_filter=num_4_4, kernel=(3, 1), pad=(1, 0), name=('%s_c_4' % name), suffix='_conv_4')
    c4_2 = Conv(data=c4, num_filter=num_4_5, kernel=(1, 3), pad=(0, 1), name=('%s_c_4' % name), suffix='_conv_5')

    m = mx.sym.Concat(*[c1, c2, c3_1, c3_2, c4_1, c4_2], name=('%s_c_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_c_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_c_relu1' % name))

    return m

def ReductionA(data,
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

def ReductionB(data,
               num_2_1, num_2_2,
               num_3_1, num_3_2, num_3_3, num_3_4,
               name):
    rb1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    rb2 = Conv(data=data, num_filter=num_2_1, name=('%s_rb_2' % name), suffix='_conv_1')
    rb2 = Conv(data=rb2, num_filter=num_2_2, kernel=(3, 3), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2')

    rb3 = Conv(data=data, num_filter=num_3_1, name=('%s_rb_3' % name), suffix='_conv_1')
    rb3 = Conv(data=rb3, num_filter=num_3_2, kernel=(7, 1), pad=(3, 0), name=('%s_rb_3' % name), suffix='_conv_2')
    rb3 = Conv(data=rb3, num_filter=num_3_3, kernel=(1, 7), pad=(0, 3), name=('%s_rb_3' % name), suffix='_conv_3')
    rb3 = Conv(data=rb3, num_filter=num_3_4, kernel=(3, 3), stride=(2, 2), name=('%s_rb_3' % name), suffix='_conv_4')

    m = mx.sym.Concat(*[rb1, rb2, rb3], name=('%s_rb_concat1' % name))
    m = mx.sym.BatchNorm(data=m, name=('%s_rb_bn1' % name))
    # m = mx.sym.Activation(data=m, act_type='relu', name=('%s_rb_relu1' % name))

    return m

# create inception_v4
def get_symbol(num_classes=1000):

    # input shape 3*229*229
    data = mx.symbol.Variable(name="data")

    # stage stem
    in_stem = inception_stem(data,
                             32, 32, 64,
                             96,
                             64, 96,
                             64, 64, 64, 96,
                             192,
                             'stem_stage')

    # stage 4 x InceptionA
    in4a = InceptionA(in_stem,
                      96,
                      96,
                      64, 96,
                      64, 96, 96,
                      'in4a_1')
    in4a = InceptionA(in4a,
                      96,
                      96,
                      64, 96,
                      64, 96, 96,
                      'in4a_2')
    in4a = InceptionA(in4a,
                      96,
                      96,
                      64, 96,
                      64, 96, 96,
                      'in4a_3')
    in4a = InceptionA(in4a,
                      96,
                      96,
                      64, 96,
                      64, 96, 96,
                      'in4a_4')

    # stage ReductionA
    re3a = ReductionA(in4a,
                      384,
                      192, 224, 256,
                      're3a')

    # stage 7 x InceptionB
    in4b = InceptionB(re3a,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_1')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_2')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_3')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_4')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_5')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_6')
    in4b = InceptionB(in4b,
                      128,
                      384,
                      192, 224, 256,
                      192, 192, 224, 224, 256,
                      'in4b_7')

    # stage ReductionB
    re3b = ReductionB(in4b,
                      192, 192,
                      256, 256, 320, 320,
                      're3b')

    # stage 3 x InceptionC
    in6c = InceptionC(re3b,
                      256,
                      256,
                      384, 256, 256,
                      384, 192, 224, 256, 256,
                      'in6c_1')
    in6c = InceptionC(in6c,
                      256,
                      256,
                      384, 256, 256,
                      384, 192, 224, 256, 256,
                      'in6c_2')
    in6c = InceptionC(in6c,
                      256,
                      256,
                      384, 256, 256,
                      384, 192, 224, 256, 256,
                      'in6c_3')


    # stage Average Pooling
    pool = mx.sym.Pooling(data=in6c, kernel=(8, 8), stride=(1, 1), pool_type="avg", name="global_pool")

    # stage Dropout
    dropout = mx.sym.Dropout(data=pool, p=0.2)
    # dropout =  mx.sym.Dropout(data=pool, p=0.8)
    flatten = mx.sym.Flatten(data=dropout, name="flatten")

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

if __name__ == '__main__':
    net = get_symbol(1000)
    mx.viz.plot_network(net).render('inceptionbn-v4')