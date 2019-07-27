import sys
import os
import math
import mxnet as mx
import memonger
import time
import numpy as np

def ConvModule(sym, num_filter, kernel, pad=(0, 0), stride=(1, 1), fix_gamma=True):
    conv = mx.sym.Convolution(data=sym, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    # bn = mx.sym.BatchNorm(data=conv, fix_gamma=fix_gamma)
    act = mx.sym.LeakyReLU(data=conv, act_type="leaky") # same memory to our act, less than CuDNN one
    return act

def ResModule(sym, base_filter, stage, layer, fix_gamma=True, mirror_memo=True):
    num_f = base_filter * int(math.pow(2, stage))
    s = 1
    if stage != 0 and layer == 0:
        s = 2
    conv1 = ConvModule(sym, num_f, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv2 = ConvModule(conv1, num_f, kernel=(3, 3), pad=(1, 1), stride=(s, s))
    conv3 = ConvModule(conv2, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(1, 1))

    if layer == 0:
        sym = ConvModule(sym, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(s, s))

    sum_sym = sym + conv3
    # Annotate the critical points that can be saved as inter-stage parameter
    if mirror_memo:
        sym._set_attr(mirror_stage='True')
    return sum_sym

def get_symbol(layers, mirror_memo=True):
    """Get a 4-stage residual net, with configurations specified as layers.

    Parameters
    ----------
    layers : list of stage configuratrion
    """
    assert(len(layers) == 4)
    base_filter = 64
    data = mx.sym.Variable(name='data')
    conv1 = ConvModule(data, base_filter, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    mp1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2))
    sym = mp1
    for j in range(len(layers)):
        for i in range(layers[j]):
            sym = ResModule(sym, base_filter, j, i, mirror_memo=mirror_memo)

    avg = mx.symbol.Pooling(data=sym, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1000, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return net

def run_model(net, dshape, dev):
    sym, data_shape = net, [('data', dshape)]
    
    data = np.arange(3*244*244*32).reshape((32,3,244,244))
    label = np.ones([32])
    # batch = mx.io.DataBatch(data, label) # empty label
    batch_iter = mx.io.NDArrayIter(data, label, batch_size=8)
    
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(data_shapes = batch_iter.provide_data, label_shapes=batch_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
    metric = mx.metric.create('acc')

    
    batch_nums = 10
    batch_size = dshape[0]
    
    tic = time.time()
    for i in range(batch_nums):
        batch_iter.reset()
        metric.reset()
        for batch in batch_iter:
            mod.forward(batch, is_train=True)
            mod.update_metric(metric, batch.label)
            mod.backward()
            mod.update()
        # for output in mod.get_outputs():
        #     output.wait_to_read()
        print('epoch %d, training %s' %(i, metric.get()))

    return batch_nums * batch_size / (time.time() - tic)

if __name__ == "__main__":
    layers = [3, 24, 36, 3]
    batch_size = 32
    net_with_all_memo = get_symbol(layers, mirror_memo=False)
    net = get_symbol(layers)
    dshape = (batch_size, 3, 224, 224)
    net_mem_planned = memonger.search_plan(net, data=dshape)
    
    init_cost = memonger.get_cost(net_with_all_memo, data=dshape)
    init_speed = run_model(net_with_all_memo, dshape, mx.gpu(0))
    print('init feature map cost=%d MB' % init_cost)
    print("init speed is : %f img/s" % init_speed)

    old_cost = memonger.get_cost(net, data=dshape)
    old_speed = run_model(net, dshape, mx.gpu(0))
    print('Old feature map cost=%d MB' % old_cost)
    print("old speed is : %f img/s" % old_speed)
    
    new_cost = memonger.get_cost(net_mem_planned, data=dshape)
    new_speed = run_model(net_mem_planned, dshape, mx.gpu(0))
    print('New feature map cost=%d MB' % new_cost)
    print("new speed is : %f img/s" % new_speed)

# You can savely feed the net to the subsequent mxnet training script.