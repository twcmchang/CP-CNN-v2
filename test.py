import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from model import VGG16
from utils import CIFAR10, CIFAR100, dpSparsifyVGG16, countFlopsParas

FIDELITY = [0.25, 0.5, 0.75, 1.0]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--output', type=str, default='output.csv', help='output filename (csv)')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')

    FLAG = parser.parse_args()

    for fidelity in FIDELITY:
        with tf.variable_scope("VGG16"+str(int(fidelity*100))):
            FLAG.fidelity = fidelity
            test(FLAG)

def test(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        test_data  = CIFAR10(train=False)
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        test_data  = CIFAR100(train=False)
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")

    Xtest, Ytest = test_data.test_data, test_data.test_labels

    if FLAG.fidelity is not None:
        data_dict = np.load(FLAG.init_from, encoding='latin1').item()
        data_dict = dpSparsifyVGG16(data_dict,FLAG.fidelity)
        vgg16.build(vgg16_npy_path=data_dict, conv_pre_training=True, fc_pre_training=True)
        print("Build model from %s using dp=%s" % (FLAG.init_from, str(FLAG.fidelity*100)))
    else:
        vgg16.build(vgg16_npy_path=FLAG.init_from, conv_pre_training=True, fc_pre_training=True)
        print("Build full model from %s" % (FLAG.init_from))

    # build model using  dp
    # dp = [(i+1)*0.05 for i in range(1,20)]
    dp = [1.0]
    vgg16.set_idp_operation(dp=dp, keep_prob=FLAG.keep_prob)

    flops, params = countFlopsParas(vgg16)
    print("Flops: %3f M, Paras: %3f M" % (flops/1e6, params/1e6))
    FLAG.flops_M = flops/1e6
    FLAG.params_M = params/1e6

    with tf.Session() as sess:
        if FLAG.save_dir is not None:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG.save_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, checkpoint)
                print("Model restored %s" % checkpoint)
                sess.run(tf.global_variables())
            else:
                print("No model checkpoint in %s" % FLAG.save_dir)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables())
        print("Initialized")
        output = []
        for dp_i in dp:
            accu = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[:5000,:], vgg16.y: Ytest[:5000,:], vgg16.is_train: False})
            accu2 = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[5000:,:], vgg16.y: Ytest[5000:,:], vgg16.is_train: False})
            output.append((accu+accu2)/2)
            print("At DP={dp:.4f}, accu={perf:.4f}".format(dp=dp_i*FLAG.fidelity, perf=(accu+accu2)/2))
        res = pd.DataFrame.from_dict({'DP':[int(dp_i*100) for dp_i in dp],'accu':output})
        res.to_csv(FLAG.output, index=False)
        print("Write into %s" % FLAG.output)

    FLAG.accuracy = (accu+accu2)/2

    header = ''
    row = ''
    for key in sorted(vars(FLAG)):
        if header is '':
            header = key
            row = str(getattr(FLAG, key))
        else:
            header += ","+key
            row += ","+str(getattr(FLAG,key))
    row += "\n"
    header += "\n"
    if os.path.exists("/home/cmchang/new_CP_CNN/performance.csv"):
        with open("/home/cmchang/new_CP_CNN/performance.csv", "a") as myfile:
            myfile.write(row)
    else:
        with open("/home/cmchang/new_CP_CNN/performance.csv", "w") as myfile:
            myfile.write(header)
            myfile.write(row)

if __name__ == '__main__':
	main()
