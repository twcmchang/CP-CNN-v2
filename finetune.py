import os
import time
import argparse
import numpy as np
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display

from model import VGG16
from utils import CIFAR10, CIFAR100, gammaSparsifyVGG16

import imgaug as ia
from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
transform = iaa.Sequential([
    sometimes(iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})),
    sometimes(iaa.Affine(scale={"x": (0.85, 1.15), "y":(0.85, 1.15)})),
    sometimes(iaa.Affine(rotate=(-45, 45))),
    sometimes(iaa.Add((-10,10), per_channel=0.5)),
    sometimes(iaa.Fliplr(0.5))
])

arr_spareness = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='keras-vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('--tesla', type=int, default=1, help='task-wise early stopping and loss aggregation')
    parser.add_argument('--lambda_s', type=float, default=0.0, help='multiplier for sparsity regularization')
    parser.add_argument('--lambda_m', type=float, default=0.0, help='multiplier for monotonicity-induced penalty')
    parser.add_argument('--log_dir', type=str, default='log', help='directory containing log text')
    parser.add_argument('--decay', type=float, default=0.0, help='multiplier for weight decay')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')    
    parser.add_argument('--note', type=str, default='', help='argument for taking notes')

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)

    finetune(FLAG)

def finetune(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        train_data = CIFAR10(train=True)
        test_data  = CIFAR10(train=False)
        tasks = ['100', '75', '50', '25']
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        train_data = CIFAR100(train=True)
        test_data  = CIFAR100(train=False)
        tasks = ['100', '50']
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")
    print("Build VGG16 models for %s..."% FLAG.dataset)

    Xtrain, Ytrain = train_data.train_data, train_data.train_labels
    Xtest, Ytest = test_data.test_data, test_data.test_labels
  
    vgg16.build(vgg16_npy_path=FLAG.init_from, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=True)

    # build model using  dp
    dp = [(i+1)*0.05 for i in range(1,20)]
    vgg16.set_idp_operation(dp=dp, decay=FLAG.decay ,keep_prob=FLAG.keep_prob)

    # define tasks
    print(tasks)

    # loss aggregation
    obj = 0.0
    for cur_task in tasks:
        obj += vgg16.loss_dict[cur_task]
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=len(tasks))
    checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')

    tvars_trainable = tf.trainable_variables()
    for rm in vgg16.gamma_var:
       tvars_trainable.remove(rm)
       print('%s is not trainable.'% rm)

    for var in tvars_trainable:
        if '_bn_' in var.name:
            tvars_trainable.remove(var)
            print('%s is not trainable.'% var)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # hyper parameters
        batch_size = 64
        epoch = 500
        early_stop_patience = 50
        min_delta = 0.0001
        opt_type = 'adam'

        # recorder
        epoch_counter = 0

        # optimizer
        global_step = tf.Variable(0, trainable=False)
        
        # Passing global_step to minimize() will increment it at each step.
        if opt_type is 'sgd':
            start_learning_rate = 4e-5 # adam # 4e-3 #sgd
            half_cycle = 20000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        else:
            start_learning_rate = 4e-5 # adam # 4e-3 #sgd
            half_cycle = 10000
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, half_cycle, 0.5, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = opt.minimize(obj, global_step=global_step, var_list=tvars_trainable)

        # progress bar
        ptrain = IntProgress()
        pval = IntProgress()
        display(ptrain)
        display(pval)
        ptrain.max = int(Xtrain.shape[0]/batch_size)
        pval.max = int(Xtest.shape[0]/batch_size)
        
        spareness = vgg16.spareness(thresh=0.02)
        print("initial spareness: %s" % sess.run(spareness))

        # re-initialize
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_accu = 0 

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience and epoch_counter < epoch):
    
            def load_batches():
                for i in range(int(Xtrain.shape[0]/batch_size)):
                    st = i*batch_size
                    ed = (i+1)*batch_size
                    batch = ia.Batch(images=Xtrain[st:ed,:,:,:], data=Ytrain[st:ed,:])
                    yield batch

            batch_loader = ia.BatchLoader(load_batches)
            bg_augmenter = ia.BackgroundAugmenter(batch_loader=batch_loader, augseq=transform, nb_workers=4)


            # start training
            stime = time.time()
            bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    print("Finished epoch.")
                    break
                x_images_aug = batch.images_aug
                y_images = batch.data
                sess.run([train_op], feed_dict={vgg16.x: x_images_aug,
                                vgg16.y: y_images,
                                vgg16.is_train: False})
                bar_train.next()
                ptrain.value +=1
                ptrain.description = "Training %s/%s" % (ptrain.value, ptrain.max)
            batch_loader.terminate()
            bg_augmenter.terminate()
            # # training an epoch
            # for i in range(int(Xtrain.shape[0]/batch_size)):
            #     st = i*batch_size
            #     ed = (i+1)*batch_size

            #     augX = transform.augment_images(Xtrain[st:ed,:,:,:])

            #     sess.run([train_op], feed_dict={vgg16.x: augX,
            #                                     vgg16.y: Ytrain[st:ed,:],
            #                                     vgg16.is_train: False})
            #     ptrain.value +=1
            #     ptrain.description = "Training %s/%s" % (i, ptrain.max)
            #     bar_train.next()

            # validation
            val_loss = 0
            val_accu = 0
            for i in range(int(Xtest.shape[0]/200)):
                st = i*200
                ed = (i+1)*200
                loss, accu = sess.run([obj, vgg16.accu_dict[cur_task]],
                                    feed_dict={vgg16.x: Xtest[st:ed,:],
                                                vgg16.y: Ytest[st:ed,:],
                                                vgg16.is_train: False})
                val_loss += loss
                val_accu += accu
                pval.value += 1
                pval.description = "Testing %s/%s" % (pval.value, pval.max)
            val_loss = val_loss/pval.value
            val_accu = val_accu/pval.value

            print("\nspareness: %s" % sess.run(spareness))
            # early stopping check
            if (val_accu - current_best_val_accu) > min_delta:
                current_best_val_accu = val_accu
                patience_counter = 0

                para_dict = sess.run(vgg16.para_dict)
                np.save(os.path.join(FLAG.save_dir, "finetune_dict.npy"), para_dict)
                print("save in %s" % os.path.join(FLAG.save_dir, "finetune_dict.npy"))
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]

            # epoch end
            # writer.add_summary(epoch_summary, epoch_counter)
            epoch_counter += 1

            ptrain.value = 0
            pval.value = 0
            bar_train.finish()
            bar_val.finish()

            print("Epoch %s (%s), %s sec >> obj loss: %.4f, task at %s: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), val_loss, cur_task, val_accu))
        saver.save(sess, checkpoint_path, global_step=epoch_counter)
    FLAG.optimizer = opt_type
    FLAG.lr = start_learning_rate
    FLAG.batch_size = batch_size
    FLAG.epoch_end = epoch_counter
    FLAG.val_accu = current_best_val_accu

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
    if os.path.exists("/home/cmchang/new_CP_CNN/model.csv"):
        with open("/home/cmchang/new_CP_CNN/model.csv", "a") as myfile:
            myfile.write(row)
    else:
        with open("/home/cmchang/new_CP_CNN/model.csv", "w") as myfile:
            myfile.write(header)
            myfile.write(row)

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): 
        sess.run(tf.variables_initializer(not_initialized_vars))

if __name__ == '__main__':
    main()