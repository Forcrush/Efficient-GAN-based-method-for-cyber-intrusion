import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
from sklearn.metrics import precision_recall_fscore_support

# network file
import bigan.backblaze_utilities as GAN
# data preprocess file
import data.backblaze as data


RANDOM_SEED = 13
# print frequency image tensorboard [20]
FREQ_PRINT = 20
# parameter of WGAN-GP
LAMBDA = 10


# to update neural net with moving avg variables, suitable for ss learning cf Saliman
def get_getter(ema):  
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


# display some parameters
def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)


# visualize the train progress
def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


# save training logs, weights, biases, etc.
def create_logdir(method, weight, rd):
    return "bigan/train_logs/backblaze/{}/{}/{}".format(weight, method, rd)


def train_and_test(nb_epochs, weight, method, degree, dataset, random_seed):
    """ 
    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
    """
    logger = logging.getLogger("BiGAN.train.{}.{}".format(dataset, method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=(None, 256), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train()
    trainx_copy = trainx.copy()
    testx, testy, contaminate_rate = data.get_test()
    # can be delete----
    lx = len(testx)
    ly = len(testy)

    # Parameters
    starting_lr = GAN.learning_rate
    batch_size = GAN.batch_size
    latent_dim = GAN.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    # ==================================================================================
    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = GAN.generator
    enc = GAN.encoder
    dis = GAN.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, latent_dim])
        x_gen = gen(z, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        l_generator, inter_layer_rct = dis(z, x_gen, is_training=is_training_pl, reuse=True)


    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = -tf.reduce_mean(l_encoder)
        loss_dis_gen = tf.reduce_mean(l_generator)
        loss_discriminator = loss_dis_enc + loss_dis_enc
        # generator
        loss_generator = -tf.reduce_mean(l_generator)
        # encoder
        loss_encoder = tf.reduce_mean(l_encoder)

        # GP
        '''
        alpha = tf.random_uniform(
            shape=[batch_size, 1],minval=0.,maxval=1.)
        differences = x_gen - input_pl
        interpolates = input_pl + (alpha * differences)
        gradients = (tf.gradients(dis(z,interpolates,reuse=True), [interpolates])[0] +
                     tf.gradients(dis(z_gen,interpolates,reuse=True), [interpolates])[0] ) / 2
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_discriminator += LAMBDA * gradient_penalty
        '''

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]
        dvars = [var for var in tvars if 'discriminator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)


    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    # ==================================================================================
    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator_ema),
                                                                    logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                    keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score

    logdir = create_logdir(weight, method, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

            # training
            for t in range(nr_batches_train):
                
                display_progression_epoch(t, nr_batches_train)             
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl:trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}

                _, ld, sm = sess.run([train_dis_op,
                                      loss_discriminator,
                                      sum_op_dis],
                                     feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator and encoder
                feed_dict = {input_pl:trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, _, le, lg, sm = sess.run([train_gen_op,
                                            train_enc_op,
                                            loss_encoder,
                                            loss_generator,
                                            sum_op_gen],
                                            feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))

            epoch += 1
            
        logger.warn('Testing evaluation...')

        # shuffling  dataset
        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  
        testy = testy[inds]
        scores = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (np.mean(inference_time)))

        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        # ============================================================================!!!!!!!!!!!!!!!!!!!! 16 need to be modified
        fill = np.ones([batch_size - size, 256])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl:batch, is_training_pl:False}

        batch_score = sess.run(list_scores, feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        # Contaminate rate = x % ===> highest x % are anomalous
        per = np.percentile(scores, 100 - contaminate_rate * 100)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)
        # can be delete---------
        cc = 0
        ccc = 0
        for i in inds:
            if i==True:
                cc += 1
        for i in inds_comp:
            if i==True:
                ccc += 1
        # ---------can be delete
        
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(testy,
                                                                   y_pred,
                                                                   average='binary')

        print("Initialization: len(testx):{} len(testy):{} contaminate_rate:{}".format(lx, ly, contaminate_rate))
        print("After discriminate: 0-number:{} 1-number:{}".format(cc, ccc))
        print("Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f " % (precision, recall, f1))


# training process
def train(nb_epochs, weight, method, degree, dataset, random_seed=24):
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, dataset, random_seed)
