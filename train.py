import tensorflow as tf
import os
import sys
import time
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models import network
from utils import anchor, losses, session_config
from configs import train_config
config = train_config.Config()
if config.dataset_name == 'SED-dataset':
    from data_loader import sed_dataset as dataset
else:
    raise ValueError("Wrong dataset name in your config/train_config.py")

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
NUM_CLASSES = len(config.label_set)+1

@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = config.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == '__main__':
    session_config.setup_gpus(True, 0.9)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ds_obj = dataset.Dataset()
    batch_generator, train_length = ds_obj.load_data_generator('train', config.batch_size, num_examples = config.num_examples)
    val_generator, val_length = ds_obj.load_data_generator('val', config.batch_size, num_examples = config.num_examples)
    # batch_generator, val_generator, info = create_batch_generator(
    #     args.data_dir, args.data_year, default_boxes,
    #     config['image_size'],
    #     args.batch_size, args.num_batches,
    #     mode='train', augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes
    
    try:
        ssd = network.create_ssd(NUM_CLASSES, config.arch,
                        config.pretrained_type,
                        checkpoint_dir=config.checkpoint_dir)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion = losses.create_losses(config.neg_ratio, NUM_CLASSES)

    steps_per_epoch = train_length // config.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * config.num_epochs * 2 / 3),
                    int(steps_per_epoch * config.num_epochs * 5 / 6)],
        values=[config.initial_lr, config.initial_lr * 0.1, config.initial_lr * 0.01])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    '''
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_fn,
        momentum=config.momentum)
    '''
    train_log_dir = 'logs/train'
    #val_log_dir = 'logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    #val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    for epoch in range(config.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if (i + 1) % 5 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)
            tf.summary.scalar('val_loss', avg_val_loss, step=epoch)
            tf.summary.scalar('val_conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('val_loc_loss', avg_val_loc_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            ssd.save_weights(
                os.path.join(config.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
