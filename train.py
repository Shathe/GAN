import tensorflow as tf
import random
import os
import argparse
import time
from utils.utils import get_parameters
from Loader import Loader
import Network
import math
from lovasz_losses_tf import lovasz_softmax
random.seed(os.urandom(7))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train",
                    default='/media/msrobot/discoGordo/city2')  # '/media/msrobot/discoGordo/city'
# /media/msrobot/discoGordo/Download_april/machine_printed_legible
# '/media/msrobot/discoGordo/city2'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=4e-4)
parser.add_argument("--lr_decay", help="1 for lr decay, 0 for not", default=1)
parser.add_argument("--min_lr", help="Initial learning rate", default=7e-5)
parser.add_argument("--max_batch_size", help="batch_size", default=16)
parser.add_argument("--n_classes", help="number of classes to classify", default=19)
parser.add_argument("--ignore_label", help="class to ignore", default=255)
parser.add_argument("--epochs", help="Number of epochs to train", default=30)
parser.add_argument("--width", help="width", default=512)
parser.add_argument("--height", help="height", default=256)
parser.add_argument("--save_model", help="save_model", default=1)
parser.add_argument("--checkpoint_path", help="checkpoint path", default='./models/mininet/')
parser.add_argument("--train", help="if true, train, if not, test", default=1)
args = parser.parse_args()

'''
Para segmetnacion Coral: 
- cambiar size a (512, 512) las clases y bueno todos los parametros que veas 
-cambiar el datast
-cambiar las variables a optimizar y a cargar y restaurar
- cambiar el paramerto de checkpoint_path y los demas
'''

# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
lr_decay = bool(int(args.lr_decay))
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model))
train_or_test = bool(int(args.train))
init_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
channels = int(args.dimensions)
change_lr_epoch = math.pow(min_learning_rate / init_learning_rate, 1.0 / total_epochs)
checkpoint_path = args.checkpoint_path
augmenter = None
if augmentation:
    augmenter = 'segmentation'

loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType='segmentation', width=width,
                height=height, ignore_label=ignore_label, median_frequency=0.00)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)

# Placeholders
training_flag = tf.placeholder(tf.bool)
input_x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')
batch_images = tf.reverse(input_x, axis=[-1])  # opencv rgb -bgr
label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes + 1],
                       name='output')  # the n_classes + 1 is for the ignore classes
mask_label = tf.placeholder(tf.float32, shape=[None, height, width], name='mask')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Network
output = Network.MiniNet(input_x=input_x, n_classes=n_classes, training=training_flag)

# Get shapes
shape_output = tf.shape(output)
label_shape = tf.shape(label)
mask_label_shape = tf.shape(mask_label)

predictions = tf.reshape(output, [shape_output[1] * shape_output[2] * shape_output[0], shape_output[3]])
labels = tf.reshape(label, [label_shape[2] * label_shape[1] * label_shape[0], label_shape[3]])
mask_labels = tf.reshape(mask_label, [mask_label_shape[1] * mask_label_shape[0] * mask_label_shape[2]])

# Last class is the ignore class 
labels_ignore = labels[:, n_classes]
labels_real = labels[:, :n_classes]

cost = tf.losses.softmax_cross_entropy(labels_real, predictions, weights=mask_labels)
# cost = lovasz_softmax(probas=tf.nn.softmax(output), labels=tf.argmax(label, axis=3), classes='present', per_image=False, ignore=n_classes, order='BHWC')

# Metrics
labels = tf.argmax(labels, 1)
predictions = tf.argmax(predictions, 1)

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1)))  # ignore all labels >= num_classes
labels = tf.cast(tf.gather(labels, indices), tf.int64)
predictions = tf.gather(predictions, indices)

correct_prediction = tf.cast(tf.equal(labels, predictions), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

acc, acc_op = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
iou, conf_mat = tf.metrics.mean_iou(labels, predictions, n_classes)
conf_matrix_all = tf.confusion_matrix(labels, predictions, num_classes=n_classes)

# SUMMARY FOR TENSORBOARD
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.image('input', batch_images, max_outputs=4)
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
tf.summary.image('output', output_image, max_outputs=4)
label_image = tf.expand_dims(tf.cast(tf.argmax(label, 3), tf.float32), -1)
tf.summary.image('label', label_image, max_outputs=4)

# Different variables
# restore_variables = [var for var in tf.trainable_variables() if 'resnet_v2_101' in var.name]  # Change name
restore_variables = tf.global_variables()  # [var for var in tf.global_variables() if 'up3' not in var.name]  # Change name
train_variables = tf.trainable_variables() # [var for var in tf.trainable_variables() if 'up' in var.name or 'm8' in var.name]
stream_vars = [i for i in tf.local_variables() if
               'count' in i.name or 'confusion_matrix' in i.name or 'total' in i.name]

# Count parameters
get_parameters()

# For batch norm
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Uso el optimizador de Adam y se quiere minimizar la funcion de coste
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # adamOptimizer does not need lr decay
    train = optimizer.minimize(cost, var_list=train_variables)  # VARIABLES TO OPTIMIZE

# Times to show information of training

saver = tf.train.Saver(tf.global_variables())
restorer = tf.train.Saver(restore_variables)


if not os.path.exists(checkpoint_path+'iou'):
    os.makedirs(checkpoint_path+'iou')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # get checkpoint if there is one
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Loading model...')
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print('Model loaded')

    # For tensorboard
    merged = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writer_test = tf.summary.FileWriter('./logs/test', sess.graph)

    if train_or_test:

        # Start variables
        global_step = 0
        epoch_learning_rate = init_learning_rate
        batch_size_decimal = float(init_batch_size)
        best_val_loss = float('Inf')
        best_iou = float('-Inf')
        # EPOCH  loop
        for epoch in range(total_epochs):
            # Calculate tvariables for the batch and inizialize others
            time_first = time.time()
            batch_size = int(batch_size_decimal)
            print ("epoch " + str(epoch + 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size))

            total_steps = int(training_samples / batch_size) + 1
            show_each_steps = int(testing_samples / batch_size)
            if show_each_steps > total_steps:
                show_each_steps = total_steps
            loss_acum = 0.0

            # steps in every epoch
            for step in range(total_steps):
                # get training data
                batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True,
                                                                augmenter=augmenter)  # , augmenter='segmentation'

                train_feed_dict = {
                    input_x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    mask_label: batch_mask,
                    training_flag: True
                }
                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                # show info
                if step % show_each_steps == 0:
                    train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                    print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                    writer_train.add_summary(train_summary)

                    x_test, y_test, mask_test = loader.get_batch(size=batch_size, train=False)
                    test_feed_dict = {
                        input_x: x_test,
                        label: y_test,
                        mask_label: mask_test,
                        learning_rate: 0,
                        training_flag: False
                    }
                    acc_update, miou_update, mean_acc_update, test_summary, val_loss = sess.run(
                        [acc_op, conf_mat, mean_acc_op, merged, cost], feed_dict=test_feed_dict)
                    acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
                                                                                  feed_dict=test_feed_dict)
                    writer_test.add_summary(test_summary)
                    loss_acum = loss_acum + val_loss

            print("TEST")
            print("Accuracy: " + str(acc_total))
            print("miou: " + str(miou_total))
            print("mean accuracy: " + str(mean_acc_total))
            print("loss: " + str(loss_acum / show_each_steps))
            # print(matrix_conf)
            # initialize metric variables for next epoch
            sess.run(tf.variables_initializer(stream_vars))

            # save models
            if save_model and best_iou < miou_total:
                best_iou = miou_total
                saver.save(sess=sess, save_path=checkpoint_path + 'iou/model.ckpt')
            if save_model and best_val_loss > loss_acum / testing_samples:
                best_val_loss = loss_acum / testing_samples
                saver.save(sess=sess, save_path=checkpoint_path + 'model.ckpt')

            # show tiem to finish training
            time_second = time.time()
            epochs_left = total_epochs - epoch - 1
            segundos_per_epoch = time_second - time_first
            print(str(segundos_per_epoch * epochs_left) + ' seconds to end the training. Hours: ' + str(
                segundos_per_epoch * epochs_left / 3600.0))

            # agument batch_size per epoch and decrease the learning rate
            if lr_decay:
                epoch_learning_rate = init_learning_rate * math.pow(change_lr_epoch,
                                                                    epoch)  # adamOptimizer does not need lr decay


    else:

        # TEST
        loss_acum = 0.0
        for i in xrange(0, testing_samples):
            x_test, y_test, mask_test = loader.get_batch(size=1, train=False)
            test_feed_dict = {
                input_x: x_test,
                label: y_test,
                mask_label: mask_test,
                learning_rate: 0,
                training_flag: 0
            }
            acc_update, miou_update, mean_acc_update, val_loss = sess.run([acc_op, conf_mat, mean_acc_op, cost],
                                                                          feed_dict=test_feed_dict)
            acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
                                                                          feed_dict=test_feed_dict)
            '''
            if math.isnan(val_loss):
                val_loss = np.inf
            '''
            loss_acum = loss_acum + val_loss

        print("TEST")
        print("Accuracy: " + str(acc_total))
        print("miou: " + str(miou_total))
        print("mean accuracy: " + str(mean_acc_total))
        print("loss: " + str(loss_acum / testing_samples))
        print('matrix_conf')
        print(matrix_conf)
