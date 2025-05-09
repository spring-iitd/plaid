import pandas as pd
import numpy as np
import glob
import json
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



# def serialize_example(x, y):
##ORIGINAL
#     """converts x, y to tf.train.Example and serialize"""
#     #Need to pay attention to whether it needs to be converted to numpy() form
#     input_features = tf.train.Int64List(value = np.array(x).flatten())
#     label = tf.train.Int64List(value = np.array([y]))
#     features = tf.train.Features(
#         feature = {
#             "input_features": tf.train.Feature(int64_list = input_features),
#             "label" : tf.train.Feature(int64_list = label)
#         }
#     )
#     example = tf.train.Example(features = features)
#     return example.SerializeToString()

def serialize_example(x, y):
    """Converts x, y to tf.train.Example and serializes."""
    input_features = tf.train.Int64List(value=np.array(x).flatten())
    
    # Ensure y is an integer scalar
    if isinstance(y, (np.ndarray, list)):
        y = y[0]  # Get the scalar value from array/list
    
    label = tf.train.Int64List(value=[int(y)])  # Convert to int if necessary
    # print("------------xxxxxxxxx--------------PRINTING LABEL------------xxxxxxxxx--------------")
    # print(label)

    features = tf.train.Features(
        feature={
            "input_features": tf.train.Feature(int64_list=input_features),
            "label": tf.train.Feature(int64_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def read_tfrecord(example):
    input_dim = 29 * 29
    feature_description = {
    'input_features': tf.io.FixedLenFeature([input_dim], tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)

def data_from_tfrecord(tf_filepath, batch_size, repeat_time):
    data = tf.data.TFRecordDataset(tf_filepath)
    data = data.map(read_tfrecord)
    data = data.shuffle(2)
    data = data.repeat(repeat_time)
    data = data.batch(batch_size)
    # print(tf.data.experimental.cardinality(data))
    iterator = data.make_one_shot_iterator()
    return iterator.get_next()

def data_helper(data_tf, sess):
    n_labels = 2
    data = sess.run(data_tf)
    x, y = data['input_features'], data['label']
    size = x.shape[0]
    y_one_hot = np.eye(n_labels)[y].reshape([size, n_labels])
    return x, y_one_hot



def write_tfrecord(data, filename):
    print('Writing {}================= '.format(filename))
    
    iterator = data.make_one_shot_iterator()  # Create an iterator for the dataset
    next_element = iterator.get_next()  # Get the next element operation
    init = tf.global_variables_initializer()
    
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    
    with tf.Session() as sess:
        sess.run(init)
        
        while True:
            try:
                # Fetch the next batch of data
                batch_data = sess.run(next_element)
                
                # Loop through each example in the batch
                for x, y in zip(batch_data['input_features'], batch_data['label']):
                    tfrecord_writer.write(serialize_example(x, y))
                    
            except tf.errors.OutOfRangeError:
                # End of dataset
                print("End of dataset")
                break
            
    tfrecord_writer.close()


def train_test_split(source_path, dest_path, DATASET_SIZE,\
                     train_label_ratio=0.1, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):

    train_size = int(DATASET_SIZE * train_ratio)
    train_label_size = int(train_size * train_label_ratio)
    val_size = int(DATASET_SIZE * val_ratio)
    test_size = int(DATASET_SIZE * test_ratio)

    print(train_label_size, train_size, val_size, test_size)
    
    dataset = tf.data.TFRecordDataset(source_path)
    dataset = dataset.map(read_tfrecord)
    dataset = dataset.shuffle(50000)
    
    train = dataset.take(train_size)
    train_label = train.take(train_label_size)
    train_unlabel = train.skip(train_label_size)
    
    val = dataset.skip(train_size)
    test = val.skip(val_size)
    val = val.take(val_size)
    
    batch_size = 10000
    train_label = train_label.batch(batch_size)
    train_unlabel = train_unlabel.batch(batch_size)
    test = test.batch(batch_size)
    val = val.batch(batch_size)

    train_test_info = {
        "train_unlabel": train_size - train_label_size,
        "train_label": train_label_size,
        "validation": val_size,
        "test": test_size
    }
    json.dump(train_test_info, open(dest_path + 'datainfo.txt', 'w'))

    print("TRAIN LABEL TYPE",type(train_label))

    write_tfrecord(train_label, dest_path + 'train_label')
    print("Train_label",train_label)

    write_tfrecord(train_unlabel, dest_path + 'train_unlabel')
    print("Train_unlabel", train_unlabel)

    write_tfrecord(test, dest_path + 'test')
    print("Test", test)

    write_tfrecord(val, dest_path + 'val')
    print("Val", val)


def main_attack(attack_types, args):
    indir = args.indir
    outdir = args.outdir + '/Train_{}_Labeled_{}'.format(args.train_ratio, args.train_label_ratio)
    data_info = json.load(open('{}/datainfo.txt'.format(indir)))
    for attack in attack_types:
        print("Attack: {} ==============".format(attack))
        source = '{}/{}'.format(indir, attack)
        dest = '{}/{}/'.format(outdir, attack)
        print("==============")
        if not os.path.exists(dest):
            os.makedirs(dest)
        source = source.replace('/TFRecord/', '/TFRecord//')
        train_test_split(source, dest, data_info[source], 
                        train_label_ratio=args.train_label_ratio, train_ratio=args.train_ratio, 
                        val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        
def main_normal(attack_types, args):
    indir = args.indir
    outdir = args.outdir + '/Train_{}_Labeled_{}'.format(args.train_ratio, args.train_label_ratio)
    normal_size = 0
    data_info = json.load(open('{}/datainfo.txt'.format(indir)))
    for attack in attack_types:
        normal_size += data_info['{}//Normal_{}'.format(indir, attack)]
    sources = ['{}//Normal_{}'.format(indir, a) for a in attack_types]
    dest = '{}/Normal/'.format(outdir)
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    train_test_split(sources, dest, normal_size, 
                    train_label_ratio=args.train_label_ratio)
    
if __name__ == '__main__':
    print("Entered main \n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/TFRecord")
    parser.add_argument('--outdir', type=str, default="./Data")
    parser.add_argument('--attack_type', type=str, nargs='+', default=['all'])
    parser.add_argument('--normal', type=bool, default=True)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--train_label_ratio', type=float, default=0.3)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    args = parser.parse_args()

    if args.attack_type[0] == 'all':
        # print('attack type all')
        # attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM', 'Smart']
        attack_types = ['DoS', 'gear', 'RPM', 'Smart']
    elif args.attack_type[0] == None:
        # print('attack type none')
        attack_types = None
    else:
        # print('args.attack_type')
        attack_types = args.attack_type
    
    if args.normal:
        main_normal(attack_types, args)
        print("---------------------xxxxxnormalxxxxx---------------------")

    if attack_types is not None:
        print("---------------------xxxxxattackxxxxx---------------------")
        main_attack(attack_types, args)



# def write_tfrecord(data, filename):
#     ###ORIGINAL
#     print('Writing {}================= '.format(filename))
#     iterator = data.make_one_shot_iterator().get_next()
#     init = tf.global_variables_initializer()
#     tfrecord_writer = tf.io.TFRecordWriter(filename)
#     with tf.Session() as sess:
#         sess.run(init)
#         while True:
#             try:
#                 batch_data = sess.run(iterator)
#                 for x, y in zip(batch_data['input_features'], batch_data['label']):
#                     tfrecord_writer.write(serialize_example(x, y))
#             except:
#                 break
            
#     tfrecord_writer.close()


# @tf.function
# def write_tfrecord(data, filename):
#     print("Writing {}================= ".format(filename))

#     iterator = tf.compat.v1.data.make_initializable_iterator(data)
#     next_element = iterator.get_next()

#     tfrecord_writer = tf.io.TFRecordWriter(filename)

#     with tf.compat.v1.Session() as sess:
#         sess.run(iterator.initializer)
#         while True:
#             try:
#                 batch_data = sess.run(next_element)
#                 for x, y in zip(batch_data['input_features'], batch_data['label']):
#                     tfrecord_writer.write(serialize_example(x, y))
#             except tf.errors.OutOfRangeError:
#                 break

#     tfrecord_writer.close()




# def write_tfrecord(data, filename):
#     print("Writing {}================= ".format(filename))
    
#     iterator = tf.data.make_initializable_iterator(data)
#     next_element = iterator.get_next()

#     tfrecord_writer = tf.io.TFRecordWriter(filename)

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(iterator.initializer)  # Initialize the iterator here

#         while True:
#             try:
#                 batch_data = sess.run(next_element)
#                 for x, y in zip(batch_data['input_features'], batch_data['label']):
#                     tfrecord_writer.write(serialize_example(x, y))
#             except tf.errors.OutOfRangeError:
#                 break

#     tfrecord_writer.close()




# def write_tfrecord(data, filename):
#     print("Writing {}================= ".format(filename))

#     # Ensure data is a TensorFlow Dataset
#     if not isinstance(data, tf.data.Dataset):
#         raise ValueError("Expected a tf.data.Dataset but got: {}".format(type(data)))

#     iterator = tf.compat.v1.data.make_one_shot_iterator(data)
#     next_element = iterator.get_next()

#     tfrecord_writer = tf.io.TFRecordWriter(filename)

#     with tf.compat.v1.Session() as sess:
#         while True:
#             try:
#                 batch_data = sess.run(next_element)
#                 for x, y in zip(batch_data['input_features'], batch_data['label']):
#                     tfrecord_writer.write(serialize_example(x, y))
#             except tf.errors.OutOfRangeError:
#                 break

#     tfrecord_writer.close()










# def write_tfrecord(data, filename):
#     print("NEW\n\n\n\n\n\n")
#     print('Writing {}================= '.format(filename))
#     # iterator = data.make_one_shot_iterator().get_next()
    

#     # iterator = tf.data.make_one_shot_iterator(data)

#     iterator = tf.data.make_initializable_iterator(data)
#     next_element = iterator.get_next()

#     # iterator = tf.compat.v1.data.make_one_shot_iterator(data)
#     # next_element = iterator.get_next()

#     # iterator = tf.compat.v1.data.make_one_shot_iterator(data)
    
#     # next_element = iterator.get_next()


#     # tf.disable_eager_execution()
#     init = tf.global_variables_initializer()
#     tfrecord_writer = tf.io.TFRecordWriter(filename)


#     with tf.Session() as sess:
#         sess.run(init)
#         while True:
#             try:
#                 print("IN TRY BLOCK")
#                 sess.run(iterator.initializer)
#                 batch_data = sess.run(next_element)
#                 for x, y in zip(batch_data['input_features'], batch_data['label']):
#                     # print(serialize_example(x,y))
#                     tfrecord_writer.write(serialize_example(x, y))
#             except tf.errors.OutOfRangeError:
#                 print("IN EXCEPT BLOCK")
#                 break
            
#     tfrecord_writer.close()

    
