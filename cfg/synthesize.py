# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
import time


def synthesize():
    absolute_beginning = time.time()
    # Load data
    L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize");
    print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        begin_of_frame_synthesis = time.time()
        init_time = begin_of_frame_synthesis - absolute_beginning

        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

            # from the start time of the synthesis until the first time we reach this point is called the Latency
            Latency_beginning = time.time() - absolute_beginning
            Latency_synthesis = time.time() - begin_of_frame_synthesis

        duration_mels = time.time() - begin_of_frame_synthesis
        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})
        duration_mags = time.time() - begin_of_frame_synthesis
        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i + 1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i + 1), hp.sr, wav)

        duration_total = begin_of_frame_synthesis
        time_measurents = {
                            "init_time":init_time,
                            "Latency_beginning":Latency_beginning,
                            "Latency_synthesis":Latency_synthesis,
                            "duration_mels":duration_mels,
                            "duration_mags":duration_mags,
                            "duration_total":duration_total,
        }
        return time_measurents


if __name__ == '__main__':
    time_measurements = synthesize()
    print("Done")
