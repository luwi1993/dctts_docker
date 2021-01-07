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


def synthesize(domain="outside"):
    info = {}

    absolute_beginning = time.time()
    info["start_time"] = absolute_beginning
    # Load data
    if domain == "outside":
        L = load_data("synthesize", domain=domain)
    elif domain == "inside":
        
    # Load graph
    synth_graph = Graph(mode="synthesize");
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
                sess.run([synth_graph.global_step, synth_graph.Y, synth_graph.max_attentions, synth_graph.alignments],
                         {synth_graph.L: L,
                          synth_graph.mels: Y,
                          synth_graph.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

            # from the start time of the synthesis until the first time we reach this point is called the Latency
            Latency_beginning = time.time() - absolute_beginning
            Latency_synthesis = time.time() - begin_of_frame_synthesis

        duration_mels = time.time() - begin_of_frame_synthesis

        mels = {}
        for i, mel in enumerate(Y):
            mels["/{}.wav".format(i + 1)] = mel

        # Get magnitude
        Z = sess.run(synth_graph.Z, {synth_graph.Y: Y})
        duration_mags = time.time() - begin_of_frame_synthesis
        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)

        samples = {}
        mags = {}
        for i, mag in enumerate(Z):
            print("Working on file", i + 1)
            mags["/{}.wav".format(i + 1)] = mag
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i + 1), hp.sr, wav)
            samples["/{}.wav".format(i + 1)] = wav

        duration_total = begin_of_frame_synthesis
        time_measurents = {
                            "init_time":init_time,
                            "Latency_beginning":Latency_beginning,
                            "Latency_synthesis":Latency_synthesis,
                            "duration_mels":duration_mels,
                            "duration_mags":duration_mags,
                            "duration_total":duration_total,
        }

        info["mels"] = mels
        info["mags"] = mags
        info["samples"] = samples
        info["time_measurents"] = time_measurents

        return info


if __name__ == '__main__':
    time_measurements = synthesize()
    print("Done")
