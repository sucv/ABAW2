from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf

from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_postprocess
from vggish import vggish_slim

import soundfile as sf

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def extract_vggish(
        wav_file,
        window_sec=0.025,
        hop_sec=0.01,
):
    examples_batch = vggish_input.wavfile_to_examples(wav_file, window_sec=window_sec, hop_sec=hop_sec)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        # print(embedding_batch)
        # postprocessed_batch = pproc.postprocess(embedding_batch)
        # print(postprocessed_batch)

        # # Write the postprocessed embeddings as a SequenceExample, in a similar
        # # format as the features released in AudioSet. Each row of the batch of
        # # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # # the rows are written as a sequence of bytes-valued features, where each
        # # feature value contains the 128 bytes of the whitened quantized embedding.
        # seq_example = tf.train.SequenceExample(
        #     feature_lists=tf.train.FeatureLists(
        #         feature_list={
        #             vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
        #                 tf.train.FeatureList(
        #                     feature=[
        #                         tf.train.Feature(
        #                             bytes_list=tf.train.BytesList(
        #                                 value=[embedding.tobytes()]))
        #                         for embedding in embedding_batch
        #                     ]
        #                 )
        #         }
        #     )
        # )

        return embedding_batch
