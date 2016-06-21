'''
This module contains functions to create the Video2GIF network and load the weights
as well as some helper functions, e.g. for generating the final GIF files.
For more information on the method, see

 Michael Gygli, Yale Song, Liangliang Cao
    "Video2GIF: Automatic Generation of Animated GIFs from Video," IEEE CVPR 2016
'''

__author__ = 'Michael Gygli'
import ConfigParser
import Queue
import collections
import numpy as np
import os
import threading
import time
import model

try:
    import lasagne
    import theano
except (ImportError,AssertionError) as e:
    print(e.message)

# Load the configuration
config=ConfigParser.SafeConfigParser()
print('Loaded config file from %s' % config.read('%s/config.ini' % os.path.dirname(__file__))[0])

# Load the mean snipplet (for mean subtraction)
snipplet_mean = np.load(config.get('paths','snipplet_mean'))

def get_prediction_function(feature_layer = None):
    '''
    Get prediction function (C3D and Video2GIF combined)
    @param feature_layer: a layer name (see model.py). If provided, pred_fn returns (score, and the activations at feature_layer)
    @return: theano function that scores sniplets
    '''
    print('Load weights and compile model...')

    # Build model
    net= model.build_model(batch_size=2)

    # Set the weights (takes some time)
    model.set_weights(net['score'],config.get('paths','c3d_weight_file'),config.get('paths','video2gif_weight_file'))
    layer='score'
    prediction = lasagne.layers.get_output(net[layer], deterministic=True)
    if feature_layer:
        features = lasagne.layers.get_output(net[feature_layer], deterministic=True)
        pred_fn = theano.function([net['input'].input_var], [prediction, features], allow_input_downcast = True)
    else:
        pred_fn = theano.function([net['input'].input_var], prediction, allow_input_downcast = True)


    return pred_fn

def get_scores(predict, segments, video, stride=8, with_features=False):
    '''
    Predict scores for segments using threaded loading
    (see https://github.com/Lasagne/Lasagne/issues/12#issuecomment-59494251)

    NOTE: Segments shorter than 16 frames (C3D input) don't get a prediction

    @param predict: prediction function
    @param segments: list of segment tuples
    @param video: moviepy VideoFileClip
    @param stride: stride of the extraction (8=50% overlap, 16=no overlap)
    @return: dictionary key: segment -> value: score
    '''

    queue = Queue.Queue(maxsize=50)
    sentinel = object()  # guaranteed unique reference

    def produce_input_data():
        '''
        Function to generate sniplets that serve as input to the network
        @return:
        '''
        frames=[]
        seg_nr=0

        for frame_idx, f in enumerate(video.iter_frames()):
            if frame_idx > segments[seg_nr][1]:
                seg_nr+=1
                if seg_nr==len(segments):
                    break
                frames=[]

            frames.append(f)

            if len(frames)==16: # Extract scores
                snip = model.get_snips(frames,snipplet_mean,0,True)
                queue.put((segments[seg_nr],snip))
                frames=frames[stride:] # shift by 'stride' frames

        queue.put(sentinel)

    def get_input_data():
        '''
        Iterator reading snipplets from the queue
        @return: (segment,snip)
        '''
        # run as consumer (read items from queue, in current thread)
        item = queue.get()
        while item is not sentinel:
            yield item
            queue.task_done()
            item = queue.get()


    # start producer (in a background thread)
    thread = threading.Thread(target=produce_input_data)
    thread.daemon = True

    segment2score=collections.OrderedDict()
    features=collections.OrderedDict()

    start=time.time()
    thread.start()
    print('Score segments...')

    for segment,snip in get_input_data():
        # only add a segment, once we certainly get a prediction
        if segment not in segment2score:
            segment2score[segment]=[]
            features[segment]=[]
        if with_features:
            scores,feat=predict(snip)
            features[segment].append(feat.mean(axis=0))
        else:
            scores=predict(snip)
        segment2score[segment].append(scores.mean(axis=0))

    for segment in segment2score.keys():
        segment2score[segment]=np.array(segment2score[segment]).mean(axis=0)
        if with_features:
            features[segment]=np.array(features[segment]).mean(axis=0)

    print("Extracting scores for %d segments took %.3fs" % (len(segments),time.time()-start))
    if with_features:
        return segment2score, features
    else:
        return segment2score

def generate_gifs(out_dir, segment2scores, video, video_id, top_k=6, bottom_k=0):
    '''
    @param out_dir: directory where the GIFs are written to
    @param segment2scores: a dict with segments (start frame, end frame) as keys and the segment score as value
    @param video: a VideoFileClip object
    @param video_id: the identifier of the video (used for naming the GIFs)
    @return:
    '''
    segment2scores = segment2scores.copy()

    nr=0
    top_k=min(top_k, len(segment2scores))
    good_gifs=[]
    for segment in sorted(segment2scores, key=lambda x: -segment2scores.get(x))[0:top_k]:

        clip = video.subclip(segment[0]/float(video.fps), segment[1]/float(video.fps))
        out_gif = "%s/%s_%.2d.gif" % (out_dir,video_id,nr)
        clip=clip.resize(height=240)
        clip.write_gif(out_gif,fps=10)
        good_gifs.append((video_id, nr, segment[0], segment[1], segment2scores[segment]))
        nr += 1

    bottom_k=min(bottom_k, len(segment2scores))
    bad_gifs=[]
    nr=len(segment2scores)
    for segment in sorted(segment2scores, key=segment2scores.get)[0:bottom_k]:
        clip = video.subclip(segment[0]/float(video.fps), segment[1]/float(video.fps))
        clip=clip.resize(height=240)
        clip.write_gif("%s/%s_%.2d.gif" % (out_dir,video_id,nr),fps=10)
        bad_gifs.append((video_id, nr, segment[0], segment[1], segment2scores[segment]))
        nr -= 1

    return good_gifs,bad_gifs
