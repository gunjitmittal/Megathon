import librosa
import numpy
import tensorflow as tf
from pydub import AudioSegment
from tensorflow import keras
import os
import json
import scipy.signal as sig
from sklearn.utils import shuffle

def mfcc_data(f_name):
 # 1. Get file path
 filename = f_name
 file_info = AudioSegment.from_mp3(filename)
 rate = file_info.frame_rate
 # 2. Load the audio as a waveform `y` Store the sampling rate as `sr`
 raw, sr = librosa.load(filename, sr=rate)
 # 3. Trim silence at start and end of audio
 y, index = librosa.effects.trim(raw)
 # 4. Filtering
 audio = sig.wiener(y)
 # 5. MFCC features
 mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
 mfcc_delta = librosa.feature.delta(mfcc)
 mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
 # 6. Labelling
 m, n = mfcc.shape
 feature = numpy.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
 slen = feature.shape
 feature = numpy.transpose(feature)
 return feature, slen[1]

labels = numpy.loadtxt("labels.dat",dtype="U")
Tests = []
Result = dict()
for test in os.listdir(os.fsencode('test')):
    filename = os.fsdecode(test)
    if filename == ".DS_Store":
     continue
    Tests.append(filename)


model = keras.models.load_model('Model_cherry_picked_10.h5')

for test in Tests:
 test_path = "test/" + test
 file_path = os.fsencode(test_path)
 features, length = mfcc_data(file_path)
 X = features[:-1]
 Xlen = length
 X = shuffle(X,random_state=1)
 prediction = labels[numpy.argmax(model.predict(X)[-1,:])]
 Result[test] = prediction
Result = json.dumps(Result)
print(Result)