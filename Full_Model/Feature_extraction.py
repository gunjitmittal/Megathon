import librosa.display
import numpy as np
import csv
import os
from pydub import AudioSegment
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
 cluster = np.full((1, n), str(l))
 feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
 slen = feature.shape
 feature = np.concatenate((feature, cluster), axis=0)
 feature = np.transpose(feature)
 return feature, slen[1]

if __name__ == "__main__":
 Lang = []
 for lang in os.listdir(os.fsencode('train')):
  langname = os.fsdecode(lang)
  if langname == ".DS_Store":
   continue
  Lang.append(langname[0:])
 np.savetxt("labels.dat", shuffle(Lang, random_state=1),fmt='%s')
 test_train_split = 0.8
 i = np.zeros(len(Lang))
 temp = []
 total = []
 Samples = 0
 train_database = 1
 test_database = 1
 doc = 1
 if os.path.exists('train_data.csv') and doc == 1:
  os.remove("train_data.csv")
  print("File Removed!")
 for l in Lang:
  print(l)
  l_path = "train/" + l
  directory = os.fsencode(l_path)
  xlen = len(os.listdir(directory))
  if xlen<5:
   test_train_split = 0.5
  else:
   test_train_split = 0.8
  for file in os.listdir(directory):
   # Feature extraction
   if (i[Lang.index(l)] >= test_train_split*xlen):
    break
   features, length = mfcc_data(os.path.join(directory, os.fsencode(file)))
   total.append(length)
   with open("train_data.csv", 'a', newline='') as csv_file:
     wr = csv.writer(csv_file)
     wr.writerows(features)
   i[Lang.index(l)]+=1
 # save window length of audio
 if os.path.exists('train_datalen.csv') and doc == 1:
  os.remove("train_datalen.csv")
  print("File Removed!")
 with open("train_datalen.csv", 'w', newline='') as csv_file:
  wr = csv.writer(csv_file)
  wr.writerows(([x] for x in total))
 # Test Database
 total = []
 if os.path.exists('test_data.csv') and doc == 1:
  os.remove("test_data.csv")
  print("File Removed!")

 for l in Lang:
  print(l)
  l_path = "train/" + l
  directory = os.fsencode(l_path)
  # xlen = len(os.listdir(directory))
  # if xlen < 5:
  #  test_train_split = 0.5
  # else:
  #  test_train_split = 0.8
  for file in os.listdir(directory)[int(i[Lang.index(l)]):]:
   # Feature extraction
   features, length = mfcc_data(os.path.join(directory, os.fsencode(file)))
   total.append(length)
   with open("test_data.csv", 'a', newline='') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerows(features)
 # save window length of audio
 if os.path.exists('test_datalen.csv') and doc ==1:
  os.remove("test_datalen.csv")
  print("File Removed!")
 with open("test_datalen.csv", 'w', newline='') as csv_file:
  wr = csv.writer(csv_file)
  wr.writerows(([x] for x in total))