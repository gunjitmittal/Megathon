import numpy as np
import tensorflow as tf
from tensorflow import keras
import Feature_extraction
import os
import json

Tests = []
Result = dict()
for test in os.listdir(os.fsencode('test')):
    filename = os.fsdecode(test)
    if filename == ".DS_Store":
     continue
    Tests.append(filename)


a = Feature_extraction.mfcc_data(Tests)
model = keras.models.load_model('Model')

for test in Tests:
 test_path = "test/" + test
 file_path = os.fsencode(test_path)
 features, length = Feature_extraction.mfcc_data(file_path)
 X = features[:-1]
 Xlen = length
 X = np.shuffle(X,random_state=1)
 prediction = model.predict(X)
 Result[test] = prediction
Result = json.dumps(Result)
print(Result)