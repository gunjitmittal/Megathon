from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import get_data
import csv
import numpy
# Load Database
train_x, train_y, train_data_order, train_len = get_data.read_dataset('train')
test_x, test_y, test_data_order, test_len = get_data.read_dataset('test')
train_x = numpy.asarray(train_x).astype(numpy.float32)
train_y = numpy.asarray(train_y).astype(numpy.float32)
test_x = numpy.asarray(test_x).astype(numpy.float32)
test_y = numpy.asarray(test_y).astype(numpy.float32)
print(numpy.unique(test_y))
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# MLP Model
model = Sequential()
model.add(Dense(64, input_dim=39, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(122, activation='sigmoid'))
# Compile model
opt = 'adadelta'
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
# Fit the model
import time
start_time = time.time()
history =model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=100,batch_size=1024)
end_time = time.time()-start_time
print("---------------------- %s seconds ---------------------" % end_time)
print(history.history)
# Save data and plot
hist = [history.history['loss'], history.history['val_loss'], history.history['accuracy'],history.history['val_accuracy']]
# evaluate the model
scores = model.evaluate(train_x, train_y)
print("\nTrain.......%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(test_x, test_y)
print("\nTest........%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Save accuracy and loss history
fname = "MLP_"+opt+".csv"
model.save("./Model.h5")
with open(fname, 'w', newline='') as csv_file:
 wr = csv.writer(csv_file)
 wr.writerows(hist)