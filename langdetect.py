import os
import random
import numpy as np
from scipy import stats
#%%
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import Dropout
import keras.optimizers

#%%
if tf.test.is_gpu_available():
    print('GPU Available')
    
BATCH_SIZE = 512  # Number of images used in each iteration
EPOCHS = 10  # Number of passes through entire dataset
    
#%%
LANGUAGES_DICT = {'en':0,'fr':1,'es':2,'it':3,'de':4}
CODE_NAMES = {'en':'English','fr':'French','es':'Spanish','it':'Italian','de':'German'}
# Length of cleaned text used for training and prediction - 140 chars
MIN_LEN = 140
NUM_SAMPLES = 250000
RANDOM_STATE = 42

from Data_Cleaning import define_alphabet
# Load the Alphabet
alphabet = define_alphabet()
print('All aphabets are:',end="  ")
print(alphabet[2])  #string of all letters
"""
define_alphabet() returns 3 things: lowercase characters, uppercase characters and 
all characters together.
SO, alphabet[2] is all characters together
"""

VOCAB_SIZE = len(alphabet[2])
"""
size of the entire vocabulary, i.e. all distinct characters will
be equal to total length of all language characters
"""
print('ALPHABET LEN(VOCAB SIZE):', VOCAB_SIZE)

# Folders from where load / store the raw, source, cleaned, samples and train_test data
data_directory = "D:/GAIP/data"
source_directory = os.path.join(data_directory, 'source')
cleaned_directory = os.path.join(data_directory, 'cleaned')
samples_directory = os.path.join('/tmp', 'samples')
train_test_directory = os.path.join('/tmp', 'train_test')

#%%
from Data_Cleaning import get_sample_text, get_input_row

# last part calculates also input_size for DNN so this code must be run before DNN is trained
path = os.path.join(cleaned_directory, "es_cleaned.txt")
with open(path, 'r',encoding='utf-8') as f:
    content = f.read()
    """will read de_cleaned.txt for now"""
    random_index = random.randrange(0,len(content))
    
    sample_text = get_sample_text(content,random_index,MIN_LEN)
    print ("1. SAMPLE TEXT: \n", sample_text)
    print ("\n2. REFERENCE ALPHABET: \n", alphabet[0]+alphabet[1])
    """lower and uppercase letters displayed"""
    
    sample_input_row = get_input_row(content, random_index, MIN_LEN, alphabet)
    print ("\n3. SAMPLE INPUT ROW: \n",sample_input_row)
    
    input_size = len(sample_input_row)
    if input_size != VOCAB_SIZE:
        print("Something strange happened!")
        
    print ("\n4. INPUT SIZE (VOCAB SIZE): ", input_size)
    #del content
# Now we will apply the transformation from raw text to Bag of Characters representation for all the data we have collected. At the end of the proprocessing, we will have 250k samples per language where every sample will be piece of text 140 characters long, represented using the Bag of Characters model.
# 
# Dataset dimension (1750k, 133):
# - rows: 1750k (250k * 7) or (NUM_SAMPLES * num_languages)
# - columns: 133 (132 + 1) or (VOCAB_SIZE + language_index)

#%%
# Utility function to return file Bytes size in MB
# just for display
def size_mb(size):
    # display formatting for no special reason
    size_mb =  '{:.2f}'.format(size/(1000*1000.0))
    return size_mb + " MB"
#%%
# prepare numpy array
# empty initial array and specified datatype of int16
sample_data = np.empty((NUM_SAMPLES * len(LANGUAGES_DICT),input_size+1),dtype = np.uint16)
print(sample_data)
stats.describe(sample_data)
offset = 0 # offset for each language data
#%%
for lang_code in LANGUAGES_DICT:
    start_index = 0
    path = os.path.join(cleaned_directory, lang_code+"_cleaned.txt")
    with open(path, 'r',encoding='utf-8') as f:
        print ("Processing file : " + path)
        file_content = f.read()
        content_length = len(file_content)
        remaining = content_length - MIN_LEN*NUM_SAMPLES
        print("remaining is",content_length,'-',MIN_LEN,'*',NUM_SAMPLES,'=',remaining)
        jump = int((remaining/NUM_SAMPLES))
        print('jump = ',jump)
        print ("File size : ",size_mb(content_length)," | # possible samples : ",int(content_length/VOCAB_SIZE),"| # skip chars : " + str(jump))
        for idx in range(NUM_SAMPLES):
            input_row = get_input_row(file_content, start_index, MIN_LEN, alphabet)
            sample_data[NUM_SAMPLES*offset+idx,] = input_row + [LANGUAGES_DICT[lang_code]]
            if idx%10000==0:
                print("storing data at sample[",NUM_SAMPLES*offset+idx,"]")
            start_index += MIN_LEN + jump
        del file_content
    offset += 1
    print (100*"-")
#%%     
np.random.shuffle(sample_data)
# reference input size
print ("Vocab Size : ",VOCAB_SIZE )
print (100*"-")
print ("Samples array size : ",sample_data.shape )
#%%
# Create the the sample dirctory if not exists
if not os.path.exists(samples_directory):
    os.makedirs(samples_directory)

# Save compressed sample data to disk
path_smpl = os.path.join(samples_directory,"lang_samples_"+str(VOCAB_SIZE)+".npz")
np.savez_compressed(path_smpl,data=sample_data)
print(path_smpl, "size : ", size_mb(os.path.getsize(path_smpl)))


#%%
def decode_langid(langid):    
    for dname, did in LANGUAGES_DICT.items():
        if did == langid:
            temp = langid
    for key,value in CODE_NAMES.items():
        if key==temp:
            return value
#%%
path_smpl = os.path.join(samples_directory,"lang_samples_"+str(VOCAB_SIZE)+".npz")
dt = np.load(path_smpl)['data']

random_index = random.randrange(0,dt.shape[0])
print ("Sample record : \n",dt[random_index,])
print ("\nSample language : ",decode_langid(dt[random_index,][VOCAB_SIZE]))
#%%
# Check if the data have equal share of different languages
print ("\nDataset shape (Total_samples, Alphabet):", dt.shape)
bins = np.bincount(dt[:,input_size])
#%%
print ("Language bins count (samples per language): ") 
for lang_code in LANGUAGES_DICT: 
    print (lang_code, bins[LANGUAGES_DICT[lang_code]])


#%%
# change the datatype of data
dt = dt.astype(np.float32)
# X and Y split
X = dt[:, 0:input_size] # Samples
Y = dt[:, input_size] # The last element is the label
del dt
np.save('X_data_new.npy',X)
np.save('Y_data_new.npy',Y)
# Random index to check random sample
random_index = random.randrange(0,X.shape[0])
print("Example data before processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])
#%%
# X PREPROCESSING
# Feature Standardization - Standar scaler
standard_scaler = preprocessing.StandardScaler().fit(X)
X = standard_scaler.transform(X)   
print ("X preprocessed shape :", X.shape)

# One-hot encoding
Y = keras.utils.to_categorical(Y, num_classes=len(LANGUAGES_DICT))

# See the sample data
print("\nExample data after processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])

np.save('X_data.npy',X)
np.save('Y_data.npy',Y)
# Train/test split. Static seed to have comparable results for different runs
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_STATE)

# Create the train / test directory if not exists
if not os.path.exists(train_test_directory):
    os.makedirs(train_test_directory)

# Save compressed train_test data to disk
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(VOCAB_SIZE)+".npz")
np.savez_compressed(path_tt,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
print(path_tt, "size : ",size_mb(os.path.getsize(path_tt)))
#del X_train,Y_train,X_test,Y_test


#%%

path_tt = os.path.join(train_test_directory, "train_test_data_"+str(VOCAB_SIZE)+".npz")
train_test_data = np.load(path_tt)

X_train = train_test_data['X_train']
Y_train = train_test_data['Y_train']

X_test = train_test_data['X_test']
Y_test = train_test_data['Y_test']

#del train_test_data

#%%

model = Sequential()


model.add(Dense(256,input_dim=input_size))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

model.add(Dense(512, kernel_initializer="glorot_uniform"))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

#model_optmizer = keras.optimizer.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e_09, decay=0.0)

model.add(Dense(len(LANGUAGES_DICT)))
model.add(Activation('softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
model.summary()

#%%
history = model.fit(X_train,Y_train, epochs=EPOCHS, validation_data=(X_train,Y_train), batch_size=BATCH_SIZE)

#%%
# Evaluation on Test set
scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%%
import matplotlib.pyplot as plt
plt.plot([history.history['acc']])
plt.show()

#%%

Y_pred = model.predict_classes(X_test)
Y_pred = keras.utils.to_categorical(Y_pred, num_classes=len(LANGUAGES_DICT))
LABELS =  list(LANGUAGES_DICT.keys())


#%%

# Plot confusion matrix 
from sklearn.metrics import confusion_matrix
from Data_Cleaning import print_confusion_matrix

cnf_matrix = confusion_matrix(np.argmax(Y_pred,axis=1), np.argmax(Y_test,axis=1))
print_confusion_matrix(cnf_matrix, LABELS)


#%% only uncomment the multiline string if running on Jupyter notebooks 
# make sure ipywidgets is installed in Anaconda
"""
print(classification_report(Y_test, Y_pred, target_names=LABELS))


from ipywidgets import interact_manual
from ipywidgets import widgets
from Data_Cleaning import clean_text


def get_prediction(TEXT):
    if len(TEXT) < MIN_LEN:
        print("Text has to be at least {} chars long, but it is {}/{}".format(MIN_LEN, len(TEXT), MIN_LEN))
        return(-1)
    # Data cleaning
    cleaned_text = clean_text(TEXT)
    
    # Get the MIN_LEN char
    input_row = get_input_row(cleaned_text, 0, MIN_LEN, alphabet)
    
    # Data preprocessing (Standardization)
    test_array = standard_scaler.transform([input_row])
    
    raw_score = model.predict(test_array)
    pred_idx= np.argmax(raw_score, axis=1)[0]
    score = raw_score[0][pred_idx]*100
    
    # Prediction
    prediction = LABELS[model.predict_classes(test_array)[0]]
    print('TEXT:', TEXT, '\nPREDICTION:', prediction.upper(), '\nSCORE:', score)

interact_manual(get_prediction, TEXT=widgets.Textarea(placeholder='Type the text to identify here'));
model.save_weights('lang_identification_weights.h5')"""
#%%
"""
Secondary network structure
"""
"""
model.add(Dense(512,input_dim=input_size)
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation="sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(len(LANGUAGES_DICT), activation="softmax"))
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],optimizer='adam')
"""
