from flask import Flask, request, jsonify,render_template
import os
import re

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Audio
from keras import layers
from keras import models
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import itertools
import visualkeras

def noise(data, random=False, rate=0.035, threshold=0.075):
    """Add some noise to sound sample. Use random if you want to add random noise with some threshold.
    Or use rate Random=False and rate for always adding fixed noise."""
    if random:
        rate = np.random.random() * threshold
    noise_amp = rate*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    """Stretching data with some rate."""
    return librosa.effects.time_stretch(data, rate=0.5)

def shift(data, rate=1000):
    """Shifting data with some rate"""
    shift_range = int(np.random.uniform(low=-5, high = 5)*rate)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor, random=False):
    """"Add some pitch to sound sample. Use random if you want to add random pitch with some threshold.
    Or use pitch_factor Random=False and rate for always adding fixed pitch."""
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

n_fft = 2048
hop_length = 512

# Zero Crossing Rate
def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_features(path, duration=2.5, offset=0.6):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)

     # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data, random=True)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) # stacking vertically

#     # data with pitching
    pitched_data = pitch(data, sample_rate, pitch_factor=3, random=False)
    
    res3 = extract_features(pitched_data, sample_rate)
    result = np.vstack((result, res3)) # stacking vertically

#     # data with pitching and noise
    new_data = pitch(data, sample_rate, pitch_factor=3, random=False)
    data_noise_pitch = noise(new_data, random=True)
    res3 = extract_features(data_noise_pitch, sample_rate)
    result = np.vstack((result, res3)) # stacking vertically

    return result

app = Flask(__name__)

model = load_model('model.h5')
@app.route('/')
def home():
    return render_template('i1.html')

# Create a dictionary to map numeric labels to sentiment labels
sentiment_mapping = {
    0: 'Sad',
    1: 'Angry',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Neutral'
}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        if audio_file.filename != '':
            # Get the filename of the uploaded audio file
            audio_filename = audio_file.filename
            k = 'final_d/' + audio_filename
            features = get_features(k)
            features = np.expand_dims(features, axis=2)
            features = np.pad(features, ((0, 0), (0, 880), (0, 0)), mode='constant')
            
            # Predict sentiment using the model
            sentiment_numeric = model.predict(features)
            
            # Map the numeric sentiment to a label
            sentiment_label = sentiment_mapping[np.argmax(sentiment_numeric)]
            
            return render_template('i1.html', prediction=sentiment_label)
    
    return render_template('i1.html', prediction='No file found')


if __name__ == "__main__":
    app.run(debug=True)