import librosa
import pyaudio
import numpy as np

from scipy import signal
from keras.models import load_model

RATE = 22050
CHUNK = RATE * 3
FORMAT = pyaudio.paInt16
CHANNELS = 1
# Load the Detection Model
model_detection = load_model('C:\\Users\\dell\\Desktop\\AI\\Grad\\2nd\\New folder\\Siren\\Detection.h5')

# Load the Recognition Model
model_recognition = load_model('C:\\Users\\dell\\Desktop\\AI\\Grad\\2nd\\New folder\\Siren\\Recognision.h5')
sos = signal.butter(5, [50, 5000], 'bandpass', fs=RATE, output='sos')
detection_input_shape = model_detection.input_shape[1:]
recognition_input_shape = model_recognition.layers[0].input_shape[0][1:]
def preprocess_detection(audio_data):
    audio_data = signal.sosfilt(sos, audio_data)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=40)
    mfccs_padded = np.pad(mfccs, ((0, 0), (0, max(0, detection_input_shape[1] - mfccs.shape[1]))), mode='constant')
    mfccs_padded = mfccs_padded.reshape(detection_input_shape)
    mfccs_padded = np.expand_dims(mfccs_padded, axis=0)
    return mfccs_padded
def preprocess_recognition(audio_data):
    audio_data = audio_data[:RATE]
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(recognition_input_shape)
    mfccs_scaled_features = np.expand_dims(mfccs_scaled_features, axis=0)
    return mfccs_scaled_features
def real_time_detection_recognition():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    chosen_device_index = -1  # Adjust if necessary
    for x in range(p.get_device_count()):
        info = p.get_device_info_by_index(x)

    # Open stream for real-time audio input
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index=chosen_device_index,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    Detection_Threshold = 0.5
    while True:
        # Read audio data from stream
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # # Convert audio data to floating-point format
        audio_data = audio_data.astype(np.float32) / 32767.0
        audio_data = 2 * ((audio_data - min(audio_data)) / (max(audio_data) - min(audio_data))) - 1
        
        # Detect siren using detection model
        prediction_feature = preprocess_detection(audio_data)
        predicted_proba_vector = model_detection.predict(prediction_feature, verbose=0)
        siren_prob = predicted_proba_vector[0][1]

        # If siren is detected, perform recognition
        if siren_prob > Detection_Threshold:
            preprocessed_data_recognition = preprocess_recognition(audio_data)
            predicted_class = np.argmax(model_recognition.predict(preprocessed_data_recognition, verbose=0)[0])
            siren_type = {0: "Ambulance", 1: "Firetruck", 2: "Traffic"}[predicted_class]
            print(f"SIREN DETECTED! Type: {siren_type}")
        else:
            print(f"No siren detected, Certainty: {(siren_prob * 100):.2f}%")

    # Close the audio stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()
if __name__ == "__main__":
    try:
        real_time_detection_recognition()
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")