from flask import Flask, request, render_template, Response, flash, redirect, session, url_for, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import biosppy
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import set_session
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from werkzeug.utils import secure_filename
from scipy.signal import find_peaks
import itertools
from scipy import interpolate
from seasonal import fit_seasons, adjust_seasons
import scipy as sp
import scipy.io
from scipy.signal import find_peaks
import random
import math
import pywt
from numpy import mean, sqrt, square, arange
from scipy.interpolate import UnivariateSpline

app = Flask(__name__)
app.config['SECRET_KEY'] = b'b\xa1\xe8\x0bs\x06\xae\xfd\xa8\xcc~!9' 
                 

def get_model():
    global model
    global modelpe
    global modelafib
    global modelVfib
    global modeleupe
    global modelbp
    model = load_model('modelarxivptbmit-020-0.991439-0.984750.h5')
    modelpe = load_model('phonewatchmodelmimicfilter2.h5')
    modelafib = load_model('resnetlstmNAFvAF2aug.h5')
    modelVfib = load_model('vf_cnnarxiv-86.29.h5')
    modeleupe = load_model('euwatchmodelmimicBPF.h5')
    modelbp = load_model('unet250norm_300e.h5')
    print(" * Models loaded!")


def predict_data(model, X):
    
    Predictions = np.zeros(shape=(len(X),1))
    Probability = np.zeros(shape=(len(X),1))
    for i in range(len(X)):
        Predictions[i] = model.predict(X[i]).argmax(axis=-1)
        Probability[i] = round(model.predict(X[i])[0][int(Predictions[i][0])]*100, 3)
    
    result = list(zip(Predictions, Probability))    
    PredictionsT = pd.DataFrame(result , columns=['Prediction', 'Probability' ])
    PredictionsT.index.names = ['Beat']
    PredictionsT['Prediction'].replace({0.0:'Normal', 1.0:'Abnormal'}, inplace=True)
    
    return PredictionsT


def resampler(datanp, fs, fs_rs):
    time = np.arange(len(datanp)) * 1 / fs
    times_rs = np.arange(0, time[-1], 1 / fs_rs)
    interp_func = interpolate.interp1d(x=time, y=datanp.T, kind='linear')
    values_rs = interp_func(times_rs)
    results = values_rs.T
    return results

def AnalyzeWindow(ppg, accx, accy, accz, Fs=125, verbose=False):
    
    ppg_bandpass = BandpassFilter(ppg, fs=Fs)
    accx_bandpass = BandpassFilter(accx, fs=Fs)
    accy_bandpass = BandpassFilter(accy, fs=Fs)
    accz_bandpass = BandpassFilter(accz, fs=Fs)
    
    # Aggregate accelerometer data into single signal
    
    accy_mean = accy-np.mean(accy_bandpass) # Center Y values
    acc_mag_unfiltered = np.sqrt(accx_bandpass**2+accy_mean**2+accz_bandpass**2)
    acc_mag = BandpassFilter(acc_mag_unfiltered, fs=Fs)
    
    negppg = -ppg_bandpass
    peaks = find_peaks(negppg , distance=40)[0]
    
    fs = Fs
    RR_list = []
    cnt = 0
    while (cnt < (len(peaks)-1)):
        RR_interval = (peaks[cnt+1] - peaks[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    bpm = 60000 / np.mean(RR_list)
    confidence = 0
        
    # Use FFT length larger than the input signal size for higher spectral resolution.
    fft_len=len(ppg_bandpass)*4

    # Create an array of frequency bins
    freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031

    # The frequencies between 40 BPM and 240 BPM Hz
    low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
    
    mag_freq_ppg, fft_ppg = FreqTransform(ppg_bandpass, freqs, low_freqs, fft_len)
    mag_freq_acc, fft_acc = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
    
    peaks_ppg = find_peaks(mag_freq_ppg, height=20, distance=1)[0]
    peaks_acc = find_peaks(mag_freq_acc, height=30, distance=1)[0]
    
    if len(peaks_ppg)!=0:
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)

        # Use the frequency peak with the highest magnitude, unless the peak is also present in the accelerometer peaks.
        use_peak = sorted_freq_peaks_ppg[0]
        for i in range(len(sorted_freq_peaks_ppg)):
            # Check nearest two peaks also
            cond1 = sorted_freq_peaks_ppg[i] in sorted_freq_peaks_acc
            cond2 = sorted_freq_peaks_ppg[i]-1 in sorted_freq_peaks_acc
            cond3 = sorted_freq_peaks_ppg[i]+1 in sorted_freq_peaks_acc
            if cond1 or cond2 or cond3:
                continue
            else:
                use_peak = sorted_freq_peaks_ppg[i]
                break

        chosen_freq = freqs[low_freqs][use_peak]
        prediction = chosen_freq * 60
        confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
       
    else: 
        prediction = bpm

        
    return (fft_ppg, fft_acc, prediction, confidence)


def AnalyzeWindow2(ppg, accx, accy, accz, Fs=125, verbose=False):
    
    ppg_bandpass = BandpassFilter(ppg, fs=Fs)
    accx_bandpass = BandpassFilter(accx, fs=Fs)
    accy_bandpass = BandpassFilter(accy, fs=Fs)
    accz_bandpass = BandpassFilter(accz, fs=Fs)
    
    # Aggregate accelerometer data into single signal
    
    accy_mean = accy-np.mean(accy_bandpass) # Center Y values
    acc_mag_unfiltered = np.sqrt(accx_bandpass**2+accy_mean**2+accz_bandpass**2)
    acc_mag = BandpassFilter(acc_mag_unfiltered, fs=Fs)
    
    peaks = find_peaks(ppg_bandpass, height = 10, distance=30)[0]
    
        
    # Use FFT length larger than the input signal size for higher spectral resolution.
    fft_len=len(ppg_bandpass)*4

    # Create an array of frequency bins
    freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031

    # The frequencies between 40 BPM and 240 BPM Hz
    low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
    
    mag_freq_ppgf, fft_ppg = FreqTransform(ppg_bandpass, freqs, low_freqs, fft_len)
    mag_freq_accf, fft_acc = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
    
    peaks_ppg = find_peaks(mag_freq_ppgf, height=20, distance=1)[0]
    peaks_acc = find_peaks(mag_freq_accf, height=20, distance=1)[0]
    
    mag_freq_ppg = np.abs(fft_ppg)
    ang_freq_ppg = np.angle(fft_ppg)
    mag_freq_acc = np.abs(fft_acc)
    phase_info = np.exp(1.0j* ang_freq_ppg)
    output = (0.88*mag_freq_ppg) - (0.7*mag_freq_acc)
    
    x_clip = np.clip(output, a_min = 0.0, a_max = None)
    X = x_clip* phase_info
    ifft_diff = np.fft.irfft(X)
    new_ppg = ifft_diff[:len(ppg)]
    
    negppg = -new_ppg
    peaks = find_peaks(negppg , distance=30)[0]
    
    fs = Fs
    RR_list = []
    cnt = 0
    while (cnt < (len(peaks)-1)):
        RR_interval = (peaks[cnt+1] - peaks[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    bpm = 60000 / np.mean(RR_list)
    confidence = 0
    
    peaks_ppg_clip = find_peaks(x_clip[low_freqs], height=1, distance=1)[0]
    
    if len(peaks_ppg_clip)!=0:
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)
        sorted_freq_peaks_ppg_clip = sorted(peaks_ppg_clip, key=lambda i:x_clip[low_freqs][i], reverse=True)
        use_peak = sorted_freq_peaks_ppg_clip[0]
        # Use the frequency peak with the highest magnitude, unless the peak is also present in the accelerometer peaks.

        chosen_freq = freqs[low_freqs][use_peak]
        prediction = chosen_freq * 60
        confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
        
    else:
        prediction = bpm
        
        
    return (fft_ppg, fft_acc, prediction, confidence, new_ppg)

def ecg_filter(datanp, threshold):
    ecg = datanp
    index = []
    data = []
    for i in range(len(ecg)-1):
        Y = float(ecg[i])
        data.append(Y)
    w = pywt.Wavelet ('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = threshold # Threshold for filtering 0.17
    coeffs = pywt.wavedec (data, 'db8', level = maxlev) # signal is decomposed by wavelet
    for i in range(1, len(coeffs)):
        coeffs [i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i])) # noise filter
    datarec = pywt.waverec (coeffs, 'db8') # signal wavelet reconstruction
    return datarec


def BandpassFilter(signal, fs):
    
    # Convert to Hz
    lo, hi = 40/60, 240/60
    
    b, a = sp.signal.butter(3, (lo, hi), btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, signal)


def BandpassFilter2(signal, fs):
    '''Bandpass filter the signal between 6 and 24 breaths per min'''
    
    # Convert to Hz
    lo, hi = 6/60, 24/60
    
    b, a = sp.signal.butter(3, (lo, hi), btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, signal)


def FreqTransform(x, freqs, low_freqs, fft_len):
    
    # Take an FFT of the normalized signal
    norm_x = (x - np.mean(x))/(max(x)-min(x))
    fft_x = np.fft.rfft(norm_x, fft_len)

    # Calculate magnitude of the lower frequencies
    mag_freq_x = np.abs(fft_x)[low_freqs]
    
    return mag_freq_x, fft_x


def CalcConfidence(chosen_freq, freqs, fft_ppg):

    win = (40/60.0)
    win_freqs = (freqs >= chosen_freq - win) & (freqs <= chosen_freq + win)
    abs_fft_ppg = np.abs(fft_ppg)
    
    # Sum frequency spectrum near pulse rate estimate and divide by sum of entire spectrum
    conf_val = np.sum(abs_fft_ppg[win_freqs])/np.sum(abs_fft_ppg)
    
    return conf_val

    
print(" * Loading model...!")
get_model()

@app.route('/upload', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[1000:]).astype(float)
        
        datanp = ecg_filter(datanp, 0.1)
        output = list()
        for i in range(0,len(datanp)-1,2):
            a = (datanp[i]+ datanp[i+1])/2
            output.append(a)
        datanp = np.array(output)
        epsil = 10E-8
        datanp = datanp-np.amin(datanp)
        datanpn = datanp/(np.amax(datanp)+epsil)
        peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.85)[0]
        
        signals = []
        count = 1   
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1]- i)
            x = peaks[count - 1] 
            y = peaks[count + 1] - (diff2*4)//5
            signal = datanpn[x:y]
            signals.append(signal)
            count += 1

        X = []
        for j in range(len(peaks)-2):
            xs =signals[j].transpose()
            xs.resize((1,len(signals[j]), 1))
            if len(signals[j])< 187:
                xin = np.zeros(shape=(1,187,1))
                xin[:,0:len(signals[j])] = xs
            else:
                xin = xs[:,0:187]
            X.append(xin)

        predictions = predict_data(model, X)
        return Response(predictions.to_json(orient="index"), mimetype='application/json')
    
@app.route('/PPGupload', methods=['GET', 'POST'])

def PPGupload():
    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[400:]).astype(float)
        
        datanpre = resampler(datanp, 100, 125)
        
        datanpr = datanpre.flatten()
        seasons, trend = fit_seasons(datanpr)
        adjusted = adjust_seasons(datanpr, seasons=seasons)
        residual = datanpr - trend
        
        negppg = -residual
        
        peaks_ppg = find_peaks(negppg, distance = 60)[0]
        epsil = 10E-8
        datanp = -negppg
        
        fs = 125
        RR_list = []
        cnt = 0
        while (cnt < (len(peaks_ppg)-1)):
            RR_interval = (peaks_ppg[cnt+1] - peaks_ppg[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal

        signals_ppg = []
        count = 1
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modelpe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.8)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:60]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(bpm)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        y_pred = model.predict(y_ecg).argmax(axis=-1)
        
        Predictions = pd.DataFrame(y_pred)
        Predictions.index.names = ['Beat']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'Abnormal'})
        
        #result = pd.concat([Predictions,Predicted_ecg])
        #resultpd = pd.DataFrame(result, columns=['result'])
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)

        return Response(result.to_json(), mimetype='application/json')
    
@app.route('/predict')
def my_form():
    return render_template('upload2.html' )

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[1000:]).astype(float)
        datanp = ecg_filter(datanp,0.1 )
        output = list()
        for i in range(0,len(datanp)-1,2):
            a = (datanp[i]+ datanp[i+1])/2
            output.append(a)
        datanp = np.array(output)
        epsil = 10E-8
        datanp = datanp-np.amin(datanp)
        datanpn = datanp/(np.amax(datanp)+epsil)
        peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.70)[0]
        RR_list = []
        cnt = 0
        fs = 125
        while (cnt < (len(peaks)-1)):
            RR_interval = (peaks[cnt+1] - peaks[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm = 60000 / np.mean(RR_list) 

        signals = []
        count = 1   
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1]- i)
            x = peaks[count - 1] 
            y = peaks[count + 1] - (diff2*4)//5
            signal = datanpn[x:y]
            signals.append(signal)
            count += 1

        X = []
        for j in range(len(peaks)-2):
            xs =signals[j].transpose()
            xs.resize((1,len(signals[j]), 1))
            if len(signals[j])< 187:
                xin = np.zeros(shape=(1,187,1))
                xin[:,0:len(signals[j])] = xs
            else:
                xin = xs[:,0:187]
            X.append(xin)

        plt.clf()
        x_coordinates = np.arange(0, len(datanp))
        ybeat = [datanp[x] for x in peaks]

        y_coordinates = datanp
        #bar_heights = df['count'].values
        plt.plot(x_coordinates, y_coordinates, label="raw signal")
        #plt.xticks(x_coordinates, x_labels[-30:], rotation='vertical')
        plt.scatter(peaks, ybeat, color='red', label="average: %.1f BPM" %bpm)
        #plt.ylabel('Count of things')
        plt.legend(loc=4, framealpha=0.6)
        plt.title('ecg plot')
        plt.grid()
        plt.tight_layout()
        plt.savefig('data/pice2.png')

        ### Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        result = str(figdata_png)[2:-1]
        #X = preprocess_csv(request.files.get('file'))

        predictions = predict_data(model, X)
        predictions = predictions.T
        return render_template('upload2.html', imagedata='Plot and prediction of '+request.files.get('file').filename ,data = predictions.to_html(), result=result)
    return render_template('upload2.html')

@app.route('/PPGAFupload', methods=['GET', 'POST'])

def PPGAFupload():
    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[400:]).astype(float)
        
        datanpre = resampler(datanp, 100, 125)
        
        datanpr = datanpre.flatten()
        seasons, trend = fit_seasons(datanpr)
        adjusted = adjust_seasons(datanpr, seasons=seasons)
        residual = datanpr - trend
        
        negppg = -residual
        
        peaks_ppg = find_peaks(negppg, distance = 60)[0]
        epsil = 10E-8
        datanp = -negppg
        
        fs = 125
        RR_list = []
        cnt = 0
        while (cnt < (len(peaks_ppg)-1)):
            RR_interval = (peaks_ppg[cnt+1] - peaks_ppg[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal

        signals_ppg = []
        count = 1
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modelpe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.8)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:60]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(bpm)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        ecg300 = resampler(datame, 125, 300)
        X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
        for i in range(len(ecg300)//2000):
            X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

        Xafib = np.expand_dims(X_afib,2)
        y_afib = modelafib.predict(Xafib).argmax(axis=-1)

        
        Predictions = pd.DataFrame(y_afib)
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'AFib'})
        
        #result = pd.concat([Predictions,Predicted_ecg])
        #resultpd = pd.DataFrame(result, columns=['result'])
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)

        return Response(result.to_json(), mimetype='application/json')
    
    
@app.route('/AFpredict')
def AFmy_form():
    return render_template('upload2.html' )

@app.route('/AFpredict', methods=['GET', 'POST'])
def AFpredict():

    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[1000:]).astype(float)
 
        epsil = 10E-8
        datanpp = datanp-np.amin(datanp)
        datanpn = datanpp/(np.amax(datanpp)+epsil)
        peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.70)[0]
        RR_list = []
        cnt = 0
        fs = 250
        while (cnt < (len(peaks)-1)):
            RR_interval = (peaks[cnt+1] - peaks[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm = 60000 / np.mean(RR_list) 
        
        ecg300 = resampler(datanp, 250, 300)
        X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
        for i in range(len(ecg300)//2000):
            X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)
            signale = X_afib[i,:] 
            epsil = 10E-8
            signalee = signale - np.amin(signale)
            signalecg = signalee/(np.amax(signalee) + epsil)
            X_afib[i,:] = signalecg

        Xafib = np.expand_dims(X_afib,2)

        plt.clf()
        x_coordinates = np.arange(0, len(datanp))
        ybeat = [datanp[x] for x in peaks]

        y_coordinates = datanp
        #bar_heights = df['count'].values
        plt.plot(x_coordinates, y_coordinates, label="raw signal")
        #plt.xticks(x_coordinates, x_labels[-30:], rotation='vertical')
        plt.scatter(peaks, ybeat, color='red', label="average: %.1f BPM" %bpm)
        #plt.ylabel('Count of things')
        plt.legend(loc=4, framealpha=0.6)
        plt.title('ecg plot')
        plt.grid()
        plt.tight_layout()
        plt.savefig('data/pice2.png')

        ### Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        result = str(figdata_png)[2:-1]
        #X = preprocess_csv(request.files.get('file'))

        y_afib = modelafib.predict(Xafib).argmax(axis=-1)
        Predictions = pd.DataFrame(y_afib, columns=['Prediction'])
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'AFib'})
        predictions = Predictions.T
        return render_template('upload2.html', imagedata='Plot and prediction of '+request.files.get('file').filename ,data = predictions.to_html(), result=result)
    return render_template('upload2.html')


@app.route('/motion_hr', methods=['GET', 'POST'])
def motion_hr():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppg = np.array(sampledf[0])[400:].astype(float)
        datanpr = ppg.flatten()
        seasons, trend = fit_seasons(datanpr)
        adjusted = adjust_seasons(datanpr, seasons=seasons)
        residual = datanpr - trend
        
        acc_x = sampledf[1]
        acc_x = acc_x.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accx = np.array(acc_x)[:].astype(float)
        accx_r = resampler(accx, 50, 100)[400:]
        
        acc_y = sampledf[2]
        acc_y = acc_y.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accy = np.array(acc_y)[:].astype(float)
        accy_r = resampler(accy, 50, 100)[400:]
        
        acc_z = sampledf[3]
        acc_z = acc_z.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accz = np.array(acc_z)[:].astype(float)
        accz_r = resampler(accz, 50, 100)[400:]
        
        if len(residual)<len(accx_r):
            ppg_s = residual
            accx_s = accx_r[:len(residual)]
            accy_s = accy_r[:len(residual)]
            accz_s = accz_r[:len(residual)]

        else:
            ppg_s = residual[:len(accx_r)]
            accx_s = accx_r
            accy_s = accy_r
            accz_s = accz_r
            
        ppgfft, accfft, pred, conf = AnalyzeWindow(ppg_s, accx_s, accy_s, accz_s, Fs=100, verbose=True)
        
        fft_diff = ppgfft - (0.7 * accfft)
        ifft_diff = np.fft.irfft(fft_diff)
        new_ppg = ifft_diff[:len(ppg_s)]
        
        datanpre = resampler(new_ppg, 100, 125)
        negppg = -datanpre
        
        peaks_ppg = find_peaks(negppg, distance = 50)[0]
        epsil = 10E-8
        datanp = -negppg
        count = 1
        signals_ppg = []
        
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modelpe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.8)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:60]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(pred)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        ecg300 = resampler(datame, 125, 300)
        X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
        for i in range(len(ecg300)//2000):
            X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

        Xafib = np.expand_dims(X_afib,2)
        y_afib = modelafib.predict(Xafib).argmax(axis=-1)

        
        Predictions = pd.DataFrame(y_afib)
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'AFib'})
        
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)
            
        return Response(result.to_json(), mimetype='application/json')
    
  
@app.route('/PPGAFwatch', methods=['GET', 'POST'])

def PPGAFwatch():
    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf[100:]).astype(float)
        
        datanpre = resampler(datanp, 25, 125)
        
        datanpr = datanpre.flatten()
        seasons, trend = fit_seasons(datanpr)
        adjusted = adjust_seasons(datanpr, seasons=seasons)
        residual = datanpr - trend
        
        negppg = -residual
        
        peaks_ppg = find_peaks(negppg, distance = 60)[0]
        epsil = 10E-8
        datanp = -negppg
        
        fs = 125
        RR_list = []
        cnt = 0
        while (cnt < (len(peaks_ppg)-1)):
            RR_interval = (peaks_ppg[cnt+1] - peaks_ppg[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal

        signals_ppg = []
        count = 1
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modelpe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.8)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:60]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(bpm)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        ecg300 = resampler(datame, 125, 300)
        X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
        for i in range(len(ecg300)//2000):
            X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

        Xafib = np.expand_dims(X_afib,2)
        y_afib = modelafib.predict(Xafib).argmax(axis=-1)

        
        Predictions = pd.DataFrame(y_afib)
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'AFib'})
        
        #result = pd.concat([Predictions,Predicted_ecg])
        #resultpd = pd.DataFrame(result, columns=['result'])
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)

        return Response(result.to_json(), mimetype='application/json')
    

@app.route('/Wmotion_hr', methods=['GET', 'POST'])
def Wmotion_hr():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppg = np.array(sampledf[0])[25:].astype(float)
        #datanpr = ppg.flatten()
        #seasons, trend = fit_seasons(datanpr)
        #adjusted = adjust_seasons(datanpr, seasons=seasons)
        #residual = datanpr - trend
        
        acc_x = sampledf[1]
        acc_x = acc_x.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accx = np.array(acc_x)[25:].astype(float)
        #accx_r = resampler(accx, 50, 100)[400:]
        
        acc_y = sampledf[2]
        acc_y = acc_y.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accy = np.array(acc_y)[25:].astype(float)
        #accy_r = resampler(accy, 50, 100)[400:]
        
        acc_z = sampledf[3]
        acc_z = acc_z.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accz = np.array(acc_z)[25:].astype(float)
        #accz_r = resampler(accz, 50, 100)[400:]
        
        if len(ppg)<len(accx):
            ppg_s = ppg
            accx_s = accx[:len(residual)]
            accy_s = accy[:len(residual)]
            accz_s = accz[:len(residual)]

        else:
            ppg_s = ppg[:len(accx)]
            accx_s = accx
            accy_s = accy
            accz_s = accz
            
        ppgfft, accfft, pred, conf = AnalyzeWindow(ppg_s, accx_s, accy_s, accz_s, Fs=25, verbose=True)
        
        fft_diff = ppgfft - (0.7 * accfft)
        ifft_diff = np.fft.irfft(fft_diff)
        new_ppg = ifft_diff[:len(ppg_s)]
        
        datanpre = resampler(new_ppg, 25, 125)
        negppg = -datanpre
        
        peaks_ppg = find_peaks(negppg, distance = 40)[0]
        epsil = 10E-8
        datanp = -negppg
        count = 1
        signals_ppg = []
        
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modelpe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.8)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:60]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(pred)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        ecg300 = resampler(datame, 125, 300)
        X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
        for i in range(len(ecg300)//2000):
            X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

        Xafib = np.expand_dims(X_afib,2)
        y_afib = modelafib.predict(Xafib).argmax(axis=-1)

        
        Predictions = pd.DataFrame(y_afib)
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'AFib'})
        
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)
            
        return Response(result.to_json(), mimetype='application/json')
    

@app.route('/euw_vf_motion_hr', methods=['GET', 'POST'])
def euw_vf_motion_hr():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppg = np.array(sampledf[0])[100:].astype(float)
        datanpr = ppg.flatten()
        seasons, trend = fit_seasons(datanpr)
        adjusted = adjust_seasons(datanpr, seasons=seasons)
        residual = datanpr - trend
        
        acc_x = sampledf[1]
        acc_x = acc_x.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accx = np.array(acc_x)[:].astype(float)
        accx_r = resampler(accx, 50, 100)[100:]
        
        acc_y = sampledf[2]
        acc_y = acc_y.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accy = np.array(acc_y)[:].astype(float)
        accy_r = resampler(accy, 50, 100)[100:]
        
        acc_z = sampledf[3]
        acc_z = acc_z.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        accz = np.array(acc_z)[:].astype(float)
        accz_r = resampler(accz, 50, 100)[100:]
        
        if len(residual)<len(accx_r):
            ppg_s = residual
            accx_s = accx_r[:len(residual)]
            accy_s = accy_r[:len(residual)]
            accz_s = accz_r[:len(residual)]

        else:
            ppg_s = residual[:len(accx_r)]
            accx_s = accx_r
            accy_s = accy_r
            accz_s = accz_r
            
        ppgfft, accfft, pred, conf = AnalyzeWindow(ppg_s, accx_s, accy_s, accz_s, Fs=100, verbose=True)
        
        fft_diff = ppgfft - (0.7 * accfft)
        ifft_diff = np.fft.irfft(fft_diff)
        new_ppg = ifft_diff[:len(ppg_s)]
        
        datanpre = resampler(new_ppg, 100, 125)
        negppg = -datanpre
        
        peaks_ppg = find_peaks(negppg, distance = 30)[0]
        epsil = 10E-8
        datanp = -negppg
        count = 1
        signals_ppg = []
        
        for i in (peaks_ppg[1:-1]):
            diff1 = abs(peaks_ppg[count - 1] - i)
            diff2 = abs(peaks_ppg[count + 1]- i)
            x = peaks_ppg [count - 1] 
            y = peaks_ppg [count + 1] - (diff2*4)//5
            signalp = datanp[x:y]
            epsil = 10E-8
            signalpp = signalp - np.amin(signalp)
            signalppg = signalpp/(np.amax(signalpp) + epsil)
            signals_ppg.append(signalppg)
            count += 1
        
        X = []
        for j in range(len(peaks_ppg)-2):
            xs =signals_ppg[j].transpose()
            xs.resize((1,len(signals_ppg[j])))

            if len(signals_ppg[j])< 187:
                xin = np.zeros(shape=(1,187))
                xin[:,0:len(signals_ppg[j])] = xs
            else:
                xin = xs[:,0:187]

            X.append(xin)
            
        Xtest = np.zeros(shape=(len(X),187))
        for i in range(len(X)):
            Xtest[i,:] = X[i]
        Xtest = np.expand_dims(Xtest,2)
        
        y_ecg = modeleupe.predict(Xtest)
        Output = []
        for i in range(len(y_ecg)):
            data = y_ecg[i].reshape(187,)
            peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.6)[0]
            if len(peak)!=0:
                output = data[0:peak[0]]
                Output.append(output)  
            else:
                output = data[0:70]
                Output.append(output)

        mergedecg = list(itertools.chain(*Output))
        mergedecg.append(pred)
        datame = np.array(mergedecg).astype(float)
        Predicted_ecg = pd.DataFrame(datame)
        
        ecg250 = resampler(datame, 125, 250)
        X_vfib = np.zeros(shape=(len(ecg250)//750,750))
        for i in range(len(ecg250)//750):
            X_vfib[i,:] = ecg250[i*750:(i+1)*750].reshape(750,)

        Xvfib = np.expand_dims(X_vfib,2)
        y_vfib = modelVfib.predict(Xvfib).argmax(axis=-1)

        
        Predictions = pd.DataFrame(y_vfib)
        Predictions.index.names = ['Beats']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'VFib'})
        
        result = Predictions.append(Predicted_ecg, ignore_index=True, sort=False)
            
        return Response(result.to_json(), mimetype='application/json')
    
   
@app.route('/euw_demo_hr', methods=['GET', 'POST'])
def euw_demo_hr():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppg = np.array(sampledf[0])[100:].astype(float)
        mean_ppg = np.mean(ppg, axis = 0, dtype = "float64")
        
        if mean_ppg>100:
            datanpr = ppg.flatten()
            seasons, trend = fit_seasons(datanpr)
            adjusted = adjust_seasons(datanpr, seasons=seasons)
            residual = datanpr - trend

            acc_x = sampledf[1]
            acc_x = acc_x.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accx = np.array(acc_x)[:].astype(float)
            accx_r = resampler(accx, 50, 100)[100:]

            acc_y = sampledf[2]
            acc_y = acc_y.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accy = np.array(acc_y)[:].astype(float)
            accy_r = resampler(accy, 50, 100)[100:]

            acc_z = sampledf[3]
            acc_z = acc_z.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accz = np.array(acc_z)[:].astype(float)
            accz_r = resampler(accz, 50, 100)[100:]

            if len(residual)<len(accx_r):
                ppg_s = residual
                accx_s = accx_r[:len(residual)]
                accy_s = accy_r[:len(residual)]
                accz_s = accz_r[:len(residual)]

            else:
                ppg_s = residual[:len(accx_r)]
                accx_s = accx_r
                accy_s = accy_r
                accz_s = accz_r

            ppgfft, accfft, pred, conf = AnalyzeWindow(ppg_s, accx_s, accy_s, accz_s, Fs=100, verbose=True)

            fft_diff = ppgfft - (0.7 * accfft)
            ifft_diff = np.fft.irfft(fft_diff)
            new_ppg = ifft_diff[:len(ppg_s)]

            datanpre = resampler(new_ppg, 100, 125)
            negppg = -datanpre

            peaks_ppg = find_peaks(negppg, distance = 30)[0]
            epsil = 10E-8
            datanp = -negppg
            count = 1
            signals_ppg = []

            for i in (peaks_ppg[1:-1]):
                diff1 = abs(peaks_ppg[count - 1] - i)
                diff2 = abs(peaks_ppg[count + 1]- i)
                x = peaks_ppg [count - 1] 
                y = peaks_ppg [count + 1] - (diff2*4)//5
                signalp = datanp[x:y]
                epsil = 10E-8
                signalpp = signalp - np.amin(signalp)
                signalppg = signalpp/(np.amax(signalpp) + epsil)
                signals_ppg.append(signalppg)
                count += 1

            X = []
            for j in range(len(peaks_ppg)-2):
                xs =signals_ppg[j].transpose()
                xs.resize((1,len(signals_ppg[j])))

                if len(signals_ppg[j])< 187:
                    xin = np.zeros(shape=(1,187))
                    xin[:,0:len(signals_ppg[j])] = xs
                else:
                    xin = xs[:,0:187]

                X.append(xin)

            Xtest = np.zeros(shape=(len(X),187))
            for i in range(len(X)):
                Xtest[i,:] = X[i]
            Xtest = np.expand_dims(Xtest,2)

            y_ecg = modeleupe.predict(Xtest)
            Output = []
            for i in range(len(y_ecg)):
                data = y_ecg[i].reshape(187,)
                peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.6)[0]
                if len(peak)!=0:
                    output = data[0:peak[0]]
                    Output.append(output)  
                else:
                    output = data[0:70]
                    Output.append(output)

            mergedecg = list(itertools.chain(*Output))
            mergedecg.append(pred)
            datame = np.array(mergedecg).astype(float)
            Predicted_ecg = pd.DataFrame(datame)

            ecg250 = resampler(datame, 125, 250)
            X_vfib = np.zeros(shape=(len(ecg250)//750,750))
            for i in range(len(ecg250)//750):
                X_vfib[i,:] = ecg250[i*750:(i+1)*750].reshape(750,)

            Xvfib = np.expand_dims(X_vfib,2)
            y_vfib = modelVfib.predict(Xvfib).argmax(axis=-1)

            num_zeros_NVF = (y_vfib == 0).sum()
            num_ones_VF = (y_vfib == 1).sum()

            if num_zeros_NVF > num_ones_VF:
                predVF = 'Non-VFib'
            else:
                predVF = 'VFib'


            ecg300 = resampler(datame, 125, 300)
            X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
            for i in range(len(ecg300)//2000):
                X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

            Xafib = np.expand_dims(X_afib,2)
            y_afib = modelafib.predict(Xafib).argmax(axis=-1)

            num_zeros_NAF = (y_afib == 0).sum()
            num_ones_AF = (y_afib == 1).sum()

            if len(y_afib) == num_ones_AF:
                predAF = 'AFib'
            else:
                predAF = 'Non-AFib'


            if predAF == 'AFib' and predVF == 'Non-VFib':
                pred = predAF
            elif predAF == 'Non-AFib' and predVF == 'VFib':
                pred = predVF
            elif predAF == 'AFib' and predVF == 'VFib':
                pred = 'AFib & VFib'
            else:
                datanpn = datame
                peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.65)[0]

                signals = []
                count = 1   
                for i in (peaks[1:-1]):
                    diff1 = abs(peaks[count - 1] - i)
                    diff2 = abs(peaks[count + 1]- i)
                    x = peaks[count - 1] 
                    y = peaks[count + 1] - (diff2*4)//5
                    signal = datanpn[x:y]
                    signals.append(signal)
                    count += 1

                X = []
                for j in range(len(peaks)-2):
                    xs =signals[j].transpose()
                    xs.resize((1,len(signals[j]), 1))
                    if len(signals[j])< 187:
                        xin = np.zeros(shape=(1,187,1))
                        xin[:,0:len(signals[j])] = xs
                    else:
                        xin = xs[:,0:187]
                    X.append(xin)

                Predictions = np.zeros(shape=(len(X),1))
                Probability = np.zeros(shape=(len(X),1))
                for i in range(len(X)):
                    Predictions[i] = model.predict(X[i]).argmax(axis=-1)

                num_zeros_N = (Predictions == 0).sum()
                num_ones_A = (Predictions == 1).sum()

                if len(Predictions) == num_ones_A:
                    pred = 'Unclassified'
                else: 
                    pred = 'Normal'
                    
        else:
            pred = 'Data not pertinent for diagnosis'
            datame = np.array(ppg).astype(float)
            Predicted_ecg = pd.DataFrame(datame)      
            
            

        
        data = []
        data.append(pred)
        Prediction = pd.DataFrame(data)
        result = Prediction.append(Predicted_ecg, ignore_index=True, sort=False)
            
        return Response(result.to_json(), mimetype='application/json')
    

@app.route('/euw_demo_hr2', methods=['GET', 'POST'])
def euw_demo_hr2():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppg = np.array(sampledf[0])[200:].astype(float)
        mean_ppg = np.mean(ppg, axis = 0, dtype = "float64")
        
        if mean_ppg>100:
            datanpr = ppg.flatten()
            seasons, trend = fit_seasons(datanpr)
            adjusted = adjust_seasons(datanpr, seasons=seasons)
            residual = datanpr - trend

            acc_x = sampledf[1]
            acc_x = acc_x.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accx = np.array(acc_x)[:].astype(float)
            accx_r = resampler(accx, 50, 100)[200:]

            acc_y = sampledf[2]
            acc_y = acc_y.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accy = np.array(acc_y)[:].astype(float)
            accy_r = resampler(accy, 50, 100)[200:]

            acc_z = sampledf[3]
            acc_z = acc_z.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
            accz = np.array(acc_z)[:].astype(float)
            accz_r = resampler(accz, 50, 100)[200:]

            if len(residual)<len(accx_r):
                ppg_s = residual
                accx_s = accx_r[:len(residual)]
                accy_s = accy_r[:len(residual)]
                accz_s = accz_r[:len(residual)]

            else:
                ppg_s = residual[:len(accx_r)]
                accx_s = accx_r
                accy_s = accy_r
                accz_s = accz_r

            ppgfft, accfft, pred, conf, new_ppg = AnalyzeWindow2(ppg_s, accx_s, accy_s, accz_s, Fs=100, verbose=True)

            datanpre = resampler(new_ppg, 100, 125)
            negppg = -datanpre

            peaks_ppg = find_peaks(negppg, distance = 30)[0]
            epsil = 10E-8
            datanp = -negppg
            count = 1
            signals_ppg = []

            for i in (peaks_ppg[1:-1]):
                diff1 = abs(peaks_ppg[count - 1] - i)
                diff2 = abs(peaks_ppg[count + 1]- i)
                x = peaks_ppg [count - 1] 
                y = peaks_ppg [count + 1] - (diff2*4)//5
                signalp = datanp[x:y]
                epsil = 10E-8
                signalpp = signalp - np.amin(signalp)
                signalppg = signalpp/(np.amax(signalpp) + epsil)
                signals_ppg.append(signalppg)
                count += 1

            X = []
            for j in range(len(peaks_ppg)-2):
                xs =signals_ppg[j].transpose()
                xs.resize((1,len(signals_ppg[j])))

                if len(signals_ppg[j])< 187:
                    xin = np.zeros(shape=(1,187))
                    xin[:,0:len(signals_ppg[j])] = xs
                else:
                    xin = xs[:,0:187]

                X.append(xin)

            Xtest = np.zeros(shape=(len(X),187))
            for i in range(len(X)):
                Xtest[i,:] = X[i]
            Xtest = np.expand_dims(Xtest,2)

            y_ecg = modeleupe.predict(Xtest)
            Output = []
            for i in range(len(y_ecg)):
                data = y_ecg[i].reshape(187,)
                peak = biosppy.signals.ecg.engzee_segmenter(signal=data, sampling_rate=125.0, threshold=0.6)[0]
                if len(peak)!=0:
                    output = data[0:peak[0]]
                    Output.append(output)  
                else:
                    output = data[0:70]
                    Output.append(output)

            mergedecg = list(itertools.chain(*Output))
            mergedecg.append(pred)
            datame = np.array(mergedecg).astype(float)
            Predicted_ecg = pd.DataFrame(datame)

            ecg250 = resampler(datame, 125, 250)
            X_vfib = np.zeros(shape=(len(ecg250)//750,750))
            for i in range(len(ecg250)//750):
                X_vfib[i,:] = ecg250[i*750:(i+1)*750].reshape(750,)

            Xvfib = np.expand_dims(X_vfib,2)
            y_vfib = modelVfib.predict(Xvfib).argmax(axis=-1)

            num_zeros_NVF = (y_vfib == 0).sum()
            num_ones_VF = (y_vfib == 1).sum()

            if num_zeros_NVF > num_ones_VF:
                predVF = 'Non-VFib'
            else:
                predVF = 'VFib'


            ecg300 = resampler(datame, 125, 300)
            X_afib = np.zeros(shape=(len(ecg300)//2000,2000))
            for i in range(len(ecg300)//2000):
                X_afib[i,:] = ecg300[i*2000:(i+1)*2000].reshape(2000,)

            Xafib = np.expand_dims(X_afib,2)
            y_afib = modelafib.predict(Xafib).argmax(axis=-1)

            num_zeros_NAF = (y_afib == 0).sum()
            num_ones_AF = (y_afib == 1).sum()

            if len(y_afib) == num_ones_AF:
                predAF = 'AFib'
            else:
                predAF = 'Non-AFib'


            if predAF == 'AFib' and predVF == 'Non-VFib':
                pred = predAF
            elif predAF == 'Non-AFib' and predVF == 'VFib':
                pred = predVF
            elif predAF == 'AFib' and predVF == 'VFib':
                pred = 'AFib & VFib'
            else:
                datanpn = datame
                peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.65)[0]

                signals = []
                count = 1   
                for i in (peaks[1:-1]):
                    diff1 = abs(peaks[count - 1] - i)
                    diff2 = abs(peaks[count + 1]- i)
                    x = peaks[count - 1] 
                    y = peaks[count + 1] - (diff2*4)//5
                    signal = datanpn[x:y]
                    signals.append(signal)
                    count += 1

                X = []
                for j in range(len(peaks)-2):
                    xs =signals[j].transpose()
                    xs.resize((1,len(signals[j]), 1))
                    if len(signals[j])< 187:
                        xin = np.zeros(shape=(1,187,1))
                        xin[:,0:len(signals[j])] = xs
                    else:
                        xin = xs[:,0:187]
                    X.append(xin)

                Predictions = np.zeros(shape=(len(X),1))
                Probability = np.zeros(shape=(len(X),1))
                for i in range(len(X)):
                    Predictions[i] = model.predict(X[i]).argmax(axis=-1)

                num_zeros_N = (Predictions == 0).sum()
                num_ones_A = (Predictions == 1).sum()

                if len(Predictions) == num_ones_A:
                    pred = 'Unclassified'
                else: 
                    pred = 'Normal'
                    
        else:
            pred = 'Data not pertinent for diagnosis'
            datame = np.array(ppg).astype(float)
            Predicted_ecg = pd.DataFrame(datame)      
            
            

        
        data = []
        data.append(pred)
        Prediction = pd.DataFrame(data)
        result = Prediction.append(Predicted_ecg, ignore_index=True, sort=False)
            
        return Response(result.to_json(), mimetype='application/json')
    
    
@app.route('/euw_vital', methods=['GET', 'POST'])
def euw_vital():
    if request.method == 'POST':
        
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        ppgg = np.array(sampledf[0])[100:].astype(float)
        datanpre1= BandpassFilter(ppgg, 100)
        ppgg1200 = -datanpre1
        peaks_ppg = find_peaks(ppgg1200, distance = 40)[0]
        
        fs = 100
        RR_list = []
        cnt = 0
        while (cnt < (len(peaks_ppg)-1)):
            RR_interval = (peaks_ppg[cnt+1] - peaks_ppg[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1
        bpm_rr = 60000 / np.mean(RR_list)
        
        fft_len = len(datanpre1)*4
        freqs = np.fft.rfftfreq(fft_len, 1 / fs)
        norm_x = (datanpre1 - np.mean(datanpre1))/(max(datanpre1)-min(datanpre1))
        fft_x = np.fft.rfft(norm_x, fft_len)
        mag_freq_x = np.abs(fft_x)

        peaks_ppg = find_peaks(mag_freq_x, height=10, distance=1)[0]
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_x[i], reverse=True)

        use_peak = sorted_freq_peaks_ppg[0]
        chosen_freq = freqs[use_peak]
        bpm_fs = chosen_freq * 60
        
        arr = np.array(RR_list)
        SD = []
        for i in range(len(arr)-1):
            diff = arr[i] - arr[i+1]
            diff2 = diff **2
            SD.append(diff2)

        MSD = np.mean(SD)
        hrv_RMSSD = np.sqrt(MSD)
        
        hrv_SDRR = np.std(arr, axis=0)
        
        ppgr = np.array(sampledf[1])[100:].astype(float)
        ppgir = np.array(sampledf[2])[100:].astype(float)
        
        rmsr = sqrt(mean(square(ppgr)))
        meanr = mean(ppgr)
        
        rmsir = sqrt(mean(square(ppgir)))
        meanir = mean(ppgir)
        
        R = rmsr/meanr
        IR = rmsir/meanir
        
        SpO2 = (R/IR)*98.56
        
        ybeat = [ppgg[x] for x in peaks_ppg]
        sampling_rate = 100
        independent_data = ybeat
        dependent_data = np.linspace(0, len(ybeat), len(ybeat))
        interpolate = UnivariateSpline(dependent_data, independent_data, k=3)
        new_data = np.linspace(0, len(ybeat), len(ppgg))
        interpolated_data = interpolate(new_data)
        breathing_signal = interpolated_data
        
        respc = BandpassFilter2(breathing_signal, sampling_rate)
        fftb_len=len(respc)*4
        freqsb = np.fft.rfftfreq(fftb_len, 1 / sampling_rate)
        
        norm_xb = (respc - np.mean(respc))/(max(respc)-min(respc))
        fft_xb = np.fft.rfft(norm_xb, fftb_len)

        mag_freq_xb = np.abs(fft_xb)
        peaks_resp = find_peaks(mag_freq_xb, height=10, distance=1)[0]
        sorted_freq_peaks_resp = sorted(peaks_resp, key=lambda i:mag_freq_xb[i], reverse=True)
        use_peakb = sorted_freq_peaks_resp[0]
        chosen_freqb = freqsb[use_peakb]
        breath_per_min = chosen_freqb * 60
        
        respsig = respc
        peaks_res = find_peaks(respsig)[0]
        fs = 100
        RRs_list = []
        cnt = 0
        while (cnt < (len(peaks_res)-1)):
            RRs_interval = (peaks_res[cnt+1] - peaks_res[cnt]) #Calculate distance between beats in # of samples
            mss_dist = ((RRs_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RRs_list.append(mss_dist) #Append to list
            cnt += 1
        brpm = 60000 / np.mean(RRs_list)
        
        datanprppg = resampler(datanpre1, 100, 125)
        X_bp = np.zeros(shape=(len(datanprppg)//250,250))
        for i in range(len(datanprppg)//250):
            X_bp[i,:] = datanprppg[i*250:(i+1)*250].reshape(250,)
            
        epsil = 10E-10
        min_train = X_bp - np.amin(X_bp, axis=1).reshape(X_bp.shape[0],1)
        norm_train = min_train/(np.amax(min_train, axis=1).reshape(X_bp.shape[0],1) + epsil)

        Xbp = np.expand_dims(norm_train,2)
        y_bp = modelbp.predict(Xbp)
        
        sbpp = []
        dbpp = []
        for i in range(y_bp.shape[0]):
            a = y_bp[i].reshape(250,)
            max_p = max(a)
            min_p = min(a)

            sbpp.append(max_p)
            dbpp.append(min_p)
            
        sbp = np.mean(sbpp)
        dbp = np.mean(dbpp)
        
       
        
        data = []
        data.append(bpm_rr)
        data.append(bpm_fs)
        data.append(hrv_RMSSD)
        data.append(hrv_SDRR)
        data.append(SpO2)
        data.append(breath_per_min)
        data.append(brpm)
        data.append(sbp)
        data.append(dbp)
        
        
        vs_results = pd.DataFrame(data)
        
        return Response(vs_results.to_json(), mimetype='application/json')
        

if __name__ == '__main__':
    app.run(debug=True)