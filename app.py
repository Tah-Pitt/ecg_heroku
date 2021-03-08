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

app = Flask(__name__)
app.config['SECRET_KEY'] = b'b\xa1\xe8\x0bs\x06\xae\xfd\xa8\xcc~!9' 
                 

def get_model():
    global model
    global modelpe
    model = load_model('modelarxivptbmit-020-0.991439-0.984750.h5')
    modelpe = load_model('proposemodel1.h5')
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

    
print(" * Loading model...!")
get_model()

@app.route('/upload', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf).astype(float)
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
        datanp = np.array(sampledf).astype(float)
        negppg = -datanp.flatten()
        peaks_ppg = find_peaks(negppg)[0]
        epsil = 10E-8

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
        y_pred = model.predict(y_ecg).argmax(axis=-1)
        
        Predictions = pd.DataFrame(y_pred, columns=['Prediction'])
        Predictions.index.names = ['Beat']
        Predictions = Predictions.replace({0.0:'Normal', 1.0:'Abnormal'})

        return Response(Predictions.to_json(orient="index"), mimetype='application/json')
    
@app.route('/predict')
def my_form():
    return render_template('upload2.html' )

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        sampledf= pd.read_csv(request.files.get('file'), header=None)
        sampledf= sampledf.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
        datanp = np.array(sampledf).astype(float)
        output = list()
        for i in range(0,len(datanp)-1,2):
            a = (datanp[i]+ datanp[i+1])/2
            output.append(a)
        datanp = np.array(output)
        epsil = 10E-8
        datanp = datanp-np.amin(datanp)
        datanpn = datanp/(np.amax(datanp)+epsil)
        peaks = biosppy.signals.ecg.engzee_segmenter(signal=datanpn, sampling_rate=125.0, threshold=0.85)[0]
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

if __name__ == '__main__':
    app.run(debug=True)