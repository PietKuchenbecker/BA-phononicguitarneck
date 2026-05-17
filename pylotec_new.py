import numpy as np
# from scipy.signal import find_peaks, peak_prominences
# import pyuff
import os                       # OS stuff
# from tqdm import tqdm               #progressbar

# import plotly.express as px
import plotly.graph_objects as go

# import matplotlib.image as mgimg
# from PIL import Image
# from scipy.fft import fft

# import cmath

# import matplotlib.image as mpimg
# from PIL import Image

from scipy.signal import find_peaks
# import pandas as pd

import Functions.PlotFunctions as plFKT

meas = os.getcwd()+"/Data/"


# Neu von Piet:

def plot_modes2(IRFs, Peaks2, **kwargs): 
    records = show_records(meas)    # Funktioniert erst im vollstängen Skript
    decision = found_uffs(records)  # Funktioniert erst im vollstängen Skript

    mess = records[int(decision)][:-4]
    fmin = kwargs.get('fmin', 200) 
    fmax = kwargs.get('fmax', 1200)
    
    Peaks2 = Peaks2[Peaks2 >= fmin]
    Peaks2 = Peaks2[Peaks2 <= fmax]

    # Daten laden
    X = np.load(meas + 'NumPyArrays/' + mess + '_Xnodes.npy')
    Y = np.load(meas + 'NumPyArrays/' + mess + '_Ynodes.npy')

    Xn = X
    Yn = Y

    x1 = 1000
    y1 = 1000

    for i in range(len(Peaks2)): 
        f = Peaks2[i]

        # komplexe Werte
        Amp_complex = IRFs[:, f]

        # 🔹 NEU: Amplitude und Phase berechnen
        amplitude = np.abs(Amp_complex)
        phase = np.angle(Amp_complex)

        # 🔹 Plot Amplitude
        plotData = plFKT.PlotModeRe(Xn, Yn, amplitude, (x1/50, y1/50))
        plotData.savefig(meas + 'img/' + mess + '_Amp_' + str(f).zfill(4) + 'Hz.png')

        # 🔹 Plot Phase
        plotData = plFKT.PlotModeRe(Xn, Yn, phase, (x1/50, y1/50))
        plotData.savefig(meas + 'img/' + mess + '_Phase_' + str(f).zfill(4) + 'Hz.png')

    return 'Modes saved!'

def compute_mean_spectrum(mess, thresh=-9, minProm=0.1, fmax=5000):
    
    # Daten laden
    X = np.load(meas + 'NumPyArrays/' + mess + '_Xnodes.npy')
    IRFs = np.load(meas + 'NumPyArrays/' + mess + '_IRFs.npy')

    nPoints = len(X)

    MeanSpec = np.zeros(len(IRFs[1]), dtype=np.complex128)

    for i in range(nPoints):
        MeanSpec += np.abs(IRFs[i])

    MeanSpec /= nPoints

    spec = np.abs(MeanSpec)[0:fmax]
    spec = np.log(np.divide(spec, max(spec)))

    # Peaks finden
    peaks, props = find_peaks(spec, distance=1, prominence=minProm, height=thresh)

    return spec, peaks, props, IRFs

def compare_spectra(mess1, mess2, **kwargs):

    thresh = kwargs.get('thresh', -9)
    minProm = kwargs.get('prom', 0.1)
    fmax = kwargs.get('fmax', 5000)
    save = kwargs.get('saveToFile', False)

    # 🔹 Spektren berechnen
    spec1, peaks1, props1, IRFs1 = compute_mean_spectrum(mess1, thresh, minProm, fmax)
    spec2, peaks2, props2, IRFs2 = compute_mean_spectrum(mess2, thresh, minProm, fmax)

    x = np.arange(fmax)

    fig = go.Figure()

    # 🔹 Spektrum 1
    fig.add_trace(go.Scatter(
        x=x, y=spec1,
        name=mess1 + " Spectrum",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=peaks1,
        y=props1['peak_heights'],
        mode='markers',
        name=mess1 + " Peaks"
    ))

    # 🔹 Spektrum 2
    fig.add_trace(go.Scatter(
        x=x, y=spec2,
        name=mess2 + " Spectrum",
        line=dict(width=2)  
    ))

    fig.add_trace(go.Scatter(
        x=peaks2,
        y=props2['peak_heights'],
        mode='markers',
        name=mess2 + " Peaks"
    ))

    fig.update_layout(
        title=f"Spektrenvergleich: {mess1} vs {mess2}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Mobility (log scale)",
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="RebeccaPurple"
        )
    )

    fig.show()

    if save:
        fig.write_html(f"compare_{mess1}_vs_{mess2}.html")

    return (IRFs1, peaks1), (IRFs2, peaks2)

def compare_spectra_diff(mess1, mess2, **kwargs):

    thresh = kwargs.get('thresh', -9)
    minProm = kwargs.get('prom', 0.1)
    fmax = kwargs.get('fmax', 5000)
    save = kwargs.get('saveToFile', False)
    dBscale = kwargs.get('dBscale', False)

    # 🔹 Spektren laden
    spec1, peaks1, props1, _ = compute_mean_spectrum(mess1, thresh, minProm, fmax)
    spec2, peaks2, props2, _ = compute_mean_spectrum(mess2, thresh, minProm, fmax)

    # 🔹 zurück zu linear (weil compute schon log gemacht hat!)
    lin1 = np.exp(spec1)
    lin2 = np.exp(spec2)

    # 🔹 Normierung (sehr wichtig!)
    # lin1 /= np.max(lin1)
    # lin2 /= np.max(lin2)

    # 🔹 logarithmische Differenz
    if dBscale:
        diff = 20 * np.log10((spec1 + 1e-12) / (spec2 + 1e-12))
        scale = "dB"
    else:
        diff = np.log((spec1 + 1e-12) / (spec2 + 1e-12))
        scale = "log(A1/A2)"

    x = np.arange(fmax)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=diff,
        name="Log-Differenz (Mess1 / Mess2)",
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"Differenzspektrum: {mess1} vs {mess2}",
        xaxis_title="Frequency (Hz)",
        yaxis_title=f'Difference ({scale})',
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="RebeccaPurple"
        )
    )

    fig.show()

    if save:
        fig.write_html(f"diff_{mess1}_vs_{mess2}.html")

    return diff