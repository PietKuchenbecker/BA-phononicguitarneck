# Alle Functions und Data Classes die das Array zur Auswertung benötigt

# import sys
# print(sys.executable)
import os
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import json
from pathlib import Path
import librosa
import plotly.graph_objects as go
from scipy.signal import filtfilt, cheby2
import plotly.express as px
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d
from itertools import product
import re
from scipy.io import wavfile
from scipy.signal import find_peaks
from plotly.subplots import make_subplots


# Vordefiniertes
TYPE_FOLDER = {
    "wav": "../data/array/raw_arrays",
    "fwav": "../data/array/cut_arrays",
    "rms": "../data/array/rms_arrays",
    "fft": "../data/array/fft_arrays",
    "dfft": "../data/array/dfft_arrays",
    # "mw": "../data/array/mw_arrays"
}

FINALIZE_PARAMETERS = {
    "audio_length": 3,      # in seconds
    "fade_time": 0.01,      # in seconds
    "filter_order": 4,      
    "filter_stopband": 120, # in dB
    "filter_cutoff": 10            # in Hz
}

# Konstruiert Klassen für das gesamte Datenset mit deren Attribute im Anschluss gefiltert werden kann.

# ---------------------------------------------------------------------
# Datenstruktur für ein Sample
# ---------------------------------------------------------------------
@dataclass
class Sample:
    filepath: str
    messung: int
    saite: int
    durchlauf: int
    arrayy: int
    arrayx: int


# ---------------------------------------------------------------------
# Parser für Dateinamen nach dem Schema:
# messung_saite_durchlauf_arrayy_arrayx.wav
# ---------------------------------------------------------------------
def parse_filename(filename: str) -> Sample | None:
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".wav":
        return None

    parts = name.split("_")
    if len(parts) != 5:
        return None

    try:
        messung = int(parts[0])
        saite = int(parts[1])
        durchlauf = int(parts[2])
        arrayy = int(parts[3])
        arrayx = int(parts[4])  # ist zweistellig, aber int() ignoriert führende 0
    except ValueError:
        return None

    return messung, saite, durchlauf, arrayy, arrayx


# ---------------------------------------------------------------------
# Alle Dateien laden und in Sample-Objekte umwandeln
# ---------------------------------------------------------------------
def load_samples(folder="../data/array/Cut Folder (Complete)"):
    samples = []
    for f in os.listdir(folder):
        parsed = parse_filename(f)
        if parsed is None:
            continue

        sample = Sample(
            filepath=os.path.join(folder, f),
            messung=parsed[0],
            saite=parsed[1],
            durchlauf=parsed[2],
            arrayy=parsed[3],
            arrayx=parsed[4]
        )
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------
# kwargs kann für jedes Attribut entweder:
# - einen festen Wert liefern: messung=2
# - ein Tuple mit (min, max) für Range: arrayx=(3,7)
# - eine Liste erlaubter Werte: arrayy=[1,3,5]
# ---------------------------------------------------------------------
def filter_samples_range(samples, **kwargs):
    result = []
    for s in samples:
        ok = True
        for key, value in kwargs.items():
            attr = getattr(s, key)
            
            # Range als Tuple
            if isinstance(value, tuple) and len(value) == 2:
                if not (value[0] <= attr <= value[1]):
                    ok = False
                    break
            # Liste von erlaubten Werten
            elif isinstance(value, list):
                if attr not in value:
                    ok = False
                    break
            # Fester Wert
            else:
                if attr != value:
                    ok = False
                    break
        if ok:
            result.append(s)
    
    # Optionale Displays
    print(f"Gefundene passende Samples: {len(result)}")
    # for s in result:
    #     print(f"{s.filepath}  |  m={s.messung}  s={s.saite}  DL={s.durchlauf}  y={s.arrayy}  x={s.arrayx}")
    
    return result


# Lade wav-Dateien in Array-Form

# ------------------------------------------------------------
# WAV-Daten in ein (T, 3, 11)-Array laden
# ------------------------------------------------------------

def load_wav_matrix(filtered):
    """
    Lädt alle WAV-Dateien aus 'filtered' und ordnet sie nach (arrayy, arrayx).
    Gibt ein Array der Form (T, 3, 11) zurück.
    """
    if len(filtered) != 33:
        raise ValueError(f"filtered benötigt 33 Samples, beinhaltet allerdings {len(filtered)} Samples.")

    # Sortieren, damit (0,0) bis (2,10) in korrekter Reihenfolge liegen
    filtered_sorted = sorted(filtered, key=lambda s: (s.arrayy, s.arrayx))

    # WAV-Dateien laden
    wavs = []
    lengths = []

    for sample in filtered_sorted:
        data, sr = sf.read(sample.filepath)     # data.shape = (T,) oder (T,Channels)
        if data.ndim > 1:
            data = data[:, 0]                   # Falls Stereo → auf Mono reduzieren
        wavs.append(data)
        lengths.append(len(data))

    # Alle WAVs auf gleiche Länge kürzen
    T = min(lengths)
    wavs = [w[:T] for w in wavs]

    # In Array bringen: (33, T)
    wav_matrix = np.vstack(wavs)  # shape: (33, T)

    # Reshape nach (T, 3, 11)
    data = wav_matrix.reshape(3, 11, T).transpose(2,0,1)

    # Optional Display
    print("Loaded new wav-Array")

    return data, sr, T

def create_metadata(filtered, sr, T):
    """
    Creates Dictonary mith metadata to safe alongside array_data
    """
    meta = {
        "sr": sr,
        "T": T,
        "messung": filtered[1].messung,
        "saite": filtered[1].saite,
        "durchlauf": filtered[1].durchlauf,
        "type": "wav"
    }

    # Optional Display
    print(f"Metadaten: {meta}")

    return meta


# Arraydaten als Datensatz mit Metadaten speichern

def save_npz(data, meta):
    # Ordner anhand type bestimmen
    folder_path = TYPE_FOLDER.get(meta['type'])
    if folder_path is None:
        raise ValueError(f"Unbekannter type: {meta['type']}")

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)  # Ordner anlegen, falls nötig

    filename = f"dataset_{meta['messung']}_{meta['saite']}_{meta['durchlauf']}_{meta['type']}.npz"
    file_path = folder_path / filename

    np.savez(file_path, data=data, meta=np.array(json.dumps(meta)))

    print(f"Datei {filename} wurde in {folder_path} abgelegt.")


# Array .npz-Datei laden
def load_npz(filename):
    _, messung, saite, durchlauf, file_type = filename.split("_")
    if file_type.count(".npz") > 0:
        file_type, _ = file_type.split(".")
    else:
        filename = f"{filename}.npz"

    folder_path = TYPE_FOLDER.get(file_type)
    if folder_path is None:
        raise ValueError(f"Unbekannter type: {file_type}")
    print(f"Searching for {filename} in {folder_path}")
    
    folder_path = Path(folder_path)
    file_path = folder_path / filename

    npz = np.load(file_path, allow_pickle=True)
    print(f"Loaded {filename}")
    data = npz["data"]
    meta = json.loads(npz["meta"].item())

    return data, meta


# Onset-Detection
def onset_detection(samples):

    onsets = {}
    warnings = {}
    for s in samples:
        # load sample
        y, sr = librosa.load(s.filepath, sr=None)

        # onset detection
        onset = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
        # print(onset, f"{s.messung}_{s.saite}_{s.durchlauf}")
        
        for counter, o in enumerate(onset, start = 1):
            if int(o) <=10000:
                onsets[f"onset_{s.messung}_{s.saite}_{s.durchlauf}_{counter}"] = int(o)
                if counter >= 2:
                    warnings[f"onset_{s.messung}_{s.saite}_{s.durchlauf}"]=counter
                    print(f"Warning! {s.messung}_{s.saite}_{s.durchlauf}")
                else:
                        pass
            else:
                pass
    print(f"{len(onsets)} Onsets detected")
    return onsets, warnings

# Save Onsets
def save_onsets(onsets, folder_path="../data/array/misc", filename="onsets.json"):

    if list(onsets)[0].count("_")==4:
        for i in list(onsets):
            new, _ = i.rsplit("_",1)
            print(new)
            onsets[new] = onsets.pop(i)

    if len(onsets) != 120:
        raise ValueError(f"Has to be 120 onsets. Your list includes {len(onsets)}.")
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)  # Ordner anlegen, falls nötig

    file_path = folder_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(onsets, f, indent=2)
    print(f"File {filename} saved at {folder_path}")

    return file_path

# Load Onsets
def load_onsets(folder_path="../data/array/misc", filename="onsets.json"):

    folder_path = Path(folder_path)
    file_path = folder_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Die Datei {file_path} existiert nicht!")
    else:
        print(f"Found {filename} in {folder_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        onsets = json.load(f)

    return onsets

def plot_multiple_onsets(samples, onsets, multiple_onsets):

    if len(multiple_onsets)==0:
        raise ValueError("This is a good thing. No multiple onsets detected!")
    for m in multiple_onsets:
        onset_times = []
        count = multiple_onsets[m]
        _, messung, saite, durchlauf = m.split("_")
        f=filter_samples_range(samples, messung=int(messung),saite=int(saite),durchlauf=int(durchlauf),arrayx=6,arrayy=2)
        y, sr = librosa.load(f[0].filepath, sr=None)
        y = y[:sr]
        for o in range(1,count+1):
            key = f"{m}_{o}"
            print(f"Onset {o}: {key}")
            onset_times.append(onsets[key] / sr)
        # print(onset_list)

        t = np.linspace(0,len(y)/sr,len(y))

        fig = go.Figure()

        # Audio
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name="Audio"
            )
        )

        # Onsets auf y = 0
        fig.add_trace(
            go.Scatter(
                x=onset_times,
                y=np.zeros(len(onset_times)),
                mode="markers",
                name="Onsets",
                marker=dict(color="red", size=8)
            )
        )

        fig.update_layout(
            xaxis_title="Zeit (s)",
            yaxis_title="Amplitude"
        )

        fig.show()


# Choose an Onset out of multiple

def choose_onset(onsets, multiple_onsets, selected):
    selected_file, selected_onset = selected.rsplit("_", 1)
    if selected_file not in multiple_onsets:
        raise ValueError(f"{selected_file} does not  have multiple onsets.")
    count = multiple_onsets[selected_file]
    delete = []
    for o in onsets:
        file, onset = o.rsplit("_",1)
        if file == selected_file:
            if onset != selected_onset:
                delete.append(o)
                print(f"Deleted {o}")
            else:
                print(f"Kept {o}")
    for i in delete:
        del onsets[i]
    del multiple_onsets[selected_file]
    print(f" onset count reduced to {len(onsets)}")



# Finalize the Array form
# Cut, apply High Pass filter, apply fade in / out to array

def filter_audio(signal, sr, order=FINALIZE_PARAMETERS["filter_order"], stopband_db=FINALIZE_PARAMETERS["filter_stopband"], cutoff=FINALIZE_PARAMETERS["filter_cutoff"]):
    nyq = sr/2
    highpass, a = cheby2(order, stopband_db, cutoff/nyq, btype="highpass", analog=False)

    # filter signal (linear phase)
    signal = filtfilt(highpass, a, signal)

    return signal

def cut_audio(signal, T_start, T_end):
    signal = signal[T_start:T_end]

    return signal

def apply_windowing(signal, T_fade):

    fade_in = np.hanning(2*T_fade)[:T_fade]
    fade_out = np.hanning(2*T_fade)[T_fade:]

    # apply window functions
    signal[:T_fade] *= fade_in
    signal[-T_fade:] *= fade_out

    return signal

def finalize_array(data, meta, onsets, length=FINALIZE_PARAMETERS["audio_length"], fade_time=FINALIZE_PARAMETERS["fade_time"]):

    # Generelle Infos herausziehen
    sr = meta["sr"]

    T_new = length * sr
    T_fade = int(fade_time * sr)
    T, rows, cols = data.shape

    # Output-Array erzeugen
    data_final = np.zeros((T_new, rows, cols), dtype=float)

    messung = meta['messung']
    if messung < 5:
        messung += 4

    # Offset finden
    T_start = onsets[f"onset_{messung}_{meta['saite']}_{meta['durchlauf']}"] - 2 * T_fade
    T_end = T_start + T_new

    # Für jedes Mikrofon individuelle Berechnung
    for y in range(rows):
        for x in range(cols):
            signal = data[:, y, x]

            # filter signal (linear phase)
            signal = filter_audio(signal, sr)

            # cut signal with onset data
            signal = cut_audio(signal, T_start, T_end)

            # apply window functions
            signal = apply_windowing(signal, T_fade)

            # plot to be very shure?

            data_final[:,y,x] = signal

    meta["type"] = "fwav"

    return data_final, meta

def finalize_pickup(signal, sr, m, s, d, onsets,
                    length=FINALIZE_PARAMETERS["audio_length"],
                    fade_time=FINALIZE_PARAMETERS["fade_time"]):

    import numpy as np

    # Zielgrößen
    T_new = int(length * sr)
    T_fade = int(fade_time * sr)

    # Korrektur wie im Original
    messung = m
    if messung < 5:
        messung += 4

    # Onset holen
    onset_key = f"onset_{messung}_{s}_{d}"
    if onset_key not in onsets:
        raise ValueError(f"Onset {onset_key} nicht gefunden!")

    T_start = onsets[onset_key] - 2 * T_fade
    T_end = T_start + T_new

    # === Verarbeitung ===

    # 1. Filter
    signal = filter_audio(signal, sr)

    # 2. Schneiden
    signal = cut_audio(signal, T_start, T_end)

    # 3. Windowing
    signal = apply_windowing(signal, T_fade)

    return signal

# rms-Array erstellen

def compute_rms_array(data, meta, frame_length=1024, hop_length=None):

    if hop_length is None:
        hop_length = frame_length // 2

    sr = meta["sr"]
    T, rows, cols = data.shape

    # Anzahl Frames berechnen
    T_rms = 1 + (T - frame_length) // hop_length
    if T_rms <= 0:
        raise ValueError("Frame length longer than signal")

    # Output-Array erzeugen
    data_rms = np.zeros((T_rms, rows, cols), dtype=float)

    # Für jedes Mikrofon getrennt RMS berechnen
    for y in range(rows):
        for x in range(cols):
            signal = data[:, y, x]

            rms_values = []
            for start in range(0, T - frame_length + 1, hop_length):
                frame = signal[start:start + frame_length]
                rms = np.sqrt(np.mean(frame**2))
                rms_values.append(rms)

            data_rms[:, y, x] = rms_values

    # "Samplerate" der neuen Zeitachse: Frames pro Sekunde
    sr_rms = sr / hop_length
    meta["type"]="rms"
    meta["sr"]=sr_rms

    return data_rms, meta


# Array-Animation mit t-Slider

def create_heatmap_animation(data,meta,t_start=0.0,t_end=None,step=None,playspeed=0):
    """
    Erzeugt eine animierte Heatmap über die Zeit.

    Parameters
    ----------
    data : ndarray (T, rows, cols)
        Zeitabhängige Array-Daten
    samplerate : float
        Abtastrate in Hz
    t_start : float
        Startzeit in Sekunden
    t_end : float or None
        Endzeit in Sekunden (None = bis Ende)
    step : float or None
        Schrittweite in Sekunden (None = 1 Sample)
    playspeed : int
        Frame-Dauer in ms
    """

    T, rows, cols = data.shape
    sr = meta["sr"]

    # -------------------------
    # Sekunden → Samples
    # -------------------------
    t_start_samp = int(round(t_start * sr))
    t_start_samp = max(0, t_start_samp)

    if t_end is None:
        t_end_samp = T
    else:
        t_end_samp = int(round(t_end * sr))
        t_end_samp = min(T, t_end_samp)

    if step is None:
        step_samp = 1
    else:
        step_samp = max(1, int(round(step * sr)))

    # -------------------------
    # Farbskala bestimmen
    # -------------------------
    data_min = data.min()
    data_max = data.max()

    # if data_min >= 0:
    #     colorscale = [
    #         [0.0, "white"],
    #         [1.0, "red"]
    #     ]
    #     zmin = 0
    #     zmax = data_max
    if data_min >= 0:
        colorscale = "RdBu"
        zmin = -data_max
        zmax = data_max
    else:
        colorscale = "RdBu"
        abs_max = max(abs(data_min), abs(data_max))
        zmin = -abs_max
        zmax = abs_max

    # -------------------------
    # Frames erzeugen
    # -------------------------
    frames = []
    for t in range(t_start_samp, t_end_samp, step_samp):
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=data[t],
                        zmin=zmin,
                        zmax=zmax,
                        colorscale=colorscale,
                        reversescale= True #bool((data_min < 0))
                    )
                ],
                name=str(t)
            )
        )

    # -------------------------
    # Initiales Frame
    # -------------------------
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=data[t_start_samp],
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                reversescale= True # bool((data_min < 0))
            )
        ],
        layout=go.Layout(
            title=(
                f"{meta['type']} im Mikrofonarray {meta['messung']}_{meta['saite']}_{meta['durchlauf']} "
                f"({t_start:.3f} s – "
                f"{(t_end_samp / sr):.3f} s)"
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": playspeed},
                                    "mode": "immediate"
                                }
                            ]
                        }
                    ]
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": f"{t / sr:.3f}s",
                            "method": "animate",
                            "args": [
                                [str(t)],
                                {"mode": "immediate", "frame": {"duration": 0}}
                            ]
                        }
                        for t in range(t_start_samp, t_end_samp, step_samp)
                    ],
                    "active": 0
                }
            ]
        ),
        frames=frames
    )

    fig.show()


# Mehrere Arraydaten aus Datensatz mit Metadaten laden

def export_picker(messung=5, saite=6, durchlauf=None, messung_2=None, suffix="rms"):

    # 3 Modi 1,2 und 5.
    # 1 Exportiert einfach ein durch a,b,c und suffix spezifiziertes Sample mit Metadaten
    # 2 Exportiert zwei aus unterschiedlichen Messreihen mit sonst gleichem Aufbau
    # 5 iteriert über 5 Durchläufe, alle anderen Parameter sind fest
    # Daten werden in einem Dictionary gespeichert mit data_array_{i} und meta_{i}

    def export_one(a, b, c, suffix):
        file = [f"dataset_{a}_{b}_{c}_{suffix}"]
        return file

    def export_two(a, a_2, b, c, suffix):
        files = []
        file_1 = f"dataset_{a}_{b}_{c}_{suffix}"
        file_2 = f"dataset_{a_2}_{b}_{c}_{suffix}"
        files.extend([file_1, file_2])
        return files

    def export_five(a, b, suffix):
        c=1
        files = []
        while c <= 5:
            file = f"dataset_{a}_{b}_{c}_{suffix}"
            c += 1
            files.append(file)
        return files
    
    # Entscheidet den Modus
    if messung_2 is None:
        if durchlauf is None:
            files = export_five(messung, saite, suffix)
        else:
            files = export_one(messung, saite, durchlauf, suffix)
    else:
        files = export_two(messung, messung_2, saite, durchlauf, suffix)
  
    print(files)
    # Lädt die Daten der generierten Filenames in ein Dictionary

    results = {}

    for i, file in enumerate(files, start=1):
        data, meta = load_npz(file)

        results[f"data_array_{i}"] = data
        results[f"meta_{i}"] = meta

    i=1
    while i <= round(len(results)/2):
        print(f"Created: {list(results)[i*2-2]}")
        print(f"Meta-Key {list(results)[i*2-1]} with Data: {results[f'meta_{i}']}")
        i += 1

    return results


# Mehrere Arraydaten aus Datensatz mit Metadaten laden

def export_picker2(messung, saite, durchlauf, suffix):

    """
    messung, saite, durchlauf:
        - int
        - tuple (start, stop)  -> inklusiver Range
        - list von konkreten Werten
    suffix:
        - string
    """

    def normalize(x):
        if isinstance(x, int):
            return [x]
        elif isinstance(x, tuple):
            return list(range(x[0], x[1] + 1))  # inklusiv
        elif isinstance(x, list):
            return x
        else:
            raise ValueError("Parameter muss int, tuple oder list sein")

    messung_vals = normalize(messung)
    saite_vals = normalize(saite)
    durchlauf_vals = normalize(durchlauf)

    files = [
        f"dataset_{a}_{b}_{c}_{suffix}.npz"
        for a, b, c in product(messung_vals, saite_vals, durchlauf_vals)
    ]

    # Lädt die Daten der generierten Filenames in ein Dictionary

    results = {}

    for i, file in enumerate(files, start=1):
        data, meta = load_npz(file)

        results[f"data_array_{i}"] = data
        results[f"meta_{i}"] = meta

    i=1
    while i <= round(len(results)/2):
        print(f"Created: {list(results)[i*2-2]}")
        print(f"Meta-Key {list(results)[i*2-1]} with Data: {results[f'meta_{i}']}")
        i += 1

    return results


# Differenz-Array erstellen

def create_difference_array(bulk_load):
    data = []
    meta = []
    for i in list(bulk_load):
        if i.count("data")>0:
            data.append(bulk_load[i])
        else:
            meta.append(bulk_load[i])
    
    messung_1 = meta[0]["messung"]
    messung_2 = meta[1]["messung"]

    T_1, rows, cols = data[0].shape
    T_2, rows, cols = data[1].shape

    if T_1 != T_2:
        raise ValueError("Arrays must have same length")
    T = T_1

    # Output-Array erzeugen
    data_diff = np.zeros((T, rows, cols), dtype=float)

    # Für jedes Mikrofon getrennt diff berechnen
    for y in range(rows):
        for x in range(cols):
            signal_1 = data[0][:, y, x]
            signal_2 = data[1][:, y, x]

            diff_values = []
            for values in zip(signal_1,signal_2):

                diff = values[0] - values[1]
                diff_values.append(diff)

            data_diff[:, y, x] = diff_values

    messung = c = int(f"{messung_1}{messung_2}")
    meta = meta[0]
    meta["messung"] = messung
    meta["type"] = f"d{meta['type']}"

    return data_diff, meta


def create_fft_difference_array(bulk_load):
    data = []
    meta = []
    for i in list(bulk_load):
        if i.count("data")>0:
            data.append(bulk_load[i])
        else:
            meta.append(bulk_load[i])
    
    messung_1 = meta[0]["messung"]
    messung_2 = meta[1]["messung"]

    print(messung_1, messung_2)

    T_1, rows, cols = data[0].shape
    T_2, rows, cols = data[1].shape

    if T_1 != T_2:
        raise ValueError("Arrays must have same length")
    T = T_1
    # print(T)

    # Output-Array erzeugen
    data_diff = np.zeros((T, rows, cols), dtype=float)
    # print(len(data_diff))

    # Für jedes Mikrofon getrennt diff berechnen
    epsilon = 1e-20

    for y in range(rows):
        for x in range(cols):
            signal_unmod = data[0][:, y, x] 
            signal_mod   = data[1][:, y, x]   
            # print(len(signal_mod))
            # print(len(signal_unmod))

            # nur die Werte, die > 0 sind, auf epsilon setzen
            signal_mod   = np.maximum(signal_mod, epsilon)
            signal_unmod = np.maximum(signal_unmod, epsilon)

            # Verhältnis in dB
            data_diff[:, y, x] = signal_mod / signal_unmod

    messung = c = int(f"{messung_1}{messung_2}")
    meta = meta[0]
    meta["messung"] = messung
    meta["type"] = f"d{meta['type']}"

    return data_diff, meta


def save_difference_array(messung_1: int, messung_2: int, saite=6, durchlauf=6, suffix="rms"):
    messung = [messung_1, messung_2]
    bulk_load = export_picker2(messung, saite, durchlauf, suffix)
    if "fft" in suffix:
        data, meta = create_fft_difference_array(bulk_load)
    else:
        data, meta = create_difference_array(bulk_load)
    save_npz(data, meta)

# Mittelwerte-Array erstellen

def create_mw_array(bulk_load):
    data = []
    meta = []
    for i in list(bulk_load):
        if i.count("data")>0:
            data.append(bulk_load[i])
        else:
            meta.append(bulk_load[i])
    
    T, rows , cols = data[0].shape
   
    # Output Array erzeugen
    data_mw = np.zeros((T, rows, cols), dtype=float)

    # für jedes Mikrofon getrennt mw berechnen
    for y in range(rows):
        for x in range(cols):
            signals = tuple(arr[:, y, x] for arr in data)
            count = len(signals)
            
            mw_values = []
            for values in zip(*signals):
                mw = sum(values) / count
                mw_values.append(mw)

            data_mw[:, y, x] = mw_values

    meta_1 = meta[0]

    meta_1["durchlauf"] = 6
    print(meta_1)

    return data_mw, meta_1

def create_sigma_array(bulk_load, mw_data, meta):
    data = []
    for i in list(bulk_load):
        if i.count("data")>0:
            data.append(bulk_load[i])
    
    T, rows , cols = data[0].shape

    # Output Array erzeugen
    sigma_data = np.zeros((T, rows, cols), dtype=float)

    # für jedes Mikrofon getrennt sigma berechnen
    for y in range(rows):
        for x in range(cols):
            signals = tuple(arr[:, y, x] for arr in data)
            mean_signal = mw_data[:, y, x]
            N = len(signals)
            
            sq_diff_sum = np.zeros_like(mean_signal)
            for sig in signals:
                sq_diff_sum += (sig - mean_signal)**2

            sigma_data[:, y, x] = np.sqrt(sq_diff_sum / N)

    meta["durchlauf"] = 7
    print(meta)

    return sigma_data, meta


def save_mw_array(messung=5, saite=6, suffix="rms"):
    bulk_load = export_picker(messung, saite, suffix=suffix)
    mw_data, mw_meta = create_mw_array(bulk_load)
    save_npz(mw_data, mw_meta)
    sigma_data, sigma_meta = create_sigma_array(bulk_load, mw_data, mw_meta)
    save_npz(sigma_data, sigma_meta)


# Signal to Noise Ratio

def harmonic_snr(
    y,
    sr,
    fmin=70,
    fmax=600,
    n_harmonics=15,
    n_fft=48192,
    hop_length=512,
    bw_bins=1
):
    """
    Schätzt das Signal-to-Noise Ratio (SNR) einer einzelnen Note
    mittels harmonischem Masking im Frequenzraum.

    Parameters
    ----------
    y : ndarray
        Audiosignal (1D)
    sr : int
        Samplerate
    fmin, fmax : float
        Suchbereich für Grundfrequenz (Hz)
    n_harmonics : int
        Anzahl berücksichtigter Harmonischer
    n_fft : int
        FFT-Größe
    hop_length : int
        Hop size
    bw_bins : int
        Anzahl FFT-Bins um jede Harmonische, die als Signal zählen

    Returns
    -------
    snr_db : float
        Signal-to-Noise Ratio in dB
    """

    # --- Grundfrequenz schätzen
    f0_series = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0 = np.nanmedian(f0_series)
    print(f0)

    if np.isnan(f0) or f0 <= 0:
        raise ValueError("Grundfrequenz konnte nicht zuverlässig bestimmt werden.")

    # --- STFT & Frequenzachse
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # --- Harmonische Bins sammeln
    harmonic_bins = set()
    nyquist = sr / 2

    for n in range(1, n_harmonics + 1):
        fn = n * f0
        if fn >= nyquist:
            break

        idx = np.argmin(np.abs(freqs - fn))
        for b in range(idx - bw_bins, idx + bw_bins + 1):
            if 0 <= b < len(freqs):
                harmonic_bins.add(b)

    harmonic_bins = np.array(sorted(harmonic_bins))
    all_bins = np.arange(len(freqs))
    noise_bins = np.setdiff1d(all_bins, harmonic_bins)

    # --- Leistung berechnen
    P_signal = np.mean(S[harmonic_bins, :])
    P_noise = np.mean(S[noise_bins, :])

    snr_db = 10 * np.log10(P_signal / P_noise)

    return snr_db

def compare_snr_db(messung: int, messung_2: int, saite=6, durchlauf=6):
    bulk_load = export_picker(messung=messung, saite=saite, durchlauf=durchlauf, messung_2=messung_2, suffix = "fwav")
    data = []
    sr = bulk_load[list(bulk_load)[1]]["sr"]

    for i in list(bulk_load):
        if i.count("data")>0:
            data.append(bulk_load[i][:,1,5])
        else:
            print(bulk_load[i])

    snr_list = []

    for i in data:
        snr = harmonic_snr(i,sr)
        snr_list.append(snr)

    print(snr_list)


# STFT

def plot_stft(y, sr, n_fft = 4096, hop_length = 512):

    y, sr = librosa.load("note.wav", sr=None)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spectrum = S_db.mean(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=mean_spectrum,
        mode="lines",
        name="Mean Spectrum"
    ))

    fig.update_layout(
        title="Mean Spectrum (STFT averaged)",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude [dB]",
        xaxis_type="log"
    )

    fig.show()

# FFT-Array erstellen

def compute_spectrum(y, sr):
    """
    Berechnet das einseitige Amplitudenspektrum eines reellen Signals.

    Parameter
    ----------
    y : list oder numpy array
        Zeitdiskretes Signal
    fs : float
        Abtastrate in Hz

    Returns
    -------
    freqs : numpy array
        Frequenzachse (Hz)
    magnitude : numpy array
        Betragsspektrum
    """
    
    # In numpy-Array umwandeln
    y = np.asarray(y)
    N = len(y)
    
    # FFT (nur positive Frequenzen)
    Y = np.fft.rfft(y)
    
    # Magnitude
    magnitude = np.abs(Y)
    
    # Frequenzachse
    freqs = np.fft.rfftfreq(N, d=1/sr)
    freqs = freqs.tolist()
    
    return freqs, magnitude

def compute_spectrum_array(data, meta):

    sr = meta["sr"]
    T, rows, cols = data.shape

    T_fft = int(T/2+1)

    # Output-Array erzeugen
    data_fft = np.zeros((T_fft, rows, cols), dtype=float)

    # Für jedes Mikrofon getrennt Spekturm berechnen
    for y in range(rows):
        for x in range(cols):
            signal = data[:, y, x]

            # print(y,x)

            freqs, fft_values = compute_spectrum(signal, sr)

            data_fft[:, y, x] = fft_values

    meta["freqs"] = freqs

    # "Samplerate" der neuen Zeitachse: Frames pro Sekunde
    meta["type"]="fft"
    meta["freqs"] = freqs

    # print("Done ", meta["type"])

    return data_fft, meta


# Spektrum plotten

def plot_spectrum(data, meta, x, y, title=None, log_freq=False, log_mag=False, freq_min=20, freq_max=None):

    # Frequenzachse
    freqs = np.array(meta["freqs"])

    # Spektrum für das spezifische Mikrofon auswählen
    spectrum = data[:, y, x]

    # Optional Magnitude in dB
    if log_mag:
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))

    # Frequenzbereich einschränken
    mask = np.ones_like(freqs, dtype=bool)
    if freq_min is not None:
        mask &= freqs >= freq_min
    if freq_max is not None:
        mask &= freqs <= freq_max

    freqs_plot = freqs[mask]
    spectrum_plot = spectrum[mask]

    # Plotly Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs_plot,
        y=spectrum_plot,
        mode='lines',
        name=f'Mikrofon ({x},{y})'
    ))

    # Achsen anpassen
    xaxis_type = "log" if log_freq else "linear"
    yaxis_title = 'Magnitude [dB]' if log_mag else 'Magnitude'

    fig.update_layout(
        title=title or f'Spektrum Mikrofon ({x},{y})',
        xaxis_title='Frequenz [Hz]',
        xaxis_type=xaxis_type,
        yaxis_title=yaxis_title,
        template='plotly_white'
    )
    tick_vals=[20,50,100,200,500,1000,2000,5000,10000, 20000]
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=[str(v) for v in tick_vals])

    fig.show()




def plot_spectrum2(data, meta, x, y, title=None, log_freq=False, log_mag=False, freq_min=20, freq_max=None):
    """
    Plottet das Magnitudenspektrum eines spezifischen Mikrofons mit optionaler Standardabweichung.

    Parameters
    ----------
    data : np.ndarray
        Magnitudes, Shape (N_Frequenzbins, Mikrofone_x, Mikrofone_y)
    meta : dict
        Metadaten, muss 'freqs' enthalten
    x : int
        Index Mikrofon x
    y : int
        Index Mikrofon y
    sigma_data : np.ndarray, optional
        Standardabweichung, gleiche Form wie data[:, y, x]
    title : str, optional
        Plot-Titel
    log_freq : bool, optional
        Ob die Frequenzachse logarithmisch dargestellt werden soll
    log_mag : bool, optional
        Ob Magnitude in dB dargestellt werden soll
    freq_min : float, optional
        Untere Frequenzgrenze (Hz), Standard 20 Hz
    freq_max : float, optional
        Obere Frequenzgrenze (Hz)
    """
    # Frequenzachse
    freqs = np.array(meta["freqs"])
    spectrum = data[:, y, x]

    # if meta["durchlauf"] == 6 and meta["type"][0] != "d":
    #     sigma_data, _ = load_npz(f"dataset_{meta['messung']}_{meta['saite']}_7_{meta['type']}")
    #     print(sigma_data[:, y, x])
    # else:
    #     sigma_data = None
    sigma_data = None

    if log_mag:
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))
        if sigma_data is not None:
            sigma_data = 20 * np.log10(np.maximum(data[:, y, x] + sigma_data[:, y, x], 1e-12)) - spectrum

    # Frequenzbereich einschränken
    mask = np.ones_like(freqs, dtype=bool)
    if freq_min is not None:
        mask &= freqs >= freq_min
    if freq_max is not None:
        mask &= freqs <= freq_max

    freqs_plot = freqs[mask]
    spectrum_plot = spectrum[mask]

    fig = go.Figure()

    # Daten plotten
    fig.add_trace(go.Scatter(
        x=freqs_plot,
        y=spectrum_plot,
        mode='lines',
        name='Mittelwert',
        line=dict(color='blue', width=2)
    ))

    # Wenn sigma_data angegeben, +sigma und -sigma plotten
    if sigma_data is not None:
        sigma_plot = sigma_data[mask]

        # Upper/Lower curves
        upper = spectrum_plot + sigma_plot
        lower = spectrum_plot - sigma_plot

        # Shaded area zwischen ±sigma
        fig.add_trace(go.Scatter(
            x=np.concatenate([freqs_plot, freqs_plot[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='±1σ'
        ))

        # Optional Linien für data+sigma und data-sigma
        fig.add_trace(go.Scatter(
            x=freqs_plot,
            y=upper,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Mittelwert + σ'
        ))
        fig.add_trace(go.Scatter(
            x=freqs_plot,
            y=lower,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Mittelwert - σ'
        ))

    # Achsen
    xaxis_type = "log" if log_freq else "linear"
    yaxis_title = 'Magnitude [dB]' if log_mag else 'Magnitude'

    fig.update_layout(
        title=title or f'Spektrum Mikrofon ({x},{y})',
        xaxis_title='Frequenz [Hz]',
        xaxis_type=xaxis_type,
        yaxis_title=yaxis_title,
        template='plotly_white'
    )

    tick_vals = [20,50,100,200,500,1000,2000,5000,10000, 20000]
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=[str(v) for v in tick_vals]
    )

    fig.show()


    from scipy.ndimage import gaussian_filter1d
import numpy as np
import plotly.graph_objects as go


def plot_spectrum3(
    data,
    meta,
    x,
    y,
    title=None,
    log_freq=False,
    log_mag=False,
    freq_min=20,
    freq_max=None,
    gaussian_smooth=False,
    sigma_smooth=2
):
    """
    Plottet das Magnitudenspektrum eines spezifischen Mikrofons
    mit optionaler Gauß-Glättung.
    """

    # Frequenzachse
    freqs = np.array(meta["freqs"])
    spectrum = data[:, y, x]
    sigma_data = None

    # In dB umrechnen falls gewünscht
    if log_mag:
        spectrum = 20 * np.log10(np.maximum(spectrum, 1e-12))
        if sigma_data is not None:
            sigma_data = 20 * np.log10(
                np.maximum(data[:, y, x] + sigma_data[:, y, x], 1e-12)
            ) - spectrum

    # 👉 OPTIONALE GAUSS-GLÄTTUNG
    if gaussian_smooth:
        spectrum = gaussian_filter1d(spectrum, sigma=sigma_smooth)
        if sigma_data is not None:
            sigma_data = gaussian_filter1d(sigma_data, sigma=sigma_smooth)

    # Frequenzbereich einschränken
    mask = np.ones_like(freqs, dtype=bool)
    if freq_min is not None:
        mask &= freqs >= freq_min
    if freq_max is not None:
        mask &= freqs <= freq_max

    freqs_plot = freqs[mask]
    spectrum_plot = spectrum[mask]

    fig = go.Figure()

    # Hauptkurve
    fig.add_trace(go.Scatter(
        x=freqs_plot,
        y=spectrum_plot,
        mode='lines',
        name='Mittelwert',
        line=dict(color='blue', width=2)
    ))

    # ±Sigma Bereich
    if sigma_data is not None:
        sigma_plot = sigma_data[mask]

        upper = spectrum_plot + sigma_plot
        lower = spectrum_plot - sigma_plot

        fig.add_trace(go.Scatter(
            x=np.concatenate([freqs_plot, freqs_plot[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='±1σ'
        ))

        fig.add_trace(go.Scatter(
            x=freqs_plot,
            y=upper,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Mittelwert + σ'
        ))

        fig.add_trace(go.Scatter(
            x=freqs_plot,
            y=lower,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Mittelwert - σ'
        ))

    # Achsen
    xaxis_type = "log" if log_freq else "linear"
    yaxis_title = 'Magnitude [dB]' if log_mag else 'Magnitude'

    fig.update_layout(
        title=title or f'Spektrum Mikrofon ({x},{y})',
        xaxis_title='Frequenz [Hz]',
        xaxis_type=xaxis_type,
        yaxis_title=yaxis_title,
        template='plotly_white'
    )

    tick_vals = [20,50,100,200,500,1000,2000,5000,10000,20000]
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=[str(v) for v in tick_vals]
    )

    fig.show()


def process_pickup_folder(input_dir, output_dir, onsets):

    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r"(\d+)_(\d+)_(\d+)_(\d+)_(\d{2})\.wav")

    processed_files = 0
    skipped_files = 0

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if not match:
            skipped_files += 1
            continue

        m, s, d, y, xx = map(int, match.groups())

        # Nur Pickup-Position
        if y != 4 or xx != 1:
            continue

        filepath = os.path.join(input_dir, fname)

        try:
            sr, signal = wavfile.read(filepath)

            # Stereo → Mono
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Finalisieren
            signal_final = finalize_pickup(signal, sr, m, s, d, onsets)

            # === Optional: saubere Normierung (empfohlen) ===
            if np.max(np.abs(signal_final)) > 0:
                signal_final = signal_final / np.max(np.abs(signal_final))
            signal_final = (signal_final * 32767).astype(np.int16)

            # Speichern
            out_name = f"{m}_{s}_{d}_4_01_final.wav"
            out_path = os.path.join(output_dir, out_name)

            wavfile.write(out_path, sr, signal_final)

            processed_files += 1

        except Exception as e:
            print(f"Fehler bei {fname}: {e}")
            skipped_files += 1

    print(f"Fertig! {processed_files} Dateien verarbeitet, {skipped_files} übersprungen.")


def compute_pickup_spectra(input_dir, output_dir):


    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r"(\d+)_(\d+)_(\d+)_4_01_final\.wav")

    processed = 0

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if not match:
            continue

        m, s, d = map(int, match.groups())

        filepath = os.path.join(input_dir, fname)

        try:
            sr, signal = wavfile.read(filepath)

            # Falls int → in float umwandeln
            signal = signal.astype(float)

            # === FFT ===
            fft = np.fft.rfft(signal)
            spectrum = np.abs(fft)

            # Frequenzachse
            freqs = np.fft.rfftfreq(len(signal), d=1/sr)

            # === Speichern ===
            base = f"{m}_{s}_{d}_4_01"

            np.save(os.path.join(output_dir, base + "_spec.npy"), spectrum)
            np.save(os.path.join(output_dir, base + "_freqs.npy"), freqs)

            processed += 1

        except Exception as e:
            print(f"Fehler bei {fname}: {e}")

    print(f"Fertig! {processed} Spektren berechnet.")


def compute_mean_std_spectra(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r"(\d+)_(\d+)_(\d+)_4_01_spec\.npy")

    # Datenstruktur: {(m, s): [spektren]}
    data = {}

    # === Dateien sammeln ===
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if not match:
            continue

        m, s, d = map(int, match.groups())

        spec = np.load(os.path.join(input_dir, fname))

        key = (m, s)

        if key not in data:
            data[key] = []

        data[key].append(spec)

    # === Mittelwert & Std berechnen ===
    for (m, s), spectra in data.items():
        spectra = np.array(spectra)

        mean_spec = np.mean(spectra, axis=0)
        std_spec = np.std(spectra, axis=0)

        base = f"m{m}_s{s}_4_01"

        np.save(os.path.join(output_dir, base + "_mean.npy"), mean_spec)
        np.save(os.path.join(output_dir, base + "_std.npy"), std_spec)

    print(f"Fertig! {len(data)} Gruppen verarbeitet.")
    print({np.load(os.path.join(input_dir, f)).shape for f in os.listdir(input_dir) if f.endswith("_spec.npy")})


def analyze_pickup_spectrum(m, s,
                            base_dir="../data/pickups",
                            thresh=-9,
                            prom=0.1,
                            fmax=5000,
                            save=True,
                            show_peak_labels=True):

    # === Pfade ===
    stats_dir = os.path.join(base_dir, "pickups_stats")
    spec_dir = os.path.join(base_dir, "pickups_spectra")
    plots_dir = os.path.join(base_dir, "pickups_plots")

    os.makedirs(plots_dir, exist_ok=True)

    base = f"m{m}_s{s}_4_01"

    mean_file = os.path.join(stats_dir, base + "_mean.npy")
    std_file = os.path.join(stats_dir, base + "_std.npy")
    freqs_file = os.path.join(spec_dir, "freqs.npy")

    # === Safety ===
    for f in [mean_file, std_file, freqs_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Fehlende Datei: {f}")

    # === Laden ===
    mean_spec = np.load(mean_file)
    std_spec = np.load(std_file)
    freqs = np.load(freqs_file)

    # === Frequenz-Maske (WICHTIG) ===
    mask = freqs <= fmax

    freqs = freqs[mask]
    mean_spec = mean_spec[mask]
    std_spec = std_spec[mask]

    # === Normierung ===
    max_val = np.max(mean_spec)

    mean_db = np.log(mean_spec / max_val)

    upper_db = np.log((mean_spec + std_spec) / max_val)
    lower_db = np.log(np.maximum(mean_spec - std_spec, 1e-12) / max_val)

    # === Peaks ===
    peaks, _ = find_peaks(mean_db, prominence=prom, height=thresh)

    peak_freqs = freqs[peaks]
    peak_vals = mean_db[peaks]
    labels = [f"{f:.1f} Hz" for f in peak_freqs]

    # === Plot ===
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=mean_db,
        mode="lines",
        name="Mean Spectrum"
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([freqs, freqs[::-1]]),
        y=np.concatenate([upper_db, lower_db[::-1]]),
        fill="toself",
        fillcolor="rgba(0,100,200,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="± Std"
    ))

    if show_peak_labels:
        fig.add_trace(go.Scatter(
            x=peak_freqs,
            y=peak_vals,
            mode="markers+text",
            name="Peaks",
            marker=dict(color="red", size=6),
            text=labels,
            textposition="top center"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=peak_freqs,
            y=peak_vals,
            mode="markers",
            name="Peaks",
            marker=dict(color="red", size=6)
        ))

    fig.update_layout(
        title=f"Pickup Spectrum m={m}, s={s}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (log scale)",
        template="plotly_white"
    )

    fig.show()

    # === Speichern ===
    if save:
        out_file = os.path.join(plots_dir, f"{base}_plot.html")
        fig.write_html(out_file)

    return peaks, peak_freqs


def compare_pickup_spectrum(m_1, m_2, s,
                             base_dir="../data/pickups",
                             fmax=5000,
                             prom=0.1,
                             thresh=-9,
                             show_peaks=True,
                             save=True):

    import os
    import numpy as np
    import plotly.graph_objects as go
    from scipy.signal import find_peaks

    # === Pfade ===
    stats_dir = os.path.join(base_dir, "pickups_stats")
    spec_dir = os.path.join(base_dir, "pickups_spectra")
    plots_dir = os.path.join(base_dir, "pickups_plots")

    os.makedirs(plots_dir, exist_ok=True)

    base_1 = f"m{m_1}_s{s}_4_01"
    base_2 = f"m{m_2}_s{s}_4_01"

    mean_1_file = os.path.join(stats_dir, base_1 + "_mean.npy")
    mean_2_file = os.path.join(stats_dir, base_2 + "_mean.npy")
    freqs_file = os.path.join(spec_dir, "freqs.npy")

    # === Safety ===
    for f in [mean_1_file, mean_2_file, freqs_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Fehlende Datei: {f}")

    # === Laden ===
    mean_1 = np.load(mean_1_file)
    mean_2 = np.load(mean_2_file)
    freqs = np.load(freqs_file)

    # === Frequenz-Maske ===
    mask = freqs <= fmax

    freqs = freqs[mask]
    mean_1 = mean_1[mask]
    mean_2 = mean_2[mask]

    # === Normierung (je separat, wichtig für Vergleich der Form!) ===
    mean_1_db = np.log(mean_1 / np.max(mean_1))
    mean_2_db = np.log(mean_2 / np.max(mean_2))

    # === Peaks ===
    peaks_1, _ = find_peaks(mean_1_db, prominence=prom, height=thresh)
    peaks_2, _ = find_peaks(mean_2_db, prominence=prom, height=thresh)

    # === Plot ===
    fig = go.Figure()

    # Spektrum 1
    fig.add_trace(go.Scatter(
        x=freqs,
        y=mean_1_db,
        mode="lines",
        name=f"m={m_1}, s={s}"
    ))

    # Spektrum 2
    fig.add_trace(go.Scatter(
        x=freqs,
        y=mean_2_db,
        mode="lines",
        name=f"m={m_2}, s={s}"
    ))

    # Peaks optional
    if show_peaks:
        fig.add_trace(go.Scatter(
            x=freqs[peaks_1],
            y=mean_1_db[peaks_1],
            mode="markers",
            name=f"Peaks m={m_1}",
            marker=dict(color="blue", size=6)
        ))

        fig.add_trace(go.Scatter(
            x=freqs[peaks_2],
            y=mean_2_db[peaks_2],
            mode="markers",
            name=f"Peaks m={m_2}",
            marker=dict(color="red", size=6)
        ))

    # === Layout ===
    fig.update_layout(
        title=f"Pickup Spectrum Comparison (s={s})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (log scale)",
        template="plotly_white"
    )

    fig.show()

    # === Save ===
    if save:
        out_file = os.path.join(
            plots_dir,
            f"compare_m{m_1}_m{m_2}_s{s}.html"
        )
        fig.write_html(out_file)

    return peaks_1, peaks_2


def difference_pickup_spectrum(m_1, m_2, s,
                                base_dir="../data/pickups",
                                fmax=5000,
                                prom=0.1,
                                thresh=-9,
                                smooth_sigma=0,
                                show_peaks=True,
                                save=True):

    # === Pfade ===
    stats_dir = os.path.join(base_dir, "pickups_stats")
    spec_dir = os.path.join(base_dir, "pickups_spectra")
    plots_dir = os.path.join(base_dir, "pickups_plots")

    os.makedirs(plots_dir, exist_ok=True)

    base_1 = f"m{m_1}_s{s}_4_01"
    base_2 = f"m{m_2}_s{s}_4_01"

    mean_1_file = os.path.join(stats_dir, base_1 + "_mean.npy")
    mean_2_file = os.path.join(stats_dir, base_2 + "_mean.npy")
    freqs_file = os.path.join(spec_dir, "freqs.npy")

    # === Safety ===
    for f in [mean_1_file, mean_2_file, freqs_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Fehlende Datei: {f}")

    # === Laden ===
    S1 = np.load(mean_1_file)
    S2 = np.load(mean_2_file)
    freqs = np.load(freqs_file)

    # === Frequenz-Maske ===
    mask = freqs <= fmax

    freqs = freqs[mask]
    S1 = S1[mask]
    S2 = S2[mask]

    # === Log-Normalisierung (wie in deiner Pipeline) ===
    S1 = np.log(S1 / np.max(S1))
    S2 = np.log(S2 / np.max(S2))

     # === Peaks (auf Einzel-Spektren!) ===
    peaks_1, _ = find_peaks(S1, prominence=prom, height=thresh)
    peaks_2, _ = find_peaks(S2, prominence=prom, height=thresh)

    # === Optional Glättung ===
    if smooth_sigma > 0:
        S1 = gaussian_filter1d(S1, sigma=smooth_sigma)
        S2 = gaussian_filter1d(S2, sigma=smooth_sigma)

    # === Differenz ===
    diff = S2 - S1
    diff = diff - np.mean(diff)

    peak_freqs_1 = freqs[peaks_1]
    peak_freqs_2 = freqs[peaks_2]

    # === Plot ===
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=diff,
        mode="lines",
        name=f"Δ (m{m_2} - m{m_1})"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # === Peaks optional ===
    if show_peaks:

        # Peaks m1 (auf Differenz interpoliert)
        fig.add_trace(go.Scatter(
            x=peak_freqs_1,
            y=np.interp(peak_freqs_1, freqs, diff),
            mode="markers",
            name=f"Peaks m{m_1}",
            marker=dict(color="blue", size=7)
        ))

        # Peaks m2
        fig.add_trace(go.Scatter(
            x=peak_freqs_2,
            y=np.interp(peak_freqs_2, freqs, diff),
            mode="markers",
            name=f"Peaks m{m_2}",
            marker=dict(color="red", size=7)
        ))

    fig.update_layout(
        title=f"Differenzspektrum s={s} (m{m_2} - m{m_1})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Δ Amplitude (log scale)",
        template="plotly_white"
    )

    fig.show()

    # === Save ===
    if save:
        out_file = os.path.join(
            plots_dir,
            f"diff_m{m_2}_m{m_1}_s{s}_sigma{smooth_sigma}.html"
        )
        fig.write_html(out_file)

    return freqs, diff


def plot_spectrum4(
    data,
    meta,
    x,
    y,
    thresh=-9,
    prom=3,
    fmax=5000,
    show_peak_labels=False,
    log_mag=True,
    gaussian_smooth=False,
    sigma_smooth=2,
    save=False,
    save_dir="../data/arrays_plots"
):

    # === Frequenz + Spektrum ===
    freqs = np.array(meta["freqs"])
    spectrum = data[:, y, x]

    # === Optional Glättung ===
    if gaussian_smooth:
        spectrum = gaussian_filter1d(spectrum, sigma=sigma_smooth)

    # === Log-Skala ===
    if log_mag:
        spectrum = np.log(np.maximum(spectrum, 1e-12))

    # === Frequenz-Maske ===
    mask = freqs <= fmax

    freqs = freqs[mask]
    spectrum = spectrum[mask]

    # === Peaks ===
    peaks, _ = find_peaks(spectrum, prominence=prom, height=thresh)

    peak_freqs = freqs[peaks]
    peak_vals = spectrum[peaks]
    labels = [f"{f:.1f} Hz" for f in peak_freqs]

    # === Plot ===
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=spectrum,
        mode="lines",
        name="Spectrum"
    ))

    # === Peaks ===
    if show_peak_labels:
        fig.add_trace(go.Scatter(
            x=peak_freqs,
            y=peak_vals,
            mode="markers+text",
            name="Peaks",
            marker=dict(color="red", size=6),
            text=labels,
            textposition="top center"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=peak_freqs,
            y=peak_vals,
            mode="markers",
            name="Peaks",
            marker=dict(color="red", size=6)
        ))

    # === Layout ===
    fig.update_layout(
        title=f"Spektrum Mikrofon ({x},{y})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (log scale)" if log_mag else "Amplitude",
        template="plotly_white"
    )

    fig.show()

    # === Save ===
    if save:
        os.makedirs(save_dir, exist_ok=True)

        out_file = os.path.join(
            save_dir,
            f"spectrum_x{x}_y{y}.html"
        )
        fig.write_html(out_file)

    return peaks, peak_freqs


def compare_spectrum4(
    data1, meta1,
    data2, meta2,
    x, y,
    thresh=-9,
    prom=3,
    fmax=5000,
    show_peaks=True,
    log_mag=True,
    gaussian_smooth=False,
    sigma_smooth=2
):

    # === Frequenzen ===
    freqs = np.array(meta1["freqs"])

    s1 = data1[:, y, x]
    s2 = data2[:, y, x]

    # === Glättung ===
    if gaussian_smooth:
        s1 = gaussian_filter1d(s1, sigma=sigma_smooth)
        s2 = gaussian_filter1d(s2, sigma=sigma_smooth)

    # === Log ===
    if log_mag:
        s1 = np.log(np.maximum(s1, 1e-12))
        s2 = np.log(np.maximum(s2, 1e-12))

    # === Mask ===
    mask = freqs <= fmax
    freqs = freqs[mask]
    s1 = s1[mask]
    s2 = s2[mask]

    # === Peaks ===
    peaks1, _ = find_peaks(s1, prominence=prom, height=thresh)
    peaks2, _ = find_peaks(s2, prominence=prom, height=thresh)

    # === Plot ===
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=s1,
        mode="lines",
        name="Signal 1"
    ))

    fig.add_trace(go.Scatter(
        x=freqs,
        y=s2,
        mode="lines",
        name="Signal 2"
    ))

    if show_peaks:

        fig.add_trace(go.Scatter(
            x=freqs[peaks1],
            y=s1[peaks1],
            mode="markers",
            name="Peaks 1",
            marker=dict(color="blue", size=6)
        ))

        fig.add_trace(go.Scatter(
            x=freqs[peaks2],
            y=s2[peaks2],
            mode="markers",
            name="Peaks 2",
            marker=dict(color="red", size=6)
        ))

    fig.update_layout(
        title=f"Spectrum Comparison ({x},{y})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (log)" if log_mag else "Amplitude",
        template="plotly_white"
    )

    fig.show()

    return peaks1, peaks2


def difference_spectrum4(
    data1, meta1,
    data2, meta2,
    x, y,
    fmax=5000,
    smooth_sigma=0,
    show_peaks=False,
    prom=3,
    thresh=-9,
    log_mag=True
):

    import numpy as np
    import plotly.graph_objects as go
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    freqs = np.array(meta1["freqs"])

    s1 = data1[:, y, x]
    s2 = data2[:, y, x]

    # === log ===

    if log_mag:
        s1 = np.log(np.maximum(s1, 1e-12))
        s2 = np.log(np.maximum(s2, 1e-12))

     # === mask ===
    mask = freqs <= fmax
    freqs = freqs[mask]
    s1 = s1[mask]
    s2 = s2[mask]
        
    # === peaks optional ===
    peaks1, _ = find_peaks(s1, prominence=prom, height=thresh)
    peaks2, _ = find_peaks(s2, prominence=prom, height=thresh)

    # === optional smoothing ===
    if smooth_sigma > 0:
        s1 = gaussian_filter1d(s1, sigma=smooth_sigma)
        s2 = gaussian_filter1d(s2, sigma=smooth_sigma)



    # === difference ===
    diff = s2 - s1
    diff = diff - np.mean(diff)

    # === plot ===
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=diff,
        mode="lines",
        name="Δ (data2 - data1)"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    if show_peaks:
        fig.add_trace(go.Scatter(
            x=freqs[peaks1],
            y=diff[peaks1],
            mode="markers",
            name="Peaks Δ",
            marker=dict(color="red", size=6)
        ))
        fig.add_trace(go.Scatter(
            x=freqs[peaks2],
            y=diff[peaks2],
            mode="markers",
            name="Peaks Δ",
            marker=dict(color="blue", size=6)
        ))

    fig.update_layout(
        title=f"Differenzspektrum ({x},{y})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Δ Amplitude",
        template="plotly_white"
    )

    fig.show()

    return freqs, diff


def create_fft_heatmap(data, meta, bin_step=1, target_freq=None):
    """
    FFT-Heatmap:
    - Ohne target_freq → Slider über Bins
    - Mit target_freq → Einzelplot nächstgelegener Frequenz

    Parameters
    ----------
    data : ndarray (F, rows, cols)
    meta : dict (enthält 'freqs')
    bin_step : int
        Schrittweite über FFT-Bins
    target_freq : float or None
        Ziel-Frequenz in Hz
    """

    freqs = np.array(meta["freqs"])
    F, rows, cols = data.shape

    # -------------------------
    # Farbskala (unipolar)
    # -------------------------
    zmin = 0
    zmax = data.max()
    colorscale = "Viridis"

    # ==========================================================
    # 🔸 FALL 1: Einzel-Frequenz
    # ==========================================================
    if target_freq is not None:

        idx = int(np.argmin(np.abs(freqs - target_freq)))
        actual_freq = freqs[idx]

        fig = go.Figure(
            data=go.Heatmap(
                z=data[idx],
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale
            )
        )

        fig.update_layout(
            title=(
                f"FFT Heatmap | Ziel: {target_freq:.2f} Hz | "
                f"Genutzt: {actual_freq:.2f} Hz (Bin {idx})<br>"
                f"{meta['messung']}_{meta['saite']}_{meta['durchlauf']}"
            )
        )

        fig.show()
        return

    # ==========================================================
    # 🔸 FALL 2: Slider über Bins
    # ==========================================================

    frames = []
    for i in range(0, F, bin_step):
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=data[i],
                        zmin=zmin,
                        zmax=zmax,
                        colorscale=colorscale
                    )
                ],
                name=str(i)
            )
        )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=data[0],
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale
            )
        ],
        layout=go.Layout(
            title=(
                f"FFT Heatmap (Slider) "
                f"{meta['messung']}_{meta['saite']}_{meta['durchlauf']}"
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100},
                                    "mode": "immediate"
                                }
                            ]
                        }
                    ]
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": f"{freqs[i]:.1f} Hz",
                            "method": "animate",
                            "args": [
                                [str(i)],
                                {"mode": "immediate", "frame": {"duration": 0}}
                            ]
                        }
                        for i in range(0, F, bin_step)
                    ],
                    "active": 0
                }
            ]
        ),
        frames=frames
    )

    fig.show()


def create_fft_heatmap2(
    data, meta,
    bin_step=1,
    target_freq=None,
    normalize=True,
    fmin=None,
    fmax=None,
    scale_mode="linear"  # "linear" oder "db"
):
    """
    FFT Heatmap + Spektrum (physikalisch konsistent)

    Features:
    ----------
    - Slider oder Einzel-Frequenz
    - Normierung nur auf Maximum
    - optional dB-Skala
    - Frequenzbereich einschränkbar
    - Spektrum des mittleren Mikros + Marker
    - korrekte Mikrofon-Geometrie:
        x: 1–cols (links → rechts)
        y: 1–rows (oben → unten)
    """

    freqs = np.array(meta["freqs"])
    F, rows, cols = data.shape

    # -------------------------
    # Frequenzbereich
    # -------------------------
    if fmin is None:
        fmin = freqs[0]
    if fmax is None:
        fmax = freqs[-1]

    idx_min = np.searchsorted(freqs, fmin)
    idx_max = np.searchsorted(freqs, fmax)

    # -------------------------
    # Achsen (physikalisch korrekt)
    # -------------------------
    x_vals = np.arange(1, cols + 1)
    y_vals = np.arange(1, rows + 1)

    # -------------------------
    # mittleres Mikro
    # -------------------------
    mid_y = rows // 2
    mid_x = cols // 2
    spectrum_mid = data[:, mid_y, mid_x]
    spectrum_plot = 20 * np.log10(spectrum_mid + 1e-12)

    # -------------------------
    # Frame-Verarbeitung
    # -------------------------
    def process_frame(frame):

        if scale_mode == "db":
            frame = 20 * np.log10(frame + 1e-12)
            if normalize:
                frame = frame - np.max(frame)  # max = 0 dB

        else:  # linear
            if normalize:
                max_val = np.max(frame)
                if max_val > 0:
                    frame = frame / max_val

        return frame

    # -------------------------
    # Farbskala
    # -------------------------
    if scale_mode == "db":
        zmin = -40
        zmax = 0
        colorscale = "Inferno"
    else:
        zmin = 0
        zmax = 1 if normalize else data.max()
        colorscale = "Viridis"

    # ==========================================================
    # 🔸 SINGLE FREQUENCY MODE
    # ==========================================================
    if target_freq is not None:

        idx = int(np.argmin(np.abs(freqs - target_freq)))
        actual_freq = freqs[idx]

        heat = process_frame(data[idx])

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.15
        )

        # Heatmap
        fig.add_trace(
            go.Heatmap(
                z=heat,
                x=x_vals,
                y=y_vals,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax
            ),
            row=1, col=1
        )

        # Spektrum
        fig.add_trace(
            go.Scatter(
                x=freqs[idx_min:idx_max],
                y=spectrum_plot[idx_min:idx_max],
                mode="lines",
                name="Spektrum (Mitte)"
            ),
            row=2, col=1
        )

        # Marker
        fig.add_trace(
            go.Scatter(
                x=[actual_freq],
                y=[spectrum_plot[idx]],
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="aktuelle Frequenz"
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=(
                f"Ziel: {target_freq:.2f} Hz | "
                f"Genutzt: {actual_freq:.2f} Hz (Bin {idx})"
            ),
            xaxis=dict(title="Mikrofon x"),
            yaxis=dict(
                title="Mikrofon y",
                tickmode="array",
                tickvals=y_vals,
                autorange="reversed"
            )
        )

        fig.show()
        return

    # ==========================================================
    # 🔸 SLIDER MODE
    # ==========================================================

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.15
    )

    # Initial Frame
    heat0 = process_frame(data[idx_min])

    fig.add_trace(
        go.Heatmap(
            z=heat0,
            x=x_vals,
            y=y_vals,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=freqs[idx_min:idx_max],
            y=spectrum_plot[idx_min:idx_max],
            mode="lines"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[freqs[idx_min]],
            y=[spectrum_plot[idx_min]],
            mode="markers",
            marker=dict(size=10, color="blue")
        ),
        row=2, col=1
    )

    # Frames
    frames = []
    for i in range(idx_min, idx_max, bin_step):

        heat = process_frame(data[i])

        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(z=heat, x=x_vals, y=y_vals),
                    go.Scatter(
                        x=freqs[idx_min:idx_max],
                        y=spectrum_plot[idx_min:idx_max]
                    ),
                    go.Scatter(
                        x=[freqs[i]],
                        y=[spectrum_plot[i]]
                    )
                ],
                name=str(i)
            )
        )

    fig.frames = frames

    fig.update_layout(
        sliders=[
            {
                "steps": [
                    {
                        "label": f"{freqs[i]:.1f} Hz",
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {"mode": "immediate", "frame": {"duration": 0}}
                        ]
                    }
                    for i in range(idx_min, idx_max, bin_step)
                ]
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100}}]
                    }
                ]
            }
        ],
        title=f"FFT Heatmap ({fmin:.1f} – {fmax:.1f} Hz) | scale: {scale_mode}",
        xaxis=dict(title="Mikrofon x"),
        yaxis=dict(
            title="Mikrofon y",
            tickmode="array",
            tickvals=y_vals,
            autorange="reversed"
        )
    )

    fig.show()