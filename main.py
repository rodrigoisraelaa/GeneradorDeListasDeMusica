from os import path
from os import mkdir
from tkinter import filedialog, Button, END, Text, Listbox, Tk
from numpy.linalg import norm
from numpy import dot
from numpy import linspace
from numpy import ndarray
from numpy import max as ma
from numpy import min as mi
from numpy import sum as suma
from scipy.spatial import geometric_slerp
from metodos import load, power_to_db
from librosa.feature import melspectrogram
from tensorflow import device
from keras.models import load_model
from pandas import read_json
import os
import shutil

roottxt = 'roottxt.txt'
global root
if path.exists(roottxt):
    with open(roottxt, 'r') as fp:
        root = fp.readlines()[0]
        if not root[-1] == '/':
            root = root + '/'
else:
    root = ''


def seleccionarCarpeta():
    dir = filedialog.askdirectory()
    global root
    root = dir
    with open(roottxt, 'w') as filep:
        filep.write(root)


def openFile():
    filepath = filedialog.askopenfilename()
    if Ta.size() < 1:
        Ta.insert(END, 'origen: ' + filepath)
    else:
        Ta.insert(END, 'deseo: ' + filepath)


def limpiarTextbox():
    Ta.delete(0, END)


def cosine_proximity(x, y):
    return 1 - dot(x, y)


def DoList():
    with device('/cpu:0'):
        model = load_model('speccy_model', compile=False)
        model.compile(optimizer="Adam", loss="cosine_similarity", metrics=["mae"], run_eagerly=True)
    sig1, sample_rate1 = load(Ta.get(first=0, last=1)[0].replace('origen: ', ''), mono=True)
    sig2, sample_rate2 = load(Ta.get(first=0, last=1)[1].replace('deseo: ', ''), mono=True)
    vect1 = getvect(sig1, sample_rate1, model)
    vect2 = getvect(sig2, sample_rate2, model)
    df = read_json('vects.json')
    num_rolas = int(Te.get("1.0", END))
    ve = vect1 / norm(vect1)
    ve2 = vect2 / norm(vect2)
    rolas = HacerLista(ve, ve2, df, numero_de_puntos=num_rolas)
    for x in rolas:
        Ta.insert(END, root + x)
    Hacefolder(rolas)


def Hacefolder(rolas):
    try:
        mkdir('listadereproducion')
    except OSError:
        print('borra el folder')
    for i, x in enumerate(rolas):
        if len(str(i)) < 3:
            i = '0' + str(i)
        if len(str(i)) < 3:
            i = '0' + str(i)
        # shutil.copy(root+x, 'listadereproducion/' + str(i) + ' ' + os.path.basename(x))
        print('listadereproducion/' + str(i) + ' ' + os.path.basename(x))


def HacerLista(seed_vect, deseo_vect, da_frame, numero_de_puntos=10):
    t_vals = linspace(0, 1, numero_de_puntos)
    puntos = geometric_slerp(seed_vect / norm(seed_vect), deseo_vect / norm(deseo_vect), t_vals)
    rolas = []
    n = 1
    for punto in puntos:
        print("punto " + str(n))
        n += 1
        distancias = []
        for vect in da_frame['song_vect'].values:
            distancia = cosine_proximity(vect, punto)
            distancias.append(distancia)
        dis_frame = da_frame.assign(distances=distancias)
        sorted_frame = dis_frame.sort_values(by=['distances'])
        for x in range(0, 200):
            if sorted_frame.iloc[x]['song_path'] not in rolas:
                rolas.append(sorted_frame.iloc[x]['song_path'])
                break
    return rolas


def mp3T0vect(y, sr, Modelo):
    try:
        slice_size = Modelo.layers[0].input_shape[0][2]
        hop_length = 512
        n_fft = 2048
        n_mels = Modelo.layers[0].input_shape[0][1]
        S = melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr / 2)
        x = ndarray(shape=(S.shape[1] // slice_size, n_mels, slice_size, 1), dtype=float)
        for slice in range(S.shape[1] // slice_size):
            log_S = power_to_db(S[:, slice * slice_size: (slice + 1) * slice_size], ref=ma)
            if ma(log_S) - mi(log_S) != 0:
                log_S = (log_S - mi(log_S)) / (ma(log_S) - mi(log_S))
            x[slice, :, :, 0] = log_S
        toreturn = Modelo.predict(x)
    except:
        print("An exception occurred")
        return 0
    return toreturn


def getvect(sig, sample_rate, model):
    return suma(mp3T0vect(sig, sample_rate, model), axis=0)


if __name__ == '__main__':
    window = Tk()
    window.geometry("1000x500")
    button = Button(text='Seleccionar archivo', command=openFile)
    button.pack()
    button3 = Button(text='Limpiar Textbox', command=limpiarTextbox)
    button3.pack()
    button2 = Button(text='Generar lista de reproduccion', command=DoList)
    button2.pack()
    button4 = Button(text='Selecciona Directorio de Musica', command=seleccionarCarpeta)
    button4.pack()
    Te = Text(window, height=1, width=5)
    Te.insert(END, '20')
    Te.pack()
    Ta = Listbox(window, height=200, width=150)
    Ta.pack()
    window.mainloop()
