import os
import torch
import torchaudio
from torchaudio.functional import resample
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys

from einops import rearrange, reduce
from read_log import make_plot

def tokenize(file, max_length=12400, max_target_sample_hz=16000):
    data, sr = torchaudio.load(file)
    sample_hz = sr

    if data.shape[0] > 1:
        data = reduce(data, 'c ... -> 1 ...', 'mean')
    
    # first resample data to the max target freq

    data = resample(data, sample_hz, max_target_sample_hz)
    sample_hz = max_target_sample_hz

    # then curtail or pad the audio depending on the max length

    audio_length = data.size(1)

    if audio_length > max_length:
        max_start = audio_length - max_length
        start = torch.randint(0, max_start, (1, ))
        data = data[:, start:start + max_length]
    else:
        data = F.pad(data, (0, max_length - audio_length), 'constant')

    data = rearrange(data, '1 ... -> ...')

    return data

def create_embeddings(directory, save_path, checksums=None, csv=None):
    # compute audio embedding for FMA dataset
    df = None

    if not checksums == None:
        df = pd.read_csv(checksums, sep="  ", header=None, names=['id', 'label'])
    if not csv == None:
        df = pd.read_csv(csv)
        df = df[['ytid']]
        df=df.rename(columns={'ytid':'id'})
        df['label'] = df['id'] + '.wav'

    for root, dirs, files in os.walk(directory):
        path = root.split(os.sep)
        for file in files:
            f = os.path.join(root, file)
            if os.path.isfile(f):
                try:
                    feats = tokenize(f)
                except:
                    continue
                wav = feats # wav tensor
                label = f.replace(directory+'\\', '')
                label = label.replace('\\', '/')
                index = df.index[df['label']==label]
                text = 'default'
                for i in index:
                    text = df.at[i, 'id']
                torch.save(wav, save_path + '/' + text + '.pt')
                #music_dict.append((wav, label))

#create_embeddings('music_data', 'music_tensors', csv='musiccaps-public.csv')

def load_embeddings(path, checksums=None, csv=None):
    music_dict = []

    df = None

    if not checksums == None:
        df = pd.read_csv(checksums, sep="  ", header=None, names=['id', 'label'])
    if not csv == None:
        df = pd.read_csv(csv)
        df = df[['ytid']]
        df=df.rename(columns={'ytid':'id'})
        df['label'] = df['id'] + '.wav'

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            try:
                wav = torch.load(f)
            except:
                continue
            wav_id = f.replace('.pt', '')
            wav_id = wav_id.replace(path+'\\', '')
            index = df.index[df['id']==wav_id]
            label='default'
            for i in index:
                label = df.at[i, 'label']
            music_dict.append((wav, label))
    
    return music_dict

def find_similarities_testing_set(text, mulan, true_id, path, csv, file=sys.stdout):
    sims = []
    true_count = 0
    true_index = 0

    df = pd.read_csv(csv)


    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        #print(f)
        if os.path.isfile(f):
            try:
                wav = torch.load(f)
            except:
                continue
            wav_id = f.replace('.pt', '')
            wav_id = wav_id.replace(path + '\\', '')
            #print(wav_id)
            index = df.index[df['ytid']==wav_id][0]
            if wav_id == true_id:
                true_index = index
            #print('here')
            wav = torch.unsqueeze(wav, dim=0).to("cuda:0")
            sims.append(mulan(texts=text, wavs=wav, return_similarities=True).item())
    reverse_sims = sorted(sims, reverse=True)
    top_k = str(reverse_sims[0])
    guess_index = sims.index(reverse_sims[0])
    if guess_index == true_index:
        true_count += 1
    true_k = sims[true_index]
    print("placement of true song: " + str(true_index) + ' k_value: ' + str(top_k), file=file)
    print("placement of guessed song: " + str(guess_index) + ' k_value: ' + str(true_k), file=file)
    print("num true: " + str(true_count), file=file)

def find_similarities(text, mulan, actual_str=None, path=None, checksums=None, file=sys.stdout, return_all=False, actual_guess=None, model_name=None, return_similarities=False):
    music_dict = []
    sims = [] 
    tracks_counter = 0

    if not path == None:
        if not checksums == None:
            df = pd.read_csv(checksums, sep="  ", header=None, names=['id', 'label'], engine='python')

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            tracks_counter += 1
            try:
                wav = torch.load(f)
            except:
                continue
            wav_id = f.replace('.pt', '')
            wav_id = wav_id.replace('fma_tensors\\', '')
            index = df.index[df['id']==wav_id]
            label='default'
            for i in index:
                label = df.at[i, 'label']
            music_dict.append(label)
            wav = torch.unsqueeze(wav, dim=0).to("cuda:0")
            #print(wav.size())
            sims.append(mulan(texts=text, wavs=wav, return_similarities=return_similarities).item())
    
    #print(sims)
    #top_matching_index = sims.topk(1, dim=0).indices.item()
    #print("sims lengths: " + str(len(sims)))
    #print("num tracks: " + str(tracks_counter))
    reverse_sims = sorted(sims, reverse=True)
    top_k = str(reverse_sims[0])
    if return_all == True:
        #print(reverse_sims)
        make_plot(reverse_sims, model_name, actual_guess)
        return
    print("top k score: " + top_k, file=file)
    top_matching_index = sims.index(reverse_sims[0])
    guess_index = str(top_matching_index)
    print("guess index: " + guess_index, file=file)
    #print("actual similarity: " + actual_similarity(actual_str, text, mulan, checksums))

    # finding actual similarity
    actual_label = 'default'
    actual_index = df.index[df['label']==actual_str]
    for i in actual_index:
        actual_label = df.at[i, 'id']
    try:
        a_wav = torch.load('fma_tensors/'+actual_label+'.pt')
        a_wav = torch.unsqueeze(a_wav, dim=0).to("cuda:0")
        actual_sim = str(mulan(texts=text, wavs=a_wav, return_similarities=True).item())
        print("actual similarity: " + actual_sim, file=file)
    except:
        print("not in dataset")

    return music_dict[top_matching_index], top_k, guess_index, actual_sim

def convert_path_to_tensor(path, checksums, tensor_path='/'):
    df = pd.read_csv(checksums, sep="  ", header=None, names=['id', 'label'])
    return

def actual_similarity(actual_song_path, text, mulan, checksums):
    # take path to the actual song supposed to be recommended
    # return the k score similarity between text and the actual song
    print(actual_song_path)
    wav = torch.load(convert_path_to_tensor(actual_song_path), checksums, 'fma_tensors')
    return mulan(texts=text, wavs=wav, return_similarities=True).item()