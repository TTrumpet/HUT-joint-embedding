import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from fma_embeddings import tokenize
import string

from spectrogram import spectrogram

def str_to_list(orig_string):
    for punct in string.punctuation:
        orig_string = orig_string.replace(punct, '')
    new_string = orig_string.split(' ')
    #new_string = np.array(new_string)
    return new_string

class MusicCapsDataset(Dataset):
    def __init__(self, file, folder, text_tokenizer, max_audio_length=1024, max_text_length=256, padding='max_length', return_tensors='pt'):
        super().__init__
        self.file = file
        self.folder = folder
        self.text_tokenizer = text_tokenizer
        self.files = os.listdir(folder)
        self.softmax = torch.nn.Softmax(dim=0)

        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.padding = padding
        self.return_tensors = return_tensors
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.file)

        f = os.path.join(self.folder, self.files[idx])
        wav = tokenize(f)
        #wav = spectrogram(f)
        yid = self.files[idx].replace('.wav', '')
        index = df.index[df['ytid'] == yid]
        for i in index:
            raw_text = df.at[i, 'caption']
            text = self.text_tokenizer(raw_text, max_length=self.max_text_length, padding=self.padding, return_tensors=self.return_tensors).input_ids.long()
            text = torch.squeeze(text)
            #raw_text = str_to_list(raw_text)
        
        #return wav, raw_text, text
        #return text, wav
        return text, wav

class AudioSetDataset(Dataset):
    def __init__(self, file, folder, text_tokenizer, max_audio_length=1024, max_text_length=256, padding='max_length', return_tensors='pt'):
        super().__init__
        self.file = file
        self.folder = folder
        self.text_tokenizer = text_tokenizer
        self.files = os.listdir(folder)
        self.softmax = torch.nn.Softmax(dim=0)

        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.padding = padding
        self.return_tensors = return_tensors
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.file)

        f = os.path.join(self.folder, self.files[idx])
        wav = tokenize(f)
        #wav = spectrogram(f)
        yid = self.files[idx].replace('.wav', '')
        index = df.index[df['ytid'] == yid]
        for i in index:
            raw_text = df.at[i, 'caption']
            text = self.text_tokenizer(raw_text, max_length=self.max_text_length, padding=self.padding, return_tensors=self.return_tensors).input_ids.long()
            text = torch.squeeze(text)
            #raw_text = str_to_list(raw_text)
        
        #return wav, raw_text, text
        #return text, wav
        return text, wav

class MockTextAudioDataset(Dataset):
    def __init__(self, length = 100, audio_length = 320 * 32):
        super().__init__()
        self.audio_length = audio_length
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        from random import randrange
        mock_audio = torch.randn(randrange(self.audio_length // 2, self.audio_length))
        mock_text = torch.randint(0, 12, (256,)).long()
        return mock_text, mock_audio