import torch
import matplotlib.pyplot as plt
from musiclm_pytorch.musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from transformers import BertTokenizer, BertModel, ASTFeatureExtractor, ASTModel
import os
import pandas as pd

from audiolm_pytorch.data import cast_tuple, exists
from audiolm_pytorch.utils import curtail_to_multiple

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import wandb

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 128,
    spec_n_fft = 254, # 254 -> 128 = F
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8,
    accept_spec=False
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

from fma_embeddings import tokenize
from musiclm_pytorch.trainer import MuLaNTrainer
from musiccaps_dataset import MusicCapsDataset, MockTextAudioDataset

directory = 'music_data'
input_tdim = 1024
text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
n = 256

mulan_trainer = MuLaNTrainer(
    mulan=mulan,
    dataset=MusicCapsDataset(
        file='musiccaps-public.csv',
        folder='music_data',
        text_tokenizer=text_tokenizer,
    ),
    batch_size=128, # 256 CUDA can't handle
    num_train_steps = 1000
)

from fma_embeddings import create_embeddings, load_embeddings, find_similarities, find_similarities_testing_set
from fma_crawling.fma_crawl import get_song_title
from create_testing_set import create_testing_set

model_name = '128x614_b128'
mulan_trainer.load("./results_" + model_name + "/mulan.990.pt")

def create_latents():
    music_dict = load_embeddings('fma_tensors', 'fma_small/checksums')
    test_set = create_testing_set('fma_small', 'fma_metadata/tracks.csv', genre=True)

    os.makedirs(model_name + "_latents", exist_ok=True)

    print("getting latents")
    all_titles = []
    for i in test_set:
        text = i[2].replace('[', '')
        text = text.replace(']', '')
        text = text.replace('\'', '')
        text = text_tokenizer(text, max_length=n, padding='max_length', return_tensors='pt').input_ids.to("cuda:0")
        if len(text) > n:
            text = text[:n]

        num = int(i[0]/1000)
        wav_file = str(num).zfill(3) + '/' + str(i[0]).zfill(6) + '.wav'
        #print('finding: ' + wav_file)

        for j in music_dict:
            #print(' - ' + j[1])
            if j[1] == wav_file:
                wav = j[0]
                wav = torch.unsqueeze(wav, dim=0).to("cuda:0")
        
        try: 
            audio_l, text_l = mulan(texts=text, wavs=wav, return_latents=True)
        except:
            continue
        title = i[3].replace('\'', '')
        title = title.replace('/', '_') # song cannot have titles with '/'
        title = title.replace('?', '_') # song cannot have titles with '?'
        title = title.replace('"', '_') # song cannot have titles with '?'
        #print(title)
        try:
            torch.save(audio_l, model_name + '_latents/' + title + '_audio.pt') # audio
            torch.save(text_l, model_name + '_latents/' + title + '_text.pt') # text
        except:
            print("cannot save: " + title)
        genre = i[1].replace('\'', '')
        all_titles.append([title, genre])
    #print(all_titles)
    df = pd.DataFrame(all_titles, columns=['title', 'genre'])
    df.to_csv(model_name + '_latents_song_titles_genre.csv')

if __name__ == '__main__':

    #create_latents()

    #wandb.init(project="HUT-embeddings")
    df = pd.read_csv(model_name + '_latents_song_titles_genre.csv', index_col=0)
    all_cols = []
    col = []
    audio_col = []
    text_col = []
    genre_col = []
    #print(df)

    for i in df['title']:
        filename = model_name + '_latents/' + i
        index = df.index[df['title'] == i]
        index = index[0]
        genre = df.at[index, 'genre']
        genre_col.append(genre)
        audio_file = filename + '_audio.pt'
        text_file = filename + '_text.pt'
        audio = torch.load(audio_file).tolist()
        text = torch.load(text_file).tolist()
        audio_col.append(audio[0])
        text_col.append(text[0])
        col = [i] + [genre] + [audio[0]] + [text[0]]
        #print(col)
        #print(len(col))
        all_cols.append(col)
        #print(all_cols)
        #plt.plot(audio, text)

    columns = ['title', 'genre']
    for i in range(128):
        columns.append("a" + str(i))
    for i in range(128):
        columns.append("t" + str(i))

    audio_or_text_col = ['title']
    for i in range(128):
        audio_or_text_col.append(str(i))

    audio_scaling = StandardScaler()
    audio_scaling.fit(audio_col)
    audio_scaled_data = audio_scaling.transform(audio_col)

    text_scaling = StandardScaler()
    text_scaling.fit(text_col)
    text_scaled_data = text_scaling.transform(text_col)

    principal=PCA(n_components=1)
    principal.fit(audio_scaled_data)
    y_scaled=principal.transform(audio_scaled_data)
    principal.fit(audio_col)
    y=principal.transform(audio_col)
    
    principal.fit(text_scaled_data)
    x_scaled=principal.transform(text_scaled_data)
    principal.fit(text_col)
    x=principal.transform(text_col)

    le = LabelEncoder()
    le.fit(genre_col)
    genres = list(le.classes_)
    genre_color = le.transform(genre_col)

    #print(genres)
    audio_list_scaled = list(map(lambda x: [x], genres))
    audio_list = list(map(lambda x: [x], genres))
    text_list_scaled = list(map(lambda x: [x], genres))
    text_list = list(map(lambda x: [x], genres))
    #print(genres)
    for i in range(len(genre_color)):
        genre = genre_color[i]
        audio_scaled = y_scaled[i][0]
        audio = y[i][0]
        text_scaled = x_scaled[i][0]
        text = x[i][0]
        audio_list[genre].append(audio)
        audio_list_scaled[genre].append(audio_scaled)
        text_list[genre].append(text)
        text_list_scaled[genre].append(text_scaled)

    genre_labels = genres
    counter = 0
    for i in audio_list:
        print(i[0] + ": " + str(len(i)))
        i.pop(0)
        genre_labels[counter] = genre_labels[counter] + ' (count: ' + str(len(i)) + ')'
        counter += 1
    for i in text_list:
        i.pop(0)
    for i in audio_list_scaled:
        i.pop(0)
    for i in text_list_scaled:
        i.pop(0)
    #print(audio_list)
    #print(text_list)
    #print(genre_labels)

    #fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    
    ax0.hist(audio_list_scaled, histtype='bar', stacked=True)
    #ax0.legend(labels=genres)
    ax0.set_title('audio')
    handles, labels = ax0.get_legend_handles_labels()

    #ax2.hist(audio_list_scaled, histtype='bar', stacked=True)
    #ax2.set_title('audio scaled')

    ax1.hist(text_list_scaled, histtype='bar', stacked=True)
    ax1.set_title('text')

    #ax3.hist(text_list_scaled, histtype='bar', stacked=True)
    #ax3.set_title('text scaled')

    fig.legend(handles, labels=genres)
    fig.supylabel('frequency')
    fig.supxlabel('embedding value')
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    plt.show()
    '''
    plt.figure(figsize=(100,100))
    scatter = plt.scatter(x, y, label=genre_col, c=genre_color)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, genres)
    #plt.legend(*scatter.legend_elements())
    plt.ylabel('audio')
    plt.xlabel('text')
    plt.show()
    '''

'''
    wandb.log({
        "Joint Embedding Space": wandb.Table(
            columns=['title', 'genre', 'audio', 'text'],
            data=all_cols
        ),
        "audio": wandb.Table(
        columns = audio_or_text_col,
            data = audio_col
        ),
        "text": wandb.Table(
            columns = audio_or_text_col,
            data = text_col
        ),
        "genres": wandb.Table(
            columns = ['title', 'genre'],
            data = genre_col
        )
    })
        
    wandb.finish()
    
    #df['audio'] = audio_col
    #df['text'] = text_col
'''