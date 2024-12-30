import torch
import matplotlib.pyplot as plt
from musiclm_pytorch.musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from transformers import BertTokenizer, BertModel, ASTFeatureExtractor, ASTModel
import os
import pandas as pd

from audiolm_pytorch.data import cast_tuple, exists
from audiolm_pytorch.utils import curtail_to_multiple

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
'''
trained_model = ASTModel.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
for part in trained_model.named_parameters():
        print(part[0])
        #print(type(part[1]))
        print(part[1].data.size())

pkg = torch.load(str("./results_128x1024_b32/mulan.990.pt"))
print(type(pkg['model']))
for key, value in pkg['model'].items():
    print(key)
    print(value.size())

exit()
'''
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

# get a ton of <sound, text> pairs and train
df = pd.read_csv("musiccaps-public.csv")

#from datasets import load_dataset
#dataset = load_dataset("agkphysics/AudioSet", "unbalanced")

# list of values
train_dict = []

# load all the .wav -> use feature extractor
from fma_embeddings import tokenize
from musiclm_pytorch.trainer import MuLaNTrainer
from musiccaps_dataset import MusicCapsDataset, MockTextAudioDataset

directory = 'music_data'
input_tdim = 1024
text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
n = 256

'''
dataset = AudioSetDataset(
    file='unbalanced_train_segments.csv',
    folder='audioset_data',
    text_tokenizer=text_tokenizer,
)

print(len(dataset))


exit()
'''
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
'''
mulan_trainer = MuLaNTrainer(
    mulan=mulan,
    dataset=MockTextAudioDataset(),
    batch_size=2,
    num_train_steps = 1
)
'''

model_name = '128x614_b128'

mulan_trainer.load("./results_" + model_name + "/mulan.990.pt")

# how I was wrongly training the mulan model previously
'''
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        try:
            feats = tokenize(f)
        except:
            continue
        #feats_data = feats.reshape(1, input_tdim, 128)
        wav = feats # wav tensor
        yid = filename.replace('.wav', '')
        index = df.index[df['ytid'] == yid]
        for i in index:
            text = df.at[i,'caption'] # text caption
            # put text through transformer <----
            text = text_tokenizer(text, max_length=n, padding='max_length', return_tensors='pt')
        train_dict.append((wav, text.input_ids))

# loop through all wav and texts and compute loss + backward pass
counter = 0
total = len(train_dict)
print(total)
for i in range(total):
    loss = mulan(train_dict[i][0], train_dict[i][1])
    loss.backward()
'''
 # the part after this probably works, just testing the mulan training

# train on just audio corpus by generating semantic embeddings based on wav
#   should be able to generate using semantic transformer
from fma_embeddings import create_embeddings, load_embeddings, find_similarities, find_similarities_testing_set
from fma_crawling.fma_crawl import get_song_title

#create_embeddings('fma_small', 'fma_tensors', 'fma_small/checksums')
#music_dict = load_embeddings('fma_tensors', 'fma_small/checksums')

# rand text equal to user input
# return greatest similarity song between all wav audio latents in FMA

print("testing")
#rand_wav = torch.rand(2, 1024)
#rand_text = torch.randint(0, 20000, (2, 256))

#df = pd.read_csv("testing.csv")
#df = df.to_numpy()

from create_testing_set import create_testing_set
#df = create_testing_set('fma_small', 'fma_metadata/tracks.csv')

df = pd.read_csv('musiccaps-public.csv')
df = df[['ytid', 'caption']]
df=df.to_numpy()

counter = 0
filename = model_name + '_results.txt'
log_file = open(filename, 'w')

return_all = True

#table = []

for i in df:
    print(counter)
    if counter == 100:
        break

    string = i[1].replace('[', '')
    string = string.replace(']', '')
    string = string.replace('\'', '')
    text = string
    '''
    if len(text) > n:
        text = text[:n]
    print(text)
    '''

    text = text_tokenizer(text, max_length=n, padding='max_length', return_tensors='pt').input_ids.to("cuda:0")

    #num = int(i[0] / 1000)
    #num = str(num).zfill(3) + '/' + str(i[0]).zfill(6) + '.wav'
    wav = i[0] + '.wav'
    wav.replace('\'', '')

    #file, top_k, guess_index, actual_sim = find_similarities(text, mulan, num, 'fma_tensors', 'fma_small/checksums', log_file, return_similarities=True)
    find_similarities_testing_set(text, mulan, i[0], 'music_tensors', 'musiccaps-public.csv', file=log_file)

    counter += 1
    if return_all:
        continue
    
    file = file.replace('/', '')
    file = file.replace('.wav','')
    track_id = int(file)

    num = num.replace('/', '')
    num = num.replace('.wav', '')

    print("true index: " + str(i[0]), file=log_file)
    print("true title: " + str(i[2]), file=log_file)

    song_title = get_song_title(track_id)
    print("guessed: " + str(track_id).zfill(9) + " actual: " + num)
    if track_id == i[0]:
        print(" - guessed correctly ")
    else:
        print(" - guessed incorrectly ")
    try:
        print(song_title)
    except:
        print('None')
        
    #table.append((i[2], i[0], actual_sim, guess_index, top_k))

log_file.close()
#table = pd.DataFrame(table, columns=['true_title', 'true_index', 'true_similarity', 'guess_index', 'guess_similarity'])
#results_file = 'results_table.txt'
#results = open(results_file, 'w')
#table.to_csv(results)

'''
sims = torch.cat([mulan(texts = rand_text, wavs = music[0], return_similarities = True) for music in music_dict], dim = 0)
top_matching_index = sims.topk(1, dim = 0).indices.item()
print(music_dict[top_matching_index][1]) # get id
'''