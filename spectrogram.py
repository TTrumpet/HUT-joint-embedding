import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa
from transformers import ASTFeatureExtractor

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def make_features(name, mel_bins, target_length=1024, sr=16000):
    waveform, sr = torchaudio.load(name)

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)
    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

def spectrogram(name, mel_bins=128, target_length=1024):
    feature_extractor = ASTFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-0.4593")
    feats = make_features(name, mel_bins=mel_bins)
    #feats = feats.expand(1, target_length, mel_bins)
    return feats
    
