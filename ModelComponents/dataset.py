import os
import random

from torch.utils.data import Dataset
import torchaudio

from Parametrization.mfcc import parametrisation


class FolderAudioDataset(Dataset):
    def __init__(self, root_dir,noise_dir, noise_prob, snr_db=10):
        self.root_dir = root_dir
        self.noise_dir = noise_dir
        self.noise_prob = noise_prob
        self.snr_db = snr_db

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_folder):
                if filename.endswith('.wav'):
                    filepath = os.path.join(cls_folder, filename)
                    self.samples.append((filepath, self.class_to_idx[cls]))

        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(filepath)

        if waveform.size(0) == 2:
            waveform = waveform.mean(dim=0)

        if random.random() < self.noise_prob:

            noise_path = random.choice(self.noise_files)
            noise_waveform, noise_sample_rate = torchaudio.load(noise_path)

            # Convertir en mono en moyennant les deux canaux
            if noise_waveform.size(0) == 2:
                noise_waveform = noise_waveform.mean(dim=0)  # Moyenne des deux canaux pour obtenir un signal mono

            # Calcul du SNR
            signal_power = waveform.norm(p=2)
            noise_power = noise_waveform.norm(p=2)

            snr = 10 ** (self.snr_db / 20)
            scaling = signal_power / (snr * noise_power)

            processed_waveform = waveform + scaling * noise_waveform
        else:
            processed_waveform = waveform

        # Paramétrisation
        mfcc = parametrisation(processed_waveform.flatten())

        return mfcc, label
