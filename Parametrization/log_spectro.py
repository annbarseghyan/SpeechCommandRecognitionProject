import torch
import torchaudio


def compute_log_spectrogram(signal, n_fft=400, win_length=400, hop_length=160):

    signal=torch.squeeze(signal)
    if signal.dim() != 1 or signal.size(0) != 16000:
        raise ValueError("Le signal doit être de taille 16000")
    
    # Calcul du spectrogramme
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2
    )(signal)
    
    # Appliquer le logarithme (avec une petite constante pour éviter log(0))
    log_spectrogram = torch.log(spectrogram + 1e-6)
    
    return (log_spectrogram)