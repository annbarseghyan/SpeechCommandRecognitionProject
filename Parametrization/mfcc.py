from math import floor

import torch


def parametrisation(signal, taille_fenetre=480, nbe_coef=40):
    # Calcul du recouvrement en échantillons
    recouvrement = 160  # 10 ms -> 160 échantillons
    nb_fen = floor((signal.size(0) - taille_fenetre) / recouvrement) + 1  # Calcul du nombre de fenêtres

    # Initialisation du tableau pour stocker les MFCC
    mfcc = torch.zeros((nb_fen, nbe_coef))

    # Fenêtre de Hamming
    hamming_window = torch.hamming_window(taille_fenetre, periodic=False, dtype=signal.dtype, device=signal.device)

    # Découpage du signal en fenêtres
    for fen in range(nb_fen):
        p = fen * recouvrement
        frame = signal[p:p + taille_fenetre]

        # Application de la fenêtre de Hamming
        windowed = frame * hamming_window

        # Calcul du spectre via FFT
        spectre = torch.abs(torch.fft.fft(windowed))

        # Clamping pour éviter des valeurs nulles ou proches de zéro
        spectre = torch.clamp(spectre, min=1e-6)

        # Calcul du cepstre
        cepstre = torch.fft.fft(torch.log(spectre))

        # Extraction des coefficients cepstraux (MFCC)
        cc = cepstre[1:nbe_coef + 1].real
        mfcc[fen, :] = cc

    return mfcc
