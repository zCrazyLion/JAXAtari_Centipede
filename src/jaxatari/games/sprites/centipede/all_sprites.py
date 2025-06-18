import numpy as np

# SPIDER
# Sprite initialisieren (RGBA: 6 Zeilen, 8 Spalten, 4 Farbkanäle)
sprite = np.zeros((6, 8, 4), dtype=np.uint8)
spect = [255, 255, 255, 255]

# Pixel setzen (weiß)
sprite[1, 0] = spect
sprite[0, 1] = spect
sprite[1, 2] = spect
sprite[1, 5] = spect
sprite[0, 6] = spect
sprite[1, 7] = spect
sprite[2, 5] = spect
sprite[2, 4] = spect
sprite[2, 3] = spect
sprite[2, 2] = spect
sprite[3, 5] = spect
sprite[3, 4] = spect
sprite[3, 3] = spect
sprite[3, 2] = spect
sprite[4, 5] = spect
sprite[4, 4] = spect
sprite[4, 3] = spect
sprite[4, 2] = spect
sprite[4, 1] = spect
sprite[4, 6] = spect
sprite[5, 0] = spect
sprite[5, 7] = spect
sprite[5, 4] = spect
sprite[5, 3] = spect

# FLEA
# Sprite initialisieren (RGBA: 6 Zeilen, 8 Spalten, 4 Farbkanäle)
sprite = np.zeros((6, 5, 4), dtype=np.uint8)
spect = [255, 255, 255, 255]

# Pixel setzen (weiß)
sprite[2, 0] = spect
sprite[3, 0] = spect
sprite[4, 1] = spect
sprite[5, 2] = spect
sprite[4, 3] = spect
sprite[5, 4] = spect
sprite[3, 3] = spect
sprite[3, 2] = spect
sprite[3, 1] = spect
sprite[1, 1] = spect
sprite[1, 2] = spect
sprite[0, 2] = spect
sprite[0, 1] = spect
sprite[1, 3] = spect
sprite[2, 3] = spect
sprite[2, 2] = spect