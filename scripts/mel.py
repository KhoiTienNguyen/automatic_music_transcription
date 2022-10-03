import librosa
import numpy
import skimage.io

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
    # img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)
    return img.shape[1]


def convert(path, path_out):
    # settings
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 2048 # number of time-steps. Width of image

    # load audio. Using example from librosa
    # path = '000051652-1_2_1.wav'
    y, sr = librosa.load(path, sr=22050)
    out = f'{path_out}/{(path.split(".")[-2]).split("/")[-1]}.png'

    # extract a fixed length window
    start_sample = 0 # starting at beginning
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples]
    
    # convert to PNG
    width = spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    print('wrote file', out)
    return width

if __name__ == "__main__":
    path = 'cello.wav'
    convert(path, '.')