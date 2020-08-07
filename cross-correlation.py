import scipy as sp
from PIL import Image
from imutils import paths
from scipy.signal.signaltools import correlate2d


def correlate(pathimage):
    img = Image.open(pathimage)
    data = img.resize((128, 128))
    data = sp.inner(data, [299, 587, 114]) / 1000.0
    return (data - data.mean()) / data.std()


def get_name(path):
    path_name = path.split("/")
    filenames.append(path_name[-1])
    return filenames


def diff(soccers):
    result = []
    for i in range(len(soccers)):
        diff = soccers[0].max() - soccers[i].max()
        if diff <= 8000:
            result.append('Repeated image: ' + filenames[i])
    return result


def soc(res):
    for i in range(len(res)):
        soccers.append(correlate2d(res[0], res[i], mode='same'))
    return soccers


if __name__ == "__main__":

    images_path = "/home/bater/PycharmProjects/pythonProject2/Test/Test"
    imagePaths = sorted(list(paths.list_images(images_path)))

    res = []
    filenames = []
    soccers = []

    for imagePath in imagePaths:
        filenames = get_name(imagePath)
        im = correlate(imagePath)
        res.append(im)

    soccers = soc(res)
    print(diff(soccers))
