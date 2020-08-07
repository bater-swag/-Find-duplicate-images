import cv2
import numpy as np
from imutils import paths


def prepare_image(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def get_name(path):
    path = path
    path_name = path.split("/")
    return path_name[-1]


if __name__ == "__main__":

    images_path = "/home/bater/PycharmProjects/pythonProject2/Test/Test"
    imagePaths = sorted(list(paths.list_images(images_path)))

    ext = ".jpg"
    cur_img = "3" + ext

    img = prepare_image(cur_img)

    for imagePath in imagePaths:
        img2 = prepare_image(imagePath)
        euclidean = np.sqrt(np.sum((img - img2) ^ 2)) / img.size
        manhattan = np.sum(np.abs(img - img2)) / img.size
        if euclidean < 0.04:
            print ('Repeated image: ' + get_name(imagePath))
        if manhattan < 100:
            print ('Repeated image: ' + get_name(imagePath))
