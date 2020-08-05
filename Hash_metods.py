import numpy
import scipy.fftpack
from imutils import paths
from sklearn import linear_model
from PIL import Image, ImageOps


class Hash:
    """
    The class contains methods: AHash, DHash, PHash.
    """

    def __init__(self, image):
        self.image = Image.open(image)

    def ahash(self, img=None):
        """
        1. Decreasing the image size.
        2. Image grey-scaling.
        3. Computing the average value.
        4. Simplifying the image. Every pixel gives a value of 0 if it is less than the average value and it gives
        a value of 1 when its value is greater than average.
        """
        im = img or self.image
        size = 16, 16  # most popular size

        im = im.resize(size, Image.ANTIALIAS)
        im = im.convert('L')

        pixels = list(im.getdata())
        average = sum(pixels) / len(pixels)
        result = ''
        for pixel in pixels:
            if pixel > average:
                result += '1'
            else:
                result += '0'
        return result

    def dhash(self, img=None):
        """
        1. Decreasing the image size. However, in this algorithm we use is not square but a rectangular matrix of
        an image with a size of 9x8 (mostly N + 1 x N)
        2. Image grey-scaling.
        3. Simplifying the image. If a value of a current pixel is less than the previous one pixels[i] > pixels[i - 1],
         the value of the hash is 1 or otherwise it is 0.
        """

        im = img or self.image
        size = 9, 8  # size...

        im = im.resize(size, Image.ANTIALIAS)
        im = im.convert('L')

        pixels = list(im.getdata())
        result = ''
        i = 0
        for i in range(len(pixels)):
            if pixels[i] != len(pixels):
                if pixels[i] > pixels[i - 1]:
                    result += '1'
                else:
                    result += '0'
        return result

    def phash(self, img=None):
        """
        1. Decreasing the image size. the larger image size is chosen (32x32) not to delete the high frequencies
        (this will happen later) but to simplify the DCT algorithm.
        2. Image grey-scaling.
        3. Then, one needs to greyscale the image and perform the DCT-transformation, which breaks the image into
        the basic of frequencies.
        4. Computing the average value.
        5. The next steps are the same as for aHash: the value of obtained matrix is converted to the values 1 or 0
        depending whether the pixel has a lesser or bigger value than an average.
        """
        im = img or self.image

        hash_size = 32
        im = im.resize((hash_size, hash_size), Image.ANTIALIAS)
        im = im.convert('L')

        pixels = list(im.getdata())
        dct = scipy.fftpack.dct(pixels)
        smalldct = dct[:32]
        med = numpy.median(smalldct)
        diff = smalldct > med
        return diff

    def prepare_image(self, crop_width_perc=0, crop_height_perc=0, fit_image=True):
        result = self.image
        result = result.convert('L')

        image_size = result.size
        width_crop_size = int(image_size[0] * crop_width_perc / 2) if crop_width_perc > 0 else 0
        height_crop_size = int(image_size[1] * crop_height_perc / 2) if crop_height_perc > 0 else 0
        if width_crop_size or height_crop_size:
            result = result.crop(
                (
                    width_crop_size,
                    height_crop_size,
                    image_size[0] - width_crop_size,
                    image_size[1] - height_crop_size
                )
            )

        resize_option = Image.ANTIALIAS
        if fit_image:
            return ImageOps.fit(result, (128, 128), resize_option)

        return result.resize((128, 128), resize_option)

    def calc_scores(self):
        alg = (
            ('crop', 0, 0, 8, True),  # original fitted
            ('crop', 0, 0.1, 8, True),  # vertical 10% crop fitted
            ('crop', 0.1, 0, 8, True),  # horizontal 10% crop fitted
            ('crop', 0.1, 0.1, 8, True),  # vertical and horizontal 10% crop fitted

            ('crop', 0, 0, 8, False),  # original resized
            ('crop', 0, 0.1, 8, False),  # vertical 10% crop resized
            ('crop', 0.1, 0, 8, False),  # horizontal 10% crop resized
            ('crop', 0.1, 0.1, 8, False)  # vertical and horizontal 10% crop resized
        )
        scores = []
        for item in alg:
            if item[0] == 'crop':
                v, h, hash_size, fit_image = item[1:]
                name = '%s_%s_%s_%s_%s' % item
                value = self.ahash(
                    self.prepare_image(
                        crop_width_perc=v,
                        crop_height_perc=h,
                        fit_image=fit_image
                    )
                )
                scores.append((name, value))
        return scores

    @staticmethod
    def calc_difference(h1, h2):
        return sum(ch1 != ch2 for ch1, ch2 in zip(h1, h2))

    @staticmethod
    def predict(vector):
        coefs = numpy.array(
            [
                [
                    0.30346249,
                    -0.33800637,
                    -0.30347395,
                    -0.33800637,
                    0.05190433,
                    -0.20001436,
                    0.07453074,
                    0.29136006
                ]
            ]
        )
        classifier = linear_model.LogisticRegression(solver='lbfgs', random_state=0)
        classifier.classes_ = numpy.array((0, 1))
        classifier.coef_ = coefs
        classifier.intercept_ = numpy.array([1.98375232])
        result = classifier.predict_proba(numpy.reshape(numpy.array(vector), (1, 8)))
        match = result[:, 1] > result[:, 0]
        return match[0]


if __name__ == "__main__":
    """
    Write the path to the search folder. Perform transformation on the given image. 
    Go through all the images in the folder and perform the appropriate transformations and predict 
    if the image is a duplicate.
    """

    images_path = "/Users/Maslov/PycharmProjects/untitled/folder"
    ext = ".jpg"
    cur_img = "2" + ext

    img_hash = Hash(cur_img)
    hash_image = Hash.ahash(img_hash)
    cur_img_score = Hash.calc_scores(img_hash)

    imagePaths = sorted(list(paths.list_images(images_path)))

    for imagePath in imagePaths:

        image = Hash(imagePath)
        image_hash = Hash.ahash(image)
        img_score = Hash.calc_scores(image)

        vector = []
        for h1, h2 in zip(cur_img_score, img_score):
            vector.append(Hash.calc_difference(h1[1], h2[1]))

        res = Hash.predict(vector)
        if res:
            print('Repeated image: ' + imagePath)
        else:
            print('Not repeat image: ' + imagePath)
