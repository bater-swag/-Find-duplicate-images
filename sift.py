import cv2
from imutils import paths


def grays(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


if __name__ == "__main__":
    """
    Load the image of interest. Find keypoints and descriptors. Search best matches and get predict 
    the image is a duplicate.
    """

    local_path = "/home/bater/PycharmProjects/pythonProject2/Test"
    ext = ".jpg"
    cur_img = "1" + ext

    img1 = cv2.imread(cur_img)
    gray = grays(img1)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray, None)

    imagePaths = sorted(list(paths.list_images(local_path)))

    for imagePath in imagePaths:
           img2 = cv2.imread(imagePath)
           gray1 = grays(img2)
           kp2, des2 = sift.detectAndCompute(gray1, None)
           bf = cv2.BFMatcher()
           matches = bf.knnMatch(des1, des2, k=2)
           good = []
           for m, n in matches:
               if m.distance < 0.3 * n.distance:
                   good.append([m])
           if len(good) > 10:   # werewr
               print('Repeated image' + imagePath)


