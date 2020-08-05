import cv2
import matplotlib.pyplot as plt
from imutils import paths

if __name__ == "__main__":
    """
    Load the image of interest. Find keypoints and descriptors. Search best matches and get predict 
    the image is a duplicate. Show match results. 
    """
    local_path = "/home/bater/PycharmProjects/pythonProject2/Test"
    ext = ".jpg"
    cur_img = "1" + ext

    img1 = cv2.imread(cur_img)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    imagePaths = sorted(list(paths.list_images(local_path)))

    for imagePath in imagePaths:
        img2 = cv2.imread(imagePath)
        gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = surf.detectAndCompute(gray1, None)
        matches = flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.3 * n.distance:
                matchesMask[i] = [1, 0]

        if matchesMask.count([1, 0]) > 10:
            print ('Repeated image: ' + imagePath)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.imshow(img3, ), plt.show()
