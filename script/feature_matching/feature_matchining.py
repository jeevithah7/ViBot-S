import cv2
import os

def run_feature_matching(data_path):
    images = sorted(os.listdir(data_path))
    images = [os.path.join(data_path, img) for img in images]

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(images) - 1):
        img1 = cv2.imread(images[i], 0)
        img2 = cv2.imread(images[i+1], 0)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

        cv2.imshow("Feature Matching", match_img)

        if cv2.waitKey(200) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_feature_matching("Simulation\data\kitti")