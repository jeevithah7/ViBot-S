import cv2
import os

def run_feature_detection(data_path):
    images = sorted(os.listdir(data_path))
    img = cv2.imread(os.path.join(data_path, images[0]), 0)

    orb = cv2.ORB_create(2000)
    kp, _ = orb.detectAndCompute(img, None)

    img_kp = cv2.drawKeypoints(img, kp, None)

    cv2.imshow("Feature Detection", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_feature_detection("data\kitti")