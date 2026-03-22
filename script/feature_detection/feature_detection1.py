import cv2
import os

def run_feature_detection(data_path):
    images = sorted(os.listdir(data_path))
    
    if len(images) == 0:
        print("No images found!")
        return

    orb = cv2.ORB_create(2000)

    print("Running Feature Detection... Press 'q' to exit")

    for img_name in images:
        img_path = os.path.join(data_path, img_name)

        img = cv2.imread(img_path, 0)

        if img is None:
            print(f"Skipping invalid image: {img_name}")
            continue

        # Detect features
        kp, _ = orb.detectAndCompute(img, None)

        # Draw keypoints
        img_kp = cv2.drawKeypoints(
            img, kp, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Resize for better viewing
        img_kp = cv2.resize(img_kp, (800, 500))

        # Display keypoint count
        cv2.putText(
            img_kp,
            f"Keypoints: {len(kp)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Feature Detection", img_kp)

        # Wait (adjust speed here)
        key = cv2.waitKey(200)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_feature_detection("Simulation/data/kitti")