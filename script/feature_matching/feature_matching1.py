import cv2
import os

def run_feature_matching(data_path):
    images = sorted(os.listdir(data_path))
    images = [os.path.join(data_path, img) for img in images]

    if len(images) < 2:
        print("Not enough images for matching")
        return

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print("Running Feature Matching... Press 'q' to exit")

    for i in range(len(images) - 1):
        img1 = cv2.imread(images[i], 0)
        img2 = cv2.imread(images[i+1], 0)

        if img1 is None or img2 is None:
            print(f"Skipping invalid image at index {i}")
            continue

        # Detect features
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        # Match features
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep best matches
        good_matches = matches[:50]

        # Draw matches
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Resize for display
        match_img = cv2.resize(match_img, (900, 500))

        # Add text info
        cv2.putText(
            match_img,
            f"Frame: {i} -> {i+1}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            match_img,
            f"Matches: {len(good_matches)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Feature Matching", match_img)

        key = cv2.waitKey(200)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_feature_matching("Simulation/data/kitti")