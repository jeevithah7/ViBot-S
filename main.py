import os

from script.visual_odometry.visual_odometry import run_visual_odometry
from script.mapping.mapping1 import create_map
from script.path_planning.path_planning1 import demo_path

def main():
    # Correct path (IMPORTANT)
    DATA_PATH = os.path.join("data", "kitti")

    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset path not found:", DATA_PATH)
        return

    print("🚀 Starting ViBot-S Simulation Pipeline...\n")

    # Step 1: Visual Odometry
    print("📍 Running Visual Odometry...")
    trajectory = run_visual_odometry(DATA_PATH)

    if trajectory is None or len(trajectory) == 0:
        print("❌ No trajectory generated.")
        return

    # Step 2: Mapping
    print("🗺️ Generating Map...")
    create_map(trajectory)

    # Step 3: Path Planning
    print("🧭 Running Path Planning...")
    demo_path()

    print("\n✅ Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()