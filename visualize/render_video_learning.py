import os

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw


def create_video_pil(log_path, map_path, video_path, fps=10):
    positions_log = np.load(log_path, allow_pickle=True)
    map_array = np.load(map_path)

    H, W = map_array.shape
    base_img = np.stack([255 - map_array * 80]*3, axis=-1).astype(np.uint8)

    with imageio.get_writer(video_path, fps=fps) as writer:
        for step, positions in enumerate(positions_log):
            frame_img = base_img.copy()
            for x, y in positions:
                if 0 <= x < H and 0 <= y < W:
                    frame_img[x, y] = [255, 0, 0]

            img = Image.fromarray(frame_img).resize((640, 640), resample=Image.NEAREST)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), f"Step {step}", fill=(0,0,0))

            writer.append_data(np.array(img))
            if step % 100 == 0:
                print(f"Step {step} written.")

    print(f"Saved video to: {video_path}")

if __name__ == "__main__":
    learning_id = "Qlearning2"  # ← ディレクトリ名
    episode_list = [0, 50, 100, 148]     # ← ここで好きなエピソード番号を複数指定

    learning_dir = os.path.join("output", "logs", learning_id)
    config_path = os.path.join(learning_dir, "run_config_used.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    map_path = config["map"]
    fps = 10

    for episode in episode_list:
        log_path = os.path.join(learning_dir, f"episode_{episode}.npy")
        video_path = os.path.join(learning_dir, f"episode_{episode}.mp4")

        if not os.path.exists(log_path):
            print(f"Episode file {log_path} not found, skipping.")
            continue

        print(f"Creating video for episode {episode}...")
        create_video_pil(log_path, map_path, video_path, fps=fps)
