#!/usr/bin/env python3
"""
軌跡データ（.npzファイル）からアニメーションを作成するプログラム
"""

import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 軌跡データファイルのパス
TRAJECTORY_PATH = r"C:\Development\FFM\output\logs\actor_only_training\run_20251129_101006\trajectories\trajectory_N1_ep10000_total20000.npz"

# マップファイルのパス
MAP_PATH = "data/maps/simple_room.npy"

# 出力動画ファイルのパス（自動生成）
OUTPUT_VIDEO_PATH = None  # Noneの場合は軌跡ファイルと同じディレクトリに保存

# 動画設定
FPS = 10  # フレームレート
RESIZE_SCALE = 640  # リサイズ後のサイズ（正方形）


def create_video_from_trajectory(
    trajectory_path, map_path, video_path, fps=10, resize_scale=640
):
    """
    軌跡データから動画を作成

    Args:
        trajectory_path: 軌跡データ（.npz）のパス
        map_path: マップデータ（.npy）のパス
        video_path: 出力動画ファイルのパス
        fps: フレームレート
        resize_scale: リサイズ後のサイズ（正方形）
    """
    print(f"軌跡データを読み込み中: {trajectory_path}")
    data = np.load(trajectory_path, allow_pickle=True)
    positions = data["positions"]

    print(f"マップデータを読み込み中: {map_path}")
    map_array = np.load(map_path)

    H, W = map_array.shape
    print(f"マップサイズ: {H} x {W}")
    print(f"総ステップ数: {len(positions)}")

    # マップを画像に変換
    # 0=空き, 1=壁, 3=出口
    base_img = np.stack([255 - map_array * 80] * 3, axis=-1).astype(np.uint8)

    # 出口を緑色で表示
    exit_mask = map_array == 3
    base_img[exit_mask] = [0, 255, 0]  # 緑色

    # メタデータを取得
    episode = data.get("episode", -1)
    N = data.get("N", -1)
    total_episode = data.get("total_episode", -1)
    steps = data.get("steps", len(positions))

    print(f"\nメタデータ:")
    print(f"  エピソード番号: {episode}")
    print(f"  人数: {N}")
    print(f"  総エピソード数: {total_episode}")
    print(f"  実行ステップ数: {steps}")
    print(f"\n動画を作成中...")

    with imageio.get_writer(video_path, fps=fps) as writer:
        for step_idx, step_positions in enumerate(positions):
            # フレーム画像を作成
            frame_img = base_img.copy()

            # エージェントの位置を描画（赤色）
            for agent_pos in step_positions:
                x, y = agent_pos
                if 0 <= x < H and 0 <= y < W:
                    frame_img[x, y] = [255, 0, 0]  # 赤色

            # PIL画像に変換してリサイズ
            img = Image.fromarray(frame_img).resize(
                (resize_scale, resize_scale), resample=Image.NEAREST
            )

            # テキストを描画
            draw = ImageDraw.Draw(img)

            # ステップ番号とメタデータを表示
            text_lines = [
                f"Step {step_idx}",
                f"Episode {episode}",
                f"N={N}",
            ]

            # フォントサイズを調整（可能であれば）
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            y_offset = 0
            for line in text_lines:
                draw.text((5, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += 20

            writer.append_data(np.array(img))

            # 進捗表示
            if (step_idx + 1) % 100 == 0 or step_idx == 0:
                print(f"  Step {step_idx + 1}/{len(positions)} written.")

    print(f"\n✅ 動画を保存しました: {video_path}")


def main():
    # 出力パスが指定されていない場合は自動生成
    if OUTPUT_VIDEO_PATH is None:
        trajectory_dir = os.path.dirname(TRAJECTORY_PATH)
        trajectory_filename = os.path.basename(TRAJECTORY_PATH)
        video_filename = os.path.splitext(trajectory_filename)[0] + ".mp4"
        video_path = os.path.join(trajectory_dir, video_filename)
    else:
        video_path = OUTPUT_VIDEO_PATH

    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # 動画を作成
    create_video_from_trajectory(
        TRAJECTORY_PATH,
        MAP_PATH,
        video_path,
        fps=FPS,
        resize_scale=RESIZE_SCALE,
    )


if __name__ == "__main__":
    main()
