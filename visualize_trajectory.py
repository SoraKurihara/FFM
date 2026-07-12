#!/usr/bin/env python3
"""
軌跡データ（.pkl / .npz）からアニメーションを一括で作成するプログラム。
指定フォルダ内の trajectory*.pkl を検出し、同フォルダの .npy をマップとして MP4 化する。
"""

import os
import re
import pickle

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# どのフォルダ内の trajectory*.pkl を MP4 にするか（このパスのみ指定）
SOURCE_DIR = r"output/logs/ffm_ac_aggregated/ob1"
# SOURCE_DIR = r"output/logs/ffm_ac_aggregated/learning_3maps_12_exam_ob1"

# 動画設定
FPS = 10  # フレームレート
RESIZE_SCALE = 640  # リサイズ後のサイズ（正方形）


class CompatibleUnpickler(pickle.Unpickler):
    """numpy 2.x で保存された pickle を読み込むための Unpickler"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        return super().find_class(module, name)


def load_trajectory_pkl(path):
    """trajectory*.pkl（list of (N,2) arrays）を読み込む"""
    with open(path, "rb") as f:
        return CompatibleUnpickler(f).load()


def get_trajectory_pkl_paths(dir_path):
    """dir_path 内で trajectory で始まる .pkl のパスをエピソード順で返す"""
    if not os.path.isdir(dir_path):
        return []
    pattern = re.compile(r"^trajectory_ep(\d+)\.pkl$", re.IGNORECASE)
    candidates = []
    for name in os.listdir(dir_path):
        m = pattern.match(name)
        if m:
            ep = int(m.group(1))
            candidates.append((ep, os.path.join(dir_path, name)))
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


def get_map_npy_path(dir_path):
    """dir_path 内の .npy を1つ返す（マップ用）。見つからなければ None"""
    if not os.path.isdir(dir_path):
        return None
    for name in sorted(os.listdir(dir_path)):
        if name.lower().endswith(".npy"):
            return os.path.join(dir_path, name)
    return None


def create_video_from_trajectory(
    trajectory_path, map_path, video_path, fps=10, resize_scale=640, use_pkl=True
):
    """
    軌跡データから動画を作成

    Args:
        trajectory_path: 軌跡データ（.pkl または .npz）のパス
        map_path: マップデータ（.npy）のパス
        video_path: 出力動画ファイルのパス
        fps: フレームレート
        resize_scale: リサイズ後のサイズ（正方形）
        use_pkl: True のとき .pkl（list of arrays）、False のとき .npz の "positions" を使用
    """
    print(f"軌跡データを読み込み中: {trajectory_path}")
    if use_pkl:
        positions = load_trajectory_pkl(trajectory_path)
        positions = list(positions)
        # ファイル名からエピソード番号を取得（trajectory_ep0010.pkl -> 10）
        basename = os.path.basename(trajectory_path)
        ep_match = re.search(r"ep(\d+)", basename, re.IGNORECASE)
        episode = int(ep_match.group(1)) if ep_match else -1
        N = int(positions[0].shape[0]) if positions else 0
        steps = len(positions)
    else:
        data = np.load(trajectory_path, allow_pickle=True)
        positions = data["positions"]
        episode = int(data.get("episode", -1))
        N = int(data.get("N", -1))
        steps = int(data.get("steps", len(positions)))

    print(f"マップデータを読み込み中: {map_path}")
    map_array = np.load(map_path)

    H, W = map_array.shape
    print(f"マップサイズ: {H} x {W}")
    print(f"総ステップ数: {len(positions)}")
    print(f"エピソード: {episode}, 人数: {N}")
    print("動画を作成中...")

    # マップを画像に変換（0=空き, 1=壁, 3=出口）
    base_img = np.stack([255 - map_array * 80] * 3, axis=-1).astype(np.uint8)
    exit_mask = map_array == 3
    base_img[exit_mask] = [0, 255, 0]

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # imageio-ffmpeg で MP4 出力
    with imageio.get_writer(video_path, fps=fps) as writer:
        for step_idx, step_positions in enumerate(positions):
            frame_img = base_img.copy()
            step_positions = np.asarray(step_positions)
            for agent_pos in step_positions:
                x, y = int(agent_pos[0]), int(agent_pos[1])
                if 0 <= x < H and 0 <= y < W:
                    frame_img[x, y] = [255, 0, 0]
            img = Image.fromarray(frame_img).resize(
                (resize_scale, resize_scale), resample=Image.NEAREST
            )
            draw = ImageDraw.Draw(img)
            for i, line in enumerate([f"Step {step_idx}", f"Episode {episode}", f"N={N}"]):
                draw.text((5, i * 20), line, fill=(255, 255, 255), font=font)
            writer.append_data(np.array(img))
            if (step_idx + 1) % 100 == 0 or step_idx == 0:
                print(f"  Step {step_idx + 1}/{len(positions)} written.")

    print(f"✅ 保存: {video_path}\n")
    return episode, steps


def main():
    source_dir = os.path.normpath(SOURCE_DIR)
    if not os.path.isdir(source_dir):
        print(f"エラー: 指定フォルダが存在しません: {source_dir}")
        return

    pkl_paths = get_trajectory_pkl_paths(source_dir)
    map_path = get_map_npy_path(source_dir)

    if not pkl_paths:
        print(f"エラー: {source_dir} に trajectory_ep*.pkl が見つかりません。")
        return
    if map_path is None:
        print(f"エラー: {source_dir} に .npy マップファイルが見つかりません。")
        return

    print(f"対象フォルダ: {source_dir}")
    print(f"マップ: {map_path}")
    print(f"trajectory pkl 数: {len(pkl_paths)} ({os.path.basename(pkl_paths[0])} ... {os.path.basename(pkl_paths[-1])})\n")

    # episode / steps 一覧を保存
    summary_rows = []

    for trajectory_path in pkl_paths:
        base = os.path.splitext(os.path.basename(trajectory_path))[0]
        video_path = os.path.join(source_dir, base + ".mp4")
        episode, steps = create_video_from_trajectory(
            trajectory_path,
            map_path,
            video_path,
            fps=FPS,
            resize_scale=RESIZE_SCALE,
            use_pkl=True,
        )
        summary_rows.append((episode, steps, os.path.basename(trajectory_path)))

    summary_path = os.path.join(source_dir, "trajectory_episode_steps.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# episode\tsteps\ttrajectory_file\n")
        for episode, steps, filename in summary_rows:
            f.write(f"{episode}\t{steps}\t{filename}\n")

    print(f"完了: {len(pkl_paths)} 本の MP4 を出力しました。")
    print(f"一覧保存: {summary_path}")


if __name__ == "__main__":
    main()
