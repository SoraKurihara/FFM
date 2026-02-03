#!/usr/bin/env python3
"""
軌跡データ（.npzファイル）の1ステップ目を表示するプログラム
"""

import numpy as np

# 軌跡データファイルのパス
TRAJECTORY_PATH = r"C:\Development\FFM\output\logs\actor_only_training\run_20251129_101006\trajectories\trajectory_N1_ep00100_total00100.npz"


def main():
    # 軌跡データを読み込み
    print(f"軌跡データを読み込み中: {TRAJECTORY_PATH}")
    data = np.load(TRAJECTORY_PATH, allow_pickle=True)

    # ファイル内のキーを表示
    print("\nファイル内のキー:")
    for key in data.keys():
        print(f"  - {key}")

    # 各データの情報を表示
    print("\n各データの情報:")
    for key in data.keys():
        value = data[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value} (type: {type(value).__name__})")

    # 1ステップ目のpositionsを取得
    positions = data["positions"]

    # ステップ数
    num_steps = len(positions)
    print(f"\n総ステップ数: {num_steps}")

    if num_steps == 0:
        print("⚠ 軌跡データが空です")
        return

    # 1ステップ目（インデックス0）を取得
    step_0_positions = positions[0]

    print("\n" + "=" * 80)
    print("1ステップ目のデータ")
    print("=" * 80)
    print(f"ステップ: 0 (最初のステップ)")
    print(f"positionsの形状: {step_0_positions.shape}")
    print(f"positionsのdtype: {step_0_positions.dtype}")
    print(f"\n1ステップ目のエージェント位置:")
    print(step_0_positions)

    # 各エージェントの位置を個別に表示
    num_agents = step_0_positions.shape[0]
    print(f"\nエージェント数: {num_agents}")
    for agent_idx in range(num_agents):
        x, y = step_0_positions[agent_idx]
        print(f"  エージェント {agent_idx}: 位置 = ({x}, {y})")

    # その他のメタデータも表示
    print("\n" + "=" * 80)
    print("メタデータ")
    print("=" * 80)
    if "episode" in data:
        print(f"エピソード番号: {data['episode']}")
    if "N" in data:
        print(f"人数: {data['N']}")
    if "total_episode" in data:
        print(f"総エピソード数: {data['total_episode']}")
    if "steps" in data:
        print(f"実行ステップ数: {data['steps']}")


if __name__ == "__main__":
    main()
