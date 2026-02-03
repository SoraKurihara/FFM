#!/usr/bin/env python3
"""
CSVファイルから特定のエピソード範囲のデータを取得し、
Nを横軸、stepを縦軸としたグラフを作成するプログラム
"""

import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pathlib import Path

# CSVファイルのパス
CSV_PATH = (
    r"C:\Development\FFM\output\logs\unified_actor_training"
    r"\run_20260119_070834\steps_per_episode.csv"
)

# 出力ディレクトリ（CSVと同じディレクトリ）
OUTPUT_DIR = Path(CSV_PATH).parent


def main():
    # CSVファイルを読み込む
    print(f"CSVファイルを読み込み中: {CSV_PATH}")

    data = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'episode_num': int(row['episode_num']),
                'N': int(row['N']),
                'steps': int(row['steps'])
            })

    print(f"総データ数: {len(data)}")

    # データを格納するリスト
    n_values = []
    step_values = []
    n_to_steps = {}  # N値ごとのstepsリスト
    range_info = {}  # エピソード範囲ごとの情報

    # エピソード範囲ごとにデータを取得
    # 範囲1: 950~1000, 範囲2: 1950~2000, 範囲3: 2950~3000, ...
    max_episode = max(row['episode_num'] for row in data)
    num_ranges = (max_episode - 950) // 1000 + 1
    print(f"\n最大エピソード数: {max_episode}")
    print(f"処理する範囲数: {num_ranges}")

    for range_idx in range(1, num_ranges + 1):
        episode_start = 950 + (range_idx - 1) * 1000
        episode_end = 1000 + (range_idx - 1) * 1000

        # 該当するエピソード範囲のデータを取得
        filtered_data = [
            row for row in data
            if episode_start <= row['episode_num'] <= episode_end
        ]

        if len(filtered_data) > 0:
            # この範囲内のNの値を確認（通常は1つの値のみ）
            n_values_in_range = set(row['N'] for row in filtered_data)
            print(
                f"\n範囲{range_idx}: episode {episode_start}~{episode_end} "
                f"のデータ数: {len(filtered_data)}, "
                f"含まれるN値: {sorted(n_values_in_range)}"
            )

            # 各Nの値ごとにデータを集計
            for n in n_values_in_range:
                n_data = [row for row in filtered_data if row['N'] == n]
                steps = np.array([row['steps'] for row in n_data])

                # グラフ用のデータに追加
                # 横軸Nとして使用するため、この範囲の代表Nを使用
                # （範囲内に複数のNが含まれる場合は各Nを個別にプロット）
                if n not in n_to_steps:
                    n_to_steps[n] = []
                n_to_steps[n].extend(steps)

                # 各データポイントを記録（N値とsteps値のペア）
                n_values.extend([n] * len(steps))
                step_values.extend(steps)

                # 範囲情報を記録
                range_info_key = (episode_start, episode_end, n)
                range_info[range_info_key] = {
                    'episode_start': episode_start,
                    'episode_end': episode_end,
                    'N': n,
                    'data_count': len(steps)
                }

    # グラフを作成
    fig, ax = plt.subplots(figsize=(12, 8))

    # データポイントをプロット
    ax.scatter(n_values, step_values, alpha=0.6, s=30,
               label='Data points')

    # 2*N-1と2*N-1+15の直線をプロット
    # データに含まれるNの値の範囲を使用
    if len(n_to_steps) > 0:
        min_n = min(n_to_steps.keys())
        max_n = max(n_to_steps.keys())
        n_range = np.arange(min_n, max_n + 1)
        line_lower = 2 * n_range - 1
        line_upper = 2 * n_range - 1 + 15

        ax.plot(n_range, line_lower, 'g--', linewidth=2, label='2*N-1')
        ax.plot(n_range, line_upper, 'r--', linewidth=2,
                label='2*N-1+15')

    # グラフの設定
    ax.set_xlabel('N', fontsize=24, fontweight='bold')
    ax.set_ylabel('Steps', fontsize=24, fontweight='bold')
    ax.set_title('Steps per Episode by N', fontsize=26, fontweight='bold')

    # 横軸の目盛りをデータに含まれるNの値に設定
    if len(n_to_steps) > 0:
        n_ticks = sorted(n_to_steps.keys())
        ax.set_xticks(n_ticks)

    # 目盛りのフォントサイズを大きく
    ax.tick_params(axis='both', which='major', labelsize=22)

    ax.grid(True, alpha=0.3)
    legend_font = FontProperties(size=22, weight='bold')
    ax.legend(prop=legend_font)

    # 保存
    output_path = OUTPUT_DIR / 'steps_by_n_analysis.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nグラフを保存しました: {output_path}")

    # 統計を出力
    print("\n" + "=" * 80)
    print("統計: 2*N-1 から 2*N-1+15 の範囲内のデータ数")
    print("=" * 80)

    stats_data = []

    for n in sorted(n_to_steps.keys()):
        steps = np.array(n_to_steps[n])
        lower_bound = 2 * n - 1
        upper_bound = 2 * n - 1 + 15

        # 範囲内のデータ数をカウント
        in_range = np.sum((steps >= lower_bound) & (steps <= upper_bound))
        total = len(steps)
        percentage = (in_range / total * 100) if total > 0 else 0

        stats_data.append({
            'N': n,
            'Lower_bound': lower_bound,
            'Upper_bound': upper_bound,
            'Count_in_range': in_range,
            'Total_count': total,
            'Percentage(%)': f'{percentage:.2f}'
        })

        print(
            f"N={n:3d}: 範囲 [{lower_bound:3d}, {upper_bound:3d}] "
            f"内のデータ数: {in_range:3d}/{total:3d} ({percentage:5.2f}%)"
        )

    # 統計をCSVに保存
    stats_csv_path = OUTPUT_DIR / 'steps_range_statistics.csv'
    fieldnames = [
        'N', 'Lower_bound', 'Upper_bound', 'Count_in_range',
        'Total_count', 'Percentage(%)'
    ]
    with open(stats_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_data)
    print(f"\n統計データをCSVに保存しました: {stats_csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
