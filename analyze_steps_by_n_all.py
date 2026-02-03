#!/usr/bin/env python3
"""
CSVファイルから全エピソード範囲のデータを取得し、
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

    # 全データをNごとに集計
    for row in data:
        n = row['N']
        steps = row['steps']

        if n not in n_to_steps:
            n_to_steps[n] = []
        n_to_steps[n].append(steps)

        # グラフ用のデータに追加
        n_values.append(n)
        step_values.append(steps)

    # 各Nのデータ数を表示
    print(f"\n含まれるN値: {sorted(n_to_steps.keys())}")
    for n in sorted(n_to_steps.keys()):
        print(f"N={n}: {len(n_to_steps[n])}データポイント")

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
    ax.set_title('Steps per Episode by N (All Episodes)',
                 fontsize=26, fontweight='bold')

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
    output_path = OUTPUT_DIR / 'steps_by_n_analysis_all.pdf'
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
    stats_csv_path = OUTPUT_DIR / 'steps_range_statistics_all.csv'
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

