import io
import pickle

import numpy as np

date = "20251112_011358"
file_name = "H_actor_N1_total10000ep.pkl"


class CompatibleUnpickler(pickle.Unpickler):
    """numpy 2.0以降で保存されたpickleをnumpy 1.xで読み込むためのUnpickler"""

    def find_class(self, module, name):
        # numpy._coreをnumpy.coreにマッピング
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        return super().find_class(module, name)


def compatible_pickle_loads(data):
    """numpy 2.0以降で保存されたpickleをnumpy 1.xで読み込むためのヘルパー関数"""
    if isinstance(data, bytes):
        return CompatibleUnpickler(io.BytesIO(data)).load()
    else:
        return pickle.loads(data)


# H_actor_N1_total100ep.pklの中身を確認
print("=" * 60)
print(f"{file_name} の内容")
print("=" * 60)

file_path = f"output/logs/actor_only_training/run_{date}/{file_name}"


def format_cell_state(state_13):
    """13セルの状態を5x5の表形式で表示（4つの角は無視）"""
    # 5x5のマトリックスを作成（角は'X'で表示）
    matrix = [["X" for _ in range(5)] for _ in range(5)]

    # 13セルを配置（角以外の位置）
    # 5x5の配置で角(0,0), (0,4), (4,0), (4,4)を除いた21個の位置から13個を選択
    cell_positions = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (0, 2),
        (4, 2),
        (2, 0),
        (2, 4),  # 中段
    ]

    for i, (row, col) in enumerate(cell_positions):
        if i < len(state_13):
            matrix[row][col] = "■" if state_13[i] else "□"

    # 表形式で表示
    result = []
    for row in matrix:
        result.append(" ".join(row))
    return "\n".join(result)


def format_h_values(h_values):
    """H値を3x3の表形式で表示（4つの角は無視）"""
    if not isinstance(h_values, (list, tuple, np.ndarray)):
        return str(h_values)

    # 3x3のマトリックスを作成（角は'X'で表示）
    matrix = [["X" for _ in range(3)] for _ in range(3)]

    # 5つの値を配置（角以外の位置）
    positions = [(0, 1), (2, 1), (1, 0), (1, 2), (1, 1)]

    for i, (row, col) in enumerate(positions):
        if i < len(h_values):
            matrix[row][col] = f"{h_values[i]:.3f}"

    # 表形式で表示
    result = []
    for row in matrix:
        result.append(" ".join(row))
    return "\n".join(result)


# 出力ファイル名
output_file = "H_actor_analysis.txt"

try:
    with open(file_path, "rb") as f:
        # メインファイルもCompatibleUnpicklerを使用
        data = CompatibleUnpickler(f).load()

        # テキストファイルに書き込み
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write("=" * 60 + "\n")
            out_f.write(f"{file_name} の内容\n")
            out_f.write("=" * 60 + "\n")
            out_f.write(f"Type: {type(data)}\n")

            if isinstance(data, dict):
                out_f.write(f"\n総状態数: {len(data)}\n")

                # 最初のキーの構造を確認（デバッグ用）
                if len(data) > 0:
                    first_key = list(data.keys())[0]
                    out_f.write(f"\n最初のキーの型: {type(first_key)}\n")
                    if isinstance(first_key, bytes):
                        try:
                            first_decoded = compatible_pickle_loads(first_key)
                            out_f.write(f"デコード後の型: {type(first_decoded)}\n")
                            if isinstance(first_decoded, tuple):
                                out_f.write(f"タプルの長さ: {len(first_decoded)}\n")
                                for i, item in enumerate(first_decoded):
                                    out_f.write(
                                        f"  要素{i}: 型={type(item)}, 値={item}\n"
                                    )
                                    if isinstance(
                                        item, (list, tuple, np.ndarray)
                                    ):
                                        out_f.write(
                                            f"    要素{i}の長さ/形状: {len(item) if hasattr(item, '__len__') else 'N/A'}\n"
                                        )
                                        if len(item) > 0 and isinstance(
                                            item[0], (list, tuple, np.ndarray)
                                        ):
                                            out_f.write(f"    要素{i}は2次元配列\n")
                        except Exception as e:
                            out_f.write(f"最初のキーのデコードエラー: {e}\n")

                # データをデコードしてブロック位置でソート
                decoded_data = []
                error_count = 0
                for key, value in data.items():
                    try:
                        if isinstance(key, bytes):
                            decoded = compatible_pickle_loads(key)
                            # デコード後の構造を確認
                            if isinstance(decoded, tuple):
                                if len(decoded) == 2:
                                    state_13, block_idx = decoded
                                    # state_13が2次元配列の場合、1次元に変換
                                    if isinstance(
                                        state_13, (list, tuple, np.ndarray)
                                    ):
                                        if len(state_13) > 0 and isinstance(
                                            state_13[0],
                                            (list, tuple, np.ndarray),
                                        ):
                                            # 2次元配列の場合、フラット化
                                            state_13 = tuple(
                                                item
                                                for sublist in state_13
                                                for item in sublist
                                            )[:13]
                                        else:
                                            # 既に1次元の場合
                                            state_13 = tuple(state_13)[:13]
                                    # block_idxが2次元の場合の処理
                                    if isinstance(
                                        block_idx, (list, tuple, np.ndarray)
                                    ):
                                        if len(block_idx) > 0 and isinstance(
                                            block_idx[0],
                                            (list, tuple, np.ndarray),
                                        ):
                                            # 2次元配列の場合、最初の要素を使用
                                            block_idx = tuple(block_idx[0])
                                        else:
                                            block_idx = tuple(block_idx)
                                    decoded_data.append(
                                        (block_idx, state_13, value)
                                    )
                                else:
                                    out_f.write(f"予期しないタプル長: {len(decoded)}\n")
                            else:
                                out_f.write(f"予期しないキー型: {type(decoded)}\n")
                        else:
                            # キーが既にデコード済みの場合
                            if isinstance(key, tuple) and len(key) == 2:
                                state_13, block_idx = key
                                # 同様の処理
                                if isinstance(
                                    state_13, (list, tuple, np.ndarray)
                                ):
                                    if len(state_13) > 0 and isinstance(
                                        state_13[0], (list, tuple, np.ndarray)
                                    ):
                                        state_13 = tuple(
                                            item
                                            for sublist in state_13
                                            for item in sublist
                                        )[:13]
                                    else:
                                        state_13 = tuple(state_13)[:13]
                                if isinstance(
                                    block_idx, (list, tuple, np.ndarray)
                                ):
                                    if len(block_idx) > 0 and isinstance(
                                        block_idx[0], (list, tuple, np.ndarray)
                                    ):
                                        block_idx = tuple(block_idx[0])
                                    else:
                                        block_idx = tuple(block_idx)
                                decoded_data.append(
                                    (block_idx, state_13, value)
                                )
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # 最初の5つのエラーのみ表示
                            import traceback

                            out_f.write(f"デコードエラー (キー型: {type(key)}): {e}\n")
                            out_f.write(f"  詳細: {traceback.format_exc()}\n")
                if error_count > 5:
                    out_f.write(f"他に {error_count - 5} 個のデコードエラーがあります\n")

                # ブロック位置でソート
                # block_idxが正しい形式かチェックしてからソート
                def sort_key(x):
                    block_idx = x[0]
                    if (
                        isinstance(block_idx, (list, tuple, np.ndarray))
                        and len(block_idx) >= 2
                    ):
                        return (block_idx[0], block_idx[1])
                    else:
                        return (0, 0)  # 不正な形式の場合は先頭に配置

                decoded_data.sort(key=sort_key)  # (x, y)でソート

                out_f.write(f"\nソート済み状態数: {len(decoded_data)}\n")
                out_f.write("\n詳細表示:\n")

                for i, (block_idx, state_13, h_values) in enumerate(
                    decoded_data
                ):
                    out_f.write(f"\n状態 #{i+1}: ブロック位置 {block_idx}\n")
                    out_f.write("13セル詳細 (5x5, 角は無視):\n")
                    out_f.write(format_cell_state(state_13) + "\n")
                    out_f.write("H値 (3x3, 角は無視):\n")
                    out_f.write(format_h_values(h_values) + "\n")

            elif isinstance(data, np.ndarray):
                out_f.write(f"Shape: {data.shape}\n")
                out_f.write(f"Dtype: {data.dtype}\n")
                out_f.write(f"Range: [{data.min():.4f}, {data.max():.4f}]\n")
                out_f.write("\n最初の10個の値:\n")
                out_f.write(str(data[:10]) + "\n")

            elif isinstance(data, list):
                out_f.write(f"Length: {len(data)}\n")
                out_f.write(
                    f"Type of elements: {type(data[0]) if len(data) > 0 else 'Empty list'}\n"
                )
                out_f.write("\n最初の10個の値:\n")
                for i, item in enumerate(data[:10]):
                    out_f.write(f"  [{i}]: {item}\n")

            else:
                out_f.write(f"Value: {data}\n")

    print(f"分析結果を {output_file} に保存しました。")

except FileNotFoundError:
    print(f"ファイルが見つかりません: {file_path}")
except Exception as e:
    import traceback

    print(f"エラーが発生しました: {e}")
    print(traceback.format_exc())

print("\n" + "=" * 60)
print("処理完了")
print("=" * 60)
