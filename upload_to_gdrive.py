#!/usr/bin/env python3
"""
指定のローカルフォルダを Google Drive の指定フォルダにアップロードするスクリプト。

【事前準備】
1. Google Cloud Console でプロジェクトを作成
2. Google Drive API を有効化
3. OAuth 2.0 クライアント ID（デスクトップアプリ）を作成
4. 認証情報をダウンロードし、このスクリプトと同じディレクトリに
   「credentials.json」として保存

【使い方】
  python upload_to_gdrive.py <ローカルフォルダパス> [Google Drive フォルダID]

  指定したフォルダごと Drive に送ります（例: output を指定すると、
  Drive 上に「output」フォルダができ、その中に中身が入ります）。
  フォルダIDはコード内の DEFAULT_DRIVE_FOLDER_ID に書いておくと省略可能。
"""

import argparse
import mimetypes
import os
import sys
# オプション: 環境に合わせて credentials のパスを変更可能
CREDENTIALS_PATH = os.environ.get("GDRIVE_CREDENTIALS", "credentials.json")
TOKEN_PATH = os.environ.get("GDRIVE_TOKEN", "token.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]

# アップロード先の Google Drive フォルダID（固定したい場合はここに入力）
# ブラウザでフォルダを開いたときの URL の .../folders/ の後ろの部分
# DEFAULT_DRIVE_FOLDER_ID = "14q5z-Ryjped6hCaZfQeD4bBX5ZbPeSve"  # critic学習データのフォルダID
DEFAULT_DRIVE_FOLDER_ID = "1FTRc_-3dxBbj7r2_Wt0UMWGZDzoThPs3"  # actor学習データのフォルダID

# 5MB 以上は resumable でアップロード
RESUMABLE_THRESHOLD_BYTES = 5 * 1024 * 1024


def get_drive_service():
    """OAuth2 で認証し、Drive API サービスを返す。"""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        print(
            "必要なパッケージがありません。以下を実行してください:\n"
            "  pip install -r requirements-gdrive.txt"
        )
        raise SystemExit(1) from e

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                print(
                    f"認証情報が見つかりません: {CREDENTIALS_PATH}\n"
                    "Google Cloud Console で OAuth クライアントを作成し、"
                    "credentials.json をダウンロードしてこのディレクトリに置いてください。"
                )
                raise SystemExit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_mime_type(file_path: str) -> str:
    """MIME タイプを返す。不明なら application/octet-stream。"""
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def ensure_drive_folder(service, name: str, parent_id: str) -> str:
    """
    指定親フォルダ内に同名フォルダがあればそのIDを、なければ作成してIDを返す。
    """
    escaped = name.replace("'", "''")
    q = (
        f"name='{escaped}' and '{parent_id}' in parents "
        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = (
        service.files()
        .list(q=q, spaces="drive", fields="files(id, name)")
        .execute()
    )
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_file(service, local_path: str, drive_parent_id: str) -> str | None:
    """1ファイルをアップロードし、アップロードしたファイルのIDを返す。"""
    from googleapiclient.http import MediaFileUpload

    name = os.path.basename(local_path)
    mime = get_mime_type(local_path)
    size = os.path.getsize(local_path)
    resumable = size >= RESUMABLE_THRESHOLD_BYTES
    kwargs = {"mimetype": mime, "resumable": resumable}
    if resumable:
        kwargs["chunksize"] = 256 * 1024
    media = MediaFileUpload(local_path, **kwargs)
    metadata = {"name": name, "parents": [drive_parent_id]}
    try:
        file = (
            service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
        return file.get("id")
    except Exception as e:
        print(f"  アップロード失敗: {local_path} - {e}")
        return None


def upload_folder_recursive(
    service,
    local_root: str,
    drive_parent_id: str,
    rel_path: str = "",
) -> tuple[int, int]:
    """
    ローカルフォルダを再帰的に Drive に反映する。
    返り値: (成功したファイル数, 失敗したファイル数)
    """
    local_dir = os.path.join(local_root, rel_path) if rel_path else local_root
    success, fail = 0, 0

    try:
        entries = sorted(os.listdir(local_dir))
    except OSError as e:
        print(f"  スキップ（読めません）: {local_dir} - {e}")
        return 0, 0

    for name in entries:
        local_full = os.path.join(local_dir, name)
        rel_child = os.path.join(rel_path, name) if rel_path else name

        if os.path.isdir(local_full):
            drive_folder_id = ensure_drive_folder(service, name, drive_parent_id)
            s, f = upload_folder_recursive(
                service, local_root, drive_folder_id, rel_child
            )
            success += s
            fail += f
        else:
            fid = upload_file(service, local_full, drive_parent_id)
            if fid:
                success += 1
                print(f"  OK: {rel_child}")
            else:
                fail += 1

    return success, fail


def main():
    parser = argparse.ArgumentParser(
        description="指定フォルダを Google Drive の指定フォルダにアップロードします。"
    )
    parser.add_argument(
        "local_folder",
        type=str,
        help="アップロードするローカルフォルダのパス",
    )
    parser.add_argument(
        "drive_folder_id",
        type=str,
        nargs="?",
        default=None,
        help="アップロード先の Drive フォルダID（省略時はコード内 DEFAULT_DRIVE_FOLDER_ID を使用）",
    )
    args = parser.parse_args()

    local_folder = os.path.abspath(args.local_folder)
    if not os.path.isdir(local_folder):
        print(f"エラー: フォルダが見つかりません: {local_folder}")
        sys.exit(1)

    drive_parent_id = (args.drive_folder_id or DEFAULT_DRIVE_FOLDER_ID).strip()
    if not drive_parent_id:
        print(
            "エラー: アップロード先のフォルダIDが未指定です。\n"
            "  - コード内の DEFAULT_DRIVE_FOLDER_ID にフォルダIDを入力する\n"
            "  - またはコマンドの第2引数で指定する"
        )
        sys.exit(1)

    print("Google Drive に接続しています...")
    service = get_drive_service()
    # 指定フォルダごと入れる: Drive に同名フォルダを作り、その中に中身をアップロード
    folder_name = os.path.basename(local_folder.rstrip(os.sep))
    drive_parent_id = ensure_drive_folder(service, folder_name, drive_parent_id)
    print(f"フォルダ作成/使用: {folder_name} (ID: {drive_parent_id})")

    print(f"アップロード開始: {local_folder} -> Drive/{folder_name}")
    success, fail = upload_folder_recursive(service, local_folder, drive_parent_id)
    print(f"完了: 成功 {success} ファイル, 失敗 {fail} ファイル")


if __name__ == "__main__":
    main()
