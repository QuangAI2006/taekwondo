# -*- coding: utf-8 -*-
"""
batch_convert_videos.py
功能：
1. 递归扫描 data/videos 及其子目录。
2. 转换非 MP4 视频为 MP4。
3. 【增强】深度递归查找 scripts 目录下的 ffmpeg.exe（防止用户放入了文件夹）。
4. 【调试】打印目录文件列表，帮助排查路径问题。

使用方法：
    python scripts/batch_convert_videos.py              (默认：转换成功后删除源文件)
    python scripts/batch_convert_videos.py --keep_source (保留源文件)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[1]
# 脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "data" / "videos"

TARGET_EXTS = {".mov", ".avi", ".mkv", ".flv", ".webm", ".wmv", ".m4v"}


def get_ffmpeg_cmd():
    print("-" * 30)
    print(f"[调试信息] 脚本所在目录: {SCRIPT_DIR}")

    # 打印当前目录下的文件，帮用户看看到底放了啥
    try:
        files = [f.name for f in SCRIPT_DIR.iterdir()]
        print(f"[调试信息] 目录下的文件: {files}")
    except Exception as e:
        print(f"[调试警告] 无法列出目录文件: {e}")

    # 1. 优先检查同级目录
    local_ffmpeg = SCRIPT_DIR / "ffmpeg.exe"
    if local_ffmpeg.exists():
        print(f"[调试信息] 在根目录找到: {local_ffmpeg.name}")
        return str(local_ffmpeg)

    # 2. 【新增】深度搜索：防止用户放了文件夹但没拿出来
    print("[调试信息] 正在子文件夹中搜索 ffmpeg.exe ...")
    # rglob 会递归查找所有子目录
    found_files = list(SCRIPT_DIR.rglob("ffmpeg.exe"))
    if found_files:
        best_match = found_files[0]
        print(f"[调试信息] 在子目录找到 FFmpeg: {best_match}")
        return str(best_match)

    # 3. 最后尝试系统变量
    if shutil_which("ffmpeg"):
        print("[调试信息] 使用系统全局 FFmpeg")
        return "ffmpeg"

    print("[调试信息] 未找到 ffmpeg.exe")
    print("-" * 30)
    return None


# 兼容 Python < 3.3 的 shutil.which 替代方案
def shutil_which(pgm):
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
        if os.path.exists(p + '.exe') and os.access(p + '.exe', os.X_OK):
            return p + '.exe'
    return None


FFMPEG_CMD = get_ffmpeg_cmd()


def check_ffmpeg():
    if not FFMPEG_CMD: return False
    try:
        subprocess.run([FFMPEG_CMD, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False


def convert_video(src_path: Path, delete_source: bool = False):
    dst_path = src_path.with_suffix(".mp4")
    if dst_path.exists():
        print(f"[跳过] {dst_path.name}")
        # 如果目标已存在且需要删除源文件，为了保险起见，这里不自动删除，
        # 除非确认目标文件是有效的（这里不做深度校验，故保守处理）
        return

    print(f"[转换中] {src_path.name} -> {dst_path.name}")
    # 添加 -pix_fmt yuv420p 以确保兼容性
    cmd = [
        FFMPEG_CMD, "-i", str(src_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(dst_path), "-y", "-loglevel", "error"
    ]

    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print(f"[成功] {dst_path.name}")
            if delete_source:
                try:
                    os.remove(src_path)
                    print(f"[删除源] {src_path.name}")
                except Exception as e:
                    print(f"[删除失败] {e}")
        else:
            print(f"[失败] 错误码: {ret.returncode}")
            if dst_path.exists():
                try:
                    os.remove(dst_path)
                except:
                    pass
    except Exception as e:
        print(f"[异常] {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(VIDEO_ROOT))
    # 修改逻辑：默认为删除源文件，只有加了 --keep_source 才保留
    parser.add_argument("--keep_source", action="store_true", help="转换成功后保留源文件 (默认删除)")
    args = parser.parse_args()

    if not check_ffmpeg():
        print("\n" + "!" * 40)
        print("[错误] 依然没找到 ffmpeg.exe！")
        print("请查看上方的 [调试信息]：")
        print("1. 确保你放入的是解压后的 ffmpeg.exe (图标通常是黑底白字)，而不是 .zip 或 .tar.xz 包。")
        print("2. 确保文件名为 ffmpeg.exe (不要叫 ffmpeg-8.0.1.tar.xz)。")
        print("!" * 40)
        return

    root_dir = Path(args.root)
    files = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in TARGET_EXTS]

    if not files:
        print(f"在 {root_dir} 未发现视频文件。")
        return

    print(f"准备转换 {len(files)} 个文件 (使用: {FFMPEG_CMD})")

    # 确定是否删除
    should_delete = not args.keep_source
    if should_delete:
        print("提示：转换成功后将自动【删除】源文件。")
    else:
        print("提示：转换后将【保留】源文件。")

    for i, p in enumerate(files, 1):
        convert_video(p, delete_source=should_delete)
    print("完成。")


if __name__ == "__main__":
    main()