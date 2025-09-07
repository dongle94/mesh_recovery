#!/usr/bin/env bash
set -euo pipefail

# Ensure we operate relative to the script directory (weights/vibe)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP="$DIR/vibe_data.zip"
URL="https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"

echo "[prepare_data] target dir: $DIR"

echo "[prepare_data] downloading vibe data..."
gdown -O "$ZIP" "$URL"

echo "[prepare_data] unzipping into $DIR ..."
unzip -o "$ZIP" -d "$DIR"

echo "[prepare_data] removing zip file"
rm -f "$ZIP"

# If contents were extracted into a top-level 'vibe_data' folder, move them up
if [ -d "$DIR/vibe_data" ]; then
	echo "[prepare_data] moving files from vibe_data/ to $DIR"
	# include dotfiles
	(shopt -s dotglob && mv -f "$DIR/vibe_data"/* "$DIR"/) || true
	rm -rf "$DIR/vibe_data"
fi

echo "[prepare_data] removing yolov3.weights"
rm "$DIR/yolov3.weights"

echo "[prepare_data] done"
