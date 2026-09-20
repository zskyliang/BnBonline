#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$project_root/build"
package_path="$build_root/crazygames-upload.zip"
godot_bin="${GODOT_BIN:-godot}"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/bnb-crazygames.XXXXXX")"
package_tmp="${staging_dir}.zip"

cleanup() {
	rm -rf "$staging_dir"
	rm -f "$package_tmp"
}
trap cleanup EXIT

mkdir -p "$build_root"

echo "[1/4] Importing Godot resources..."
"$godot_bin" --headless --editor --path "$project_root" --import

echo "[2/4] Running release regression checks..."
"$godot_bin" \
	--headless \
	--path "$project_root" \
	--script res://tests/test_runner.gd
"$godot_bin" \
	--headless \
	--path "$project_root" \
	--script res://tests/storybook_asset_runner.gd \
	-- --mute
"$godot_bin" \
	--headless \
	--path "$project_root" \
	--script res://tests/storybook_visual_runner.gd \
	-- --mute
"$godot_bin" \
	--headless \
	--path "$project_root" \
	--script res://tests/smoke_runner.gd \
	-- --mute

echo "[3/4] Exporting the CrazyGames release build..."
"$godot_bin" \
	--headless \
	--path "$project_root" \
	--export-release "CrazyGames" \
	"$staging_dir/index.html"

echo "[4/4] Validating and packaging the upload ZIP..."
required_files=(
	"index.html"
	"index.js"
	"index.pck"
	"index.wasm"
)
for required_file in "${required_files[@]}"; do
	if [[ ! -f "$staging_dir/$required_file" ]]; then
		echo "Missing required CrazyGames file: $required_file" >&2
		exit 1
	fi
done

file_count="$(find "$staging_dir" -type f | wc -l | tr -d ' ')"
raw_bytes="$(find "$staging_dir" -type f -exec stat -f%z {} + | awk '{sum += $1} END {print sum + 0}')"
gzip_bytes=0
while IFS= read -r -d '' file; do
	compressed_bytes="$(gzip -9 -c "$file" | wc -c | tr -d ' ')"
	gzip_bytes=$((gzip_bytes + compressed_bytes))
done < <(find "$staging_dir" -type f -print0)

max_files=1500
max_total_bytes=$((250 * 1024 * 1024))
max_initial_bytes=$((50 * 1024 * 1024))

if ((file_count > max_files)); then
	echo "File count $file_count exceeds CrazyGames limit $max_files" >&2
	exit 1
fi
if ((raw_bytes > max_total_bytes)); then
	echo "Raw package size $raw_bytes exceeds CrazyGames total limit" >&2
	exit 1
fi
if ((gzip_bytes > max_initial_bytes)); then
	echo "Estimated initial transfer $gzip_bytes exceeds 50 MiB" >&2
	exit 1
fi

(
	cd "$staging_dir"
	zip -q -r "$package_tmp" . -x "*.DS_Store" "._*"
)
zip -T "$package_tmp" >/dev/null
mv -f "$package_tmp" "$package_path"

raw_mib="$(awk -v bytes="$raw_bytes" 'BEGIN {printf "%.2f", bytes / 1048576}')"
gzip_mib="$(awk -v bytes="$gzip_bytes" 'BEGIN {printf "%.2f", bytes / 1048576}')"
package_mib="$(du -m "$package_path" | awk '{print $1}')"
package_sha256="$(LC_ALL=C shasum -a 256 "$package_path" | awk '{print $1}')"

echo "CrazyGames package ready: $package_path"
echo "Files: $file_count / $max_files"
echo "Raw total: ${raw_mib} MiB / 250 MiB"
echo "Estimated gzip transfer: ${gzip_mib} MiB / 50 MiB"
echo "Upload ZIP: ${package_mib} MiB"
echo "SHA-256: $package_sha256"
