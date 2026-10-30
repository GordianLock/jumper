#!/bin/bash
# The contest video, end to end: every clip at 1920x1080 / 60 fps (four-view panels at
# 960x540, tiled), then the assembly. Meant for a GPU machine (Colab: run the notebook's
# setup cells first, so MuJoCo renders through NVIDIA EGL); on a CPU it takes hours.
#
#     PY=/content/jumper/.venv/bin/python bash studies/footage/render_final.sh [out_dir]
#
# Needs only ffmpeg beside the repository's Python environment: the slides, the speed
# labels and the stock-shell warning are committed as images
# (studies/footage/slides, studies/footage/assets/overlays), so no fonts are drawn here.
#
# Order: opening title 5 s | tuck roll set | statement slide 15 s | shoulder roll set |
# spin drift set | stock title 3 s | stock lay down + self-stand | stock stats 5 s
# (fading out over its last second) | 4 s black. A set: turntable | stats slide 3 s |
# four views (last 3 s cut) | 0.5x close-up, labelled, fading out over its last second.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=$(realpath -m "${1:-out/footage_final}")
C=$OUT/clips
W=$OUT/work
mkdir -p "$C" "$W"
export MUJOCO_GL=${MUJOCO_GL:-egl}
F=studies/footage
REC=$F/recordings
COL=design/carapace/colour
S=$F/slides
O=$F/assets/overlays
SCENE="--scene default --floor hardwood --surround grass --fps 60"
ENC="-c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p -r 60"
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }

render() {  # out-file, args...: skip a clip already rendered
  local file=$1; shift
  if [ -s "$C/$file" ] && [ "$(dur "$C/$file" 2>/dev/null | cut -d. -f1)" -gt 0 ] 2>/dev/null; then
    echo "have $file"; return
  fi
  local t0=$SECONDS
  "$PY" $F/render_footage.py "$@" 2>&1 | grep -E "^wrote|Error|error" || true
  [ -s "$C/$file" ] || { echo "FAILED: $file"; exit 1; }
  echo "  $file in $((SECONDS - t0)) s"
}

set_clips() {  # tag model colour recording phase
  local tag=$1 rec=$REC/$4 phase=$5
  local A="--model $2 --colour $COL/$3 $SCENE"
  render ${tag}_orbit.mp4 $rec "$C" $A --name $tag --orbit 10 --orbit-peak 85 --orbit-end-hold 1.5 --fade-out 1.5
  for ang in side front low top; do
    render ${tag}_${ang}_1x.mp4 $rec "$C" $A --name $tag --angles $ang --speeds 1 --center $phase --width 960 --height 540
  done
  render ${tag}_close_track_0.5x.mp4 $rec "$C" $A --name ${tag}_close --track $phase --track-pre 2.3 --track-pad 1.2 --speeds 0.5
}

set_clips tk CF1-tri30m-rb-v3 CF1-tri30m-rb-v3__30-oct-fa tuck_roll_v3_walk.npz jumper.tuck_roll
set_clips sr CF1-tri30m-rb-v4b CF1-tri30m-rb-v4b__30-oct-c shoulder_roll_v4b_walk.npz jumper.diag_roll
set_clips sd CF1-tri30m-rb-v4 CF1-tri30m-rb-v4__30-oct-thue-morse spin_drift_v4_walk.npz jumper.spin_drift
ST=$REC/lay_down_self_right_stock_walk.npz
render st_side_1x.mp4 $ST "$C" --model jumper $SCENE --name st --angles side --speeds 1 --center jumper.lie_down
render st_close_track_0.25x.mp4 $ST "$C" --model jumper $SCENE --name st_close --track jumper.self_right --track-pre 0.52 --track-pad 1.03 --speeds 0.25

echo "assembling"
for tag in tk sr sd; do
  d=$(dur $C/${tag}_side_1x.mp4)
  ffmpeg -loglevel error -y -i $C/${tag}_side_1x.mp4 -i $C/${tag}_front_1x.mp4 -i $C/${tag}_low_1x.mp4 -i $C/${tag}_top_1x.mp4 \
    -filter_complex "[0][1]hstack[t];[2][3]hstack[b];[t][b]vstack,trim=0:$(awk "BEGIN{print $d-3}"),setpts=PTS-STARTPTS" $ENC $W/${tag}_four.mp4
  d=$(dur $C/${tag}_close_track_0.5x.mp4)
  ffmpeg -loglevel error -y -i $C/${tag}_close_track_0.5x.mp4 -i $O/speed_0.5x.png \
    -filter_complex "[0][1]overlay=0:0,fade=t=out:st=$(awk "BEGIN{print $d-1}"):d=1" $ENC $W/${tag}_close.mp4
done
# Stock: the side view held 4 s at the landing (6.76 s in) under the warning, its last
# 3 s cut; then the 0.25x close-up of the self-stand, fading out as it walks off.
AT=6.76
ffmpeg -loglevel error -y -ss $AT -i $C/st_side_1x.mp4 -frames:v 1 $W/st_landing.png
d=$(dur $C/st_side_1x.mp4)
ffmpeg -loglevel error -y -i $C/st_side_1x.mp4 -loop 1 -t 4 -i $W/st_landing.png -i $O/stock_shell_warning.png \
  -filter_complex "[0]trim=0:$AT,setpts=PTS-STARTPTS,fps=60,format=yuv420p[a];[1][2]overlay=0:0,fps=60,format=yuv420p,setsar=1[b];[0]trim=start=$AT:end=$(awk "BEGIN{print $d-3}"),setpts=PTS-STARTPTS,fps=60,format=yuv420p[c];[a][b][c]concat=n=3:v=1:a=0" \
  $ENC $W/st_side.mp4
d=$(dur $C/st_close_track_0.25x.mp4)
ffmpeg -loglevel error -y -i $C/st_close_track_0.25x.mp4 -i $O/speed_0.25x.png \
  -filter_complex "[0][1]overlay=0:0,fade=t=out:st=$(awk "BEGIN{print $d-1}"):d=1" $ENC $W/st_close.mp4

in=(); filt=""; n=0
add_still() { in+=(-loop 1 -t "$2" -i "$1"); filt+="[$n]scale=1920:1080,setsar=1,fps=60,format=yuv420p${3:-}[v$n];"; n=$((n+1)); }
add_clip() { in+=(-i "$1"); filt+="[$n]scale=1920:1080,setsar=1,fps=60,format=yuv420p[v$n];"; n=$((n+1)); }
add_set() { add_clip $C/$1_orbit.mp4; add_still $S/$2.png 3; add_clip $W/$1_four.mp4; add_clip $W/$1_close.mp4; }
add_still $S/opening_title.png 5
add_set tk tuck_roll_v3
add_still $S/form_follows_function.png 15
add_set sr shoulder_roll_v4b
add_set sd spin_drift_v4
add_still $S/stock_self_righting.png 3
add_clip $W/st_side.mp4
add_clip $W/st_close.mp4
add_still $S/self_righting_stock.png 5 ",fade=t=out:st=4:d=1"
in+=(-f lavfi -t 4 -i color=black:s=1920x1080:r=60); filt+="[$n]setsar=1,format=yuv420p[v$n];"; n=$((n+1))
cat=""; for i in $(seq 0 $((n - 1))); do cat+="[v$i]"; done
FINAL=$OUT/jumper_crab_kingkong_challenge.mp4
ffmpeg -loglevel error -y "${in[@]}" -filter_complex "${filt}${cat}concat=n=$n:v=1:a=0" $ENC -movflags +faststart "$FINAL"
echo "wrote $FINAL ($(dur "$FINAL") s)"
