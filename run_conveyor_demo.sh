#!/usr/bin/env bash
# Conveyor bottle inspection: capless bottles are detected and picked into the reject bin (Isaac Sim 6.0.1).
#
#   ./run_v2_demo.sh                 live in the Isaac Sim window (stays open when done)
#   ./run_v2_demo.sh --record        live + real-time screen recording of the Isaac Sim window
#                                    (3D view left, perception panel right) -> screen_recording.mp4
#   ./run_v2_demo.sh --headless      no window; writes demo.mp4 from rendered frames (fallback)
#   options: --seed N (bottle stream), env N=<bottles> (default 24)
#
# Outputs: /home/vishal/poc/logs/final_demo/v2_<mode>_seed<N>/
set -euo pipefail
ISAAC=/home/vishal/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64
SCRIPT=/home/vishal/poc/conveyor_inspection/conveyor_demo.py
export DISPLAY=${DISPLAY:-:1}

MODE=live
SEED=1
while [ $# -gt 0 ]; do
  case "$1" in
    --record) MODE=record ;;
    --headless) MODE=headless ;;
    --seed) SEED="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

OUT=/home/vishal/poc/logs/conveyor/video_${MODE}_seed${SEED}
rm -rf "$OUT"; mkdir -p "$OUT"
pkill -f "kit/ki[t] apps/isaacsim" 2>/dev/null || true
sleep 1
cd "$ISAAC"

if [ "$MODE" = headless ]; then
  echo "[conveyor] headless run, seed $SEED -> $OUT"
  DEMO_LIVE=0 DEMO_FRAMES=1 DEMO_ROOM=1 DEMO_N=${N:-24} DEMO_SEED=$SEED DEMO_OUT=$OUT \
    ./kit/kit apps/isaacsim.exp.base.python.kit --no-window --exec "$SCRIPT" > "$OUT/kit.log" 2>&1 || true
  grep -E "SUMMARY|FAILED" "$OUT/demo_log.txt" || true
  ffmpeg -loglevel error -y -framerate 30 -i "$OUT/frames/f_%06d.jpg" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/demo.mp4"
  rm -rf "$OUT/frames"
  echo "[conveyor] video: $OUT/demo.mp4"
  exit 0
fi

QUIT=0; [ "$MODE" = record ] && QUIT=1
echo "[conveyor] live run, seed $SEED -> $OUT   (progress: tail -f $OUT/demo_log.txt)"
DEMO_LIVE=1 DEMO_QUIT_WHEN_DONE=$QUIT DEMO_N=${N:-24} DEMO_SEED=$SEED DEMO_OUT=$OUT \
  ./kit/kit apps/isaacsim.exp.full.kit --exec "$SCRIPT" \
  --/app/window/width=1920 --/app/window/height=1080 --/app/window/x=0 --/app/window/y=0 \
  --/app/file/ignoreUnsavedOnExit=true \
  > "$OUT/kit.log" 2>&1 &
KIT=$!

if [ "$MODE" = live ]; then
  wait $KIT
  exit 0
fi

# ---- record: wait until the scene is up, then grab the Isaac Sim window in real time
until grep -q "ready: seed" "$OUT/demo_log.txt" 2>/dev/null; do
  kill -0 $KIT 2>/dev/null || { echo "[conveyor] Isaac Sim exited early, see $OUT/kit.log"; exit 1; }
  sleep 1
done
# the client window (class "Isaac Sim ..."), not the window-manager frame around it
WIN=$(xwininfo -root -tree | grep -i '"Isaac Sim' | grep -v "mutter-x11-frames" | awk '{print $1}' | head -1)
read -r X Y W H < <(xwininfo -id "$WIN" | awk '/Absolute upper-left X/{x=$4} /Absolute upper-left Y/{y=$4}
                                               /Width:/{w=$2} /Height:/{h=$2} END{print x, y, w, h}')
MENU=48   # crop the app menu bar off the top
STATUS=52 # and the status bar at the bottom
Y=$((Y + MENU)); H=$((H - MENU - STATUS))
W=$((W / 2 * 2)); H=$((H / 2 * 2))
echo "[conveyor] recording window $WIN at ${W}x${H}+${X},${Y}"
ffmpeg -loglevel warning -y -f x11grab -draw_mouse 0 -framerate 30 -video_size ${W}x${H} -i "${DISPLAY}+${X},${Y}" \
  -vf "scale=1920:-2" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart \
  "$OUT/screen_recording.mp4" < /dev/null 2> "$OUT/ffmpeg.log" &
FF=$!
until grep -qE "SUMMARY|FAILED" "$OUT/demo_log.txt" 2>/dev/null; do
  kill -0 $KIT 2>/dev/null || break
  sleep 1
done
sleep 8   # keep the DONE panel in the recording
kill -INT $FF 2>/dev/null || true
wait $FF 2>/dev/null || true
wait $KIT 2>/dev/null || true
grep -E "SUMMARY|FAILED" "$OUT/demo_log.txt" || true
echo "[conveyor] real-time recording: $OUT/screen_recording.mp4"
