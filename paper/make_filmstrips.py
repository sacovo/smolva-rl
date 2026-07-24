"""Regenerate the filmstrip figures in figs/ from the rollout videos in ../analysis_videos/.

Layout: 10 evenly spaced frames per episode, each resized to 112x112 and
concatenated left-to-right with 2 px white dividers. drawer_compare stacks
two strips (top: co-trained policy succeeds, bottom: 250k base policy fails)
with a 12 px white divider.
"""
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent / "analysis_videos"
FIGS_DIR = SCRIPT_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

FRAME_PX = 112
N_FRAMES = 10
SEP_PX = 2
DIVIDER_PX = 12


def strip(video_path: Path | str) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, n - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {idx} of {video_path}")
        frames.append(cv2.resize(frame, (FRAME_PX, FRAME_PX), interpolation=cv2.INTER_AREA))
    cap.release()
    sep = np.full((FRAME_PX, SEP_PX, 3), 255, dtype=np.uint8)
    out = [frames[0]]
    for f in frames[1:]:
        out += [sep, f]
    return np.hstack(out)


def main():
    top = strip(VIDEOS_DIR / "cotrain_goalT0_drawer_SUCCESS_ep0.mp4")
    bottom = strip(VIDEOS_DIR / "250k_goalT0_drawer_FAIL_ep0.mp4")
    divider = np.full((DIVIDER_PX, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.imwrite(str(FIGS_DIR / "drawer_compare.png"), np.vstack([top, divider, bottom]))

    cv2.imwrite(str(FIGS_DIR / "lib10_dualobj_fail.png"), strip(VIDEOS_DIR / "cotrain_lib10T7_putBOTH_FAIL_ep0.mp4"))
    print(f"wrote {FIGS_DIR}/drawer_compare.png, {FIGS_DIR}/lib10_dualobj_fail.png")


if __name__ == "__main__":
    main()
