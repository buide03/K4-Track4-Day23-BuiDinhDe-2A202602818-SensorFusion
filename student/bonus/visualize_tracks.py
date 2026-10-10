"""Bonus: BEV visualization of tracks, measurements and the camera update (RUBRIC §2, +3).

Replays the tracking loop of ``run_lab.run`` (lidar-only and fused, original
calibration, seed 0) on the detection cache built by ``calibration_study.py``
and writes annotated PNGs to ``student/bonus/``:

1. ``bev_overview.png``        — one frame: GT, LiDAR measurements, confirmed tracks of
                                  both modes with 2σ ellipses, camera FOV.
2. ``camera_update_effect.png`` — one track, one frame: state/ellipse after the LiDAR
                                  update (prior of AssocC) vs after the camera update, with
                                  the back-projected camera ray of the associated pixel.
3. ``track_error_timeline.png`` — the same track over time: XY error to the nearest GT and
                                  lateral std sqrt(P[1,1]) for lidar-only vs fused.

Usage (from repo root, after ``set -a; source .env; set +a``):

    python student/bonus/visualize_tracks.py --cache /tmp/day23_calib_cache.pkl

The cache is created by ``calibration_study.py --cache <same path>``.
``student/artifacts/`` is never touched.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_study import CAMERA_NOISE_PX  # noqa: E402
from calibration_study import _camera_sensor  # noqa: E402

from fusion_lab import tracking_params  # noqa: E402
from fusion_lab.paths import student_root  # noqa: E402
from fusion_lab.scripts.run_lab import _setup_import_paths  # noqa: E402
from fusion_lab.tracking.filter import Filter  # noqa: E402
from fusion_lab.tracking.manager import TrackManager  # noqa: E402
from fusion_lab.tracking.sensors import Sensor  # noqa: E402
from fusion_lab.workspace_loader import load_workspace  # noqa: E402

GATE_XY_M = 2.0


class CameraRecorder(Filter):
    """Filter that stores (prior, posterior, pixel) for every camera update."""

    def __init__(self, kalman_mod):
        super().__init__(kalman_mod)
        self.frame = -1
        self.events = []

    def update(self, track, meas):
        if meas.sensor.name != "camera":
            super().update(track, meas)
            return
        x0, P0 = track.x.copy(), track.P.copy()
        super().update(track, meas)
        self.events.append({"frame": self.frame, "id": track.id, "x0": x0, "P0": P0,
                            "x1": track.x.copy(), "P1": track.P.copy(),
                            "z": np.asarray(meas.z).reshape(-1)})


def replay(cache, ws, fused: bool):
    """Return per-frame snapshots {frame: {id: (x, P, state)}} and camera update events."""
    kalman, assoc, cam_mod = ws["kalman"], ws["association"], ws["camera_fusion"]
    rng = np.random.default_rng(0)
    lidar = Sensor("lidar", None, cam_mod)
    camera = _camera_sensor(cache["calib"], cam_mod, 0.0) if fused else None
    kf = CameraRecorder(kalman)
    manager = TrackManager(ws["track_management"])
    snaps = {}
    for rec in cache["frames"]:
        cnt = rec["frame"]
        kf.frame = cnt
        obs = []
        for z in rec["lidar_z"]:
            lidar.generate_measurement(cnt, z, obs)
        for track in manager.track_list:
            kf.predict(track)
            track.set_t(cnt * tracking_params.dt)
        assoc.associate_and_update(manager, obs, kf, lidar)
        if camera is not None and rec["cam"] is not None:
            cam_obs = []
            for centre in rec["cam"]:
                camera.generate_measurement(
                    cnt, np.array(centre) + rng.normal(0, CAMERA_NOISE_PX, 2), cam_obs)
            assoc.associate_and_update(manager, cam_obs, kf, camera)
        snaps[cnt] = {t.id: (np.asarray(t.x).reshape(-1).copy(), np.asarray(t.P).copy(), t.state)
                      for t in manager.track_list}
    return snaps, kf.events, camera


def _ellipse(ax, x, P, n_sigma=2.0, **kw):
    """Draw an n-sigma XY covariance ellipse in BEV axes (horizontal = y, vertical = x)."""
    cov = np.asarray(P)[:2, :2]
    vals, vecs = np.linalg.eigh(cov)
    # BEV plotting swaps axes: plotted (h, v) = (y, x).
    swap = np.array([[0, 1], [1, 0]])
    vecs = swap @ vecs
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_sigma * np.sqrt(vals[1]), 2 * n_sigma * np.sqrt(vals[0])
    ax.add_patch(Ellipse((x[1], x[0]), w, h, angle=angle, fill=False, **kw))


def _nearest_gt_xy(x, gt):
    if not gt:
        return None
    d = [np.hypot(x[0] - g[0], x[1] - g[1]) for g in gt]
    return min(d) if min(d) <= GATE_XY_M else None


def _camera_ray(camera, u, length=80.0):
    """Back-project pixel column u to a BEV ray (vehicle frame) from the camera centre."""
    s2v = np.asarray(camera.sens_to_veh)
    origin = s2v[:3, 3]
    direction = s2v[:3, :3] @ np.array([1.0, (camera.c_i - u) / camera.f_i, 0.0])
    direction /= np.linalg.norm(direction[:2])
    return origin, origin + length * direction


def _fov_lines(ax, camera, length=60.0):
    s2v = np.asarray(camera.sens_to_veh)
    o = s2v[:3, 3]
    for a in camera.fov:
        d = s2v[:3, :3] @ np.array([np.cos(a), np.sin(a), 0.0])
        ax.plot([o[1], o[1] + length * d[1]], [o[0], o[0] + length * d[0]],
                color="tab:purple", lw=0.8, ls=":")


def _bev_axes(ax, title):
    ax.set_xlabel("y (m, trái +)")
    ax.set_ylabel("x (m, phía trước)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    _setup_import_paths()
    cache = pickle.loads(args.cache.read_bytes())
    ws = load_workspace()
    lid, _, _ = replay(cache, ws, fused=False)
    fus, events, camera = replay(cache, ws, fused=True)
    frames = {r["frame"]: r for r in cache["frames"]}
    out = student_root() / "bonus"

    # Track with the most camera updates; its median-in-time camera update is shown.
    ids = [e["id"] for e in events]
    tid = max(set(ids), key=ids.count)
    tev = [e for e in events if e["id"] == tid]
    ev = tev[len(tev) // 2]
    f0 = ev["frame"]

    # ---------- Figure 1: BEV overview at frame f0 ----------
    rec = frames[f0]
    fig, ax = plt.subplots(figsize=(8, 8))
    _fov_lines(ax, camera)
    ax.scatter([g[1] for g in rec["gt"]], [g[0] for g in rec["gt"]], marker="s", s=70,
               facecolors="none", edgecolors="k", label="GT xe (valid)")
    ax.scatter([z[1] for z in rec["lidar_z"]], [z[0] for z in rec["lidar_z"]], marker="x",
               color="tab:blue", s=50, label="Đo LiDAR (tâm hộp)")
    for snaps, color, name in ((lid, "tab:orange", "lidar-only"), (fus, "tab:green", "fused")):
        first = True
        for i, (x, P, state) in snaps[f0].items():
            if state != "confirmed":
                continue
            ax.plot(x[1], x[0], "o", color=color, ms=5,
                    label=f"Track confirmed ({name}) + ellipse 2σ" if first else None)
            _ellipse(ax, x, P, color=color, lw=1.2)
            if name == "fused":
                ax.annotate(f"ID {i}", (x[1], x[0]), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color="darkgreen")
            first = False
    ax.plot([], [], color="tab:purple", ls=":", label="FOV camera FRONT")
    _bev_axes(ax, f"BEV frame {f0}: GT, đo LiDAR và track của hai mode (chấm tím = FOV camera)")
    pts = np.array([(g[0], g[1]) for g in rec["gt"]] + [(z[0], z[1]) for z in rec["lidar_z"]])
    ax.set_ylim(0, pts[:, 0].max() + 6)
    ax.set_xlim(pts[:, 1].max() + 8, pts[:, 1].min() - 8)
    ax.legend(loc="lower right", fontsize=8)
    # Inset: the ellipses are ~0.2 m wide, invisible at full scale.
    xt = fus[f0][tid][0]
    ins = ax.inset_axes([0.58, 0.30, 0.40, 0.40])
    for snaps, color in ((lid, "tab:orange"), (fus, "tab:green")):
        x, P, _ = snaps[f0][tid]
        ins.plot(x[1], x[0], "o", color=color, ms=4)
        _ellipse(ins, x, P, color=color, lw=1.2)
    gx = min(rec["gt"], key=lambda c: np.hypot(c[0] - xt[0], c[1] - xt[1]))
    ins.plot(gx[1], gx[0], "ks", mfc="none", ms=8)
    for z in rec["lidar_z"]:
        ins.plot(z[1], z[0], "x", color="tab:blue", ms=7)
    ins.set_xlim(xt[1] + 0.4, xt[1] - 0.4)
    ins.set_ylim(xt[0] - 0.4, xt[0] + 0.4)
    ins.set_aspect("equal")
    ins.tick_params(labelsize=6)
    ins.set_title(f"phóng to ID {tid} (±0.4 m): cam = lidar-only, xanh = fused", fontsize=7)
    ax.indicate_inset_zoom(ins, edgecolor="gray")
    fig.tight_layout()
    fig.savefig(out / "bev_overview.png", dpi=130)
    plt.close(fig)

    # ---------- Figure 2: one camera update, prior vs posterior ----------
    x0, P0 = np.asarray(ev["x0"]).reshape(-1), np.asarray(ev["P0"])
    x1, P1 = np.asarray(ev["x1"]).reshape(-1), np.asarray(ev["P1"])
    gt = rec["gt"]
    g = min(gt, key=lambda c: np.hypot(c[0] - x1[0], c[1] - x1[1]))
    lz = [z for z in rec["lidar_z"] if np.hypot(z[0] - x0[0], z[1] - x0[1]) < GATE_XY_M]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    o, e = _camera_ray(camera, ev["z"][0])
    ax.plot([o[1], e[1]], [o[0], e[0]], color="tab:purple", lw=1,
            label=f"Tia camera của pixel đo u = {ev['z'][0]:.1f}")
    ax.plot(g[1], g[0], "ks", mfc="none", ms=10, label="GT")
    for z in lz:
        ax.plot(z[1], z[0], "x", color="tab:blue", ms=9, label="Đo LiDAR")
    ax.plot(x0[1], x0[0], "o", color="tab:orange", label="Sau update LiDAR (prior của camera)")
    _ellipse(ax, x0, P0, color="tab:orange", lw=1.5)
    ax.plot(x1[1], x1[0], "o", color="tab:green", label="Sau update camera (posterior)")
    _ellipse(ax, x1, P1, color="tab:green", lw=1.5)
    ax.annotate("", xy=(x1[1], x1[0]), xytext=(x0[1], x0[0]),
                arrowprops=dict(arrowstyle="->", color="k"))
    e0 = np.hypot(x0[0] - g[0], x0[1] - g[1])
    e1 = np.hypot(x1[0] - g[0], x1[1] - g[1])
    shift = np.hypot(x1[0] - x0[0], x1[1] - x0[1])
    span = max(0.6, 3 * np.sqrt(max(P0[0, 0], P0[1, 1])), 1.5 * shift, 1.5 * e0)
    ax.set_xlim(x0[1] + span, x0[1] - span)
    ax.set_ylim(x0[0] - span, x0[0] + span)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("y (m, trái +)")
    ax.set_ylabel("x (m, phía trước)")
    ax.set_title(f"Track ID {tid}, frame {f0}: tác dụng của một camera update", fontsize=10)
    txt = (f"dịch state: {shift * 100:.1f} cm\n"
           f"lỗi XY tới GT: {e0 * 100:.1f} → {e1 * 100:.1f} cm\n"
           f"σx: {np.sqrt(P0[0, 0]) * 100:.1f} → {np.sqrt(P1[0, 0]) * 100:.1f} cm\n"
           f"σy: {np.sqrt(P0[1, 1]) * 100:.1f} → {np.sqrt(P1[1, 1]) * 100:.1f} cm")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "camera_update_effect.png", dpi=130)
    plt.close(fig)

    # ---------- Figure 3: timeline of the same track ----------
    fr = sorted(f for f in fus if tid in fus[f] and tid in lid[f])
    err_l = [_nearest_gt_xy(lid[f][tid][0], frames[f]["gt"]) for f in fr]
    err_f = [_nearest_gt_xy(fus[f][tid][0], frames[f]["gt"]) for f in fr]
    sy_l = [np.sqrt(lid[f][tid][1][1, 1]) for f in fr]
    sy_f = [np.sqrt(fus[f][tid][1][1, 1]) for f in fr]
    cam_frames = {e["frame"] for e in tev}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    a1.plot(fr, [np.nan if v is None else v * 100 for v in err_l], color="tab:orange", label="lidar-only")
    a1.plot(fr, [np.nan if v is None else v * 100 for v in err_f], color="tab:green", label="fused")
    a1.set_ylabel("lỗi XY tới GT gần nhất (cm)")
    both = [(a, b) for a, b in zip(err_l, err_f) if a is not None and b is not None]
    a1.set_title(f"Track ID {tid} (frame {fr[0]}–{fr[-1]}): lỗi XY trung bình "
                 f"lidar-only {np.mean([a for a, _ in both]) * 100:.1f} cm, "
                 f"fused {np.mean([b for _, b in both]) * 100:.1f} cm "
                 f"({len(both)} frame có GT ≤ 2 m)", fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.plot(fr, np.array(sy_l) * 100, color="tab:orange", label="lidar-only")
    a2.plot(fr, np.array(sy_f) * 100, color="tab:green", label="fused")
    a2.scatter(sorted(cam_frames), [0] * len(cam_frames), marker="|", color="tab:purple",
               label=f"frame có camera update ({len(cam_frames)})")
    a2.axvline(f0, color="gray", ls="--", lw=0.8)
    a2.set_ylabel("σy = sqrt(P[1,1]) cuối frame (cm)")
    a2.set_xlabel("frame")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "track_error_timeline.png", dpi=130)
    plt.close(fig)

    print(f"track {tid}: {len(tev)} camera updates, shown frame {f0}; "
          f"shift {shift * 100:.1f} cm, err {e0 * 100:.1f}->{e1 * 100:.1f} cm, "
          f"sigma_y {np.sqrt(P0[1, 1]) * 100:.1f}->{np.sqrt(P1[1, 1]) * 100:.1f} cm")


if __name__ == "__main__":
    main()
