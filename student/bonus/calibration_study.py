"""Bonus: camera extrinsic miscalibration study (RUBRIC §2, +4).

Rotates the FRONT camera ``veh_to_sens`` by a small yaw offset (about the camera
z axis, i.e. the tracker believes the camera points slightly to one side) and
re-runs fused tracking on the same cached LiDAR detections and the same seeded
camera noise. Reports camera innovation statistics, chi-square gate behaviour,
and RMSE / matches / ghosts / misses for each offset level.

Phase 1 (slow, once): run the provided detector over the segment and cache
LiDAR observations, valid GT centres, FRONT camera label centres and the FRONT
calibration. Phase 2 (fast): replay the tracking loop of ``run_lab.run`` for
lidar-only and for each yaw offset in fused mode.

Usage (from repo root, after ``set -a; source .env; set +a``):

    python student/bonus/calibration_study.py --config student/config/paths.yaml \
        --cache /tmp/day23_calib_cache.pkl

Outputs go to ``student/bonus/`` only; ``student/artifacts/`` is never touched.
The 0-degree row must reproduce ``student/artifacts/metrics_fused.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2

from fusion_lab import tracking_params
from fusion_lab.evaluation import tracking_counts
from fusion_lab.evaluation import valid_ground_truth
from fusion_lab.paths import student_root
from fusion_lab.scripts.run_lab import _load_paths_config
from fusion_lab.scripts.run_lab import _resolve_weights
from fusion_lab.scripts.run_lab import _setup_import_paths
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace

YAW_LEVELS_DEG = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
CAMERA_NOISE_PX = 0.5  # same as run_lab._front_observations


def build_cache(config_path: Path, cache_path: Path) -> dict:
    """Run detection once and store everything the tracking loop needs."""
    _setup_import_paths()
    cfg = _load_paths_config(config_path)
    ws = load_workspace()
    bev, det_pipe = ws["bev_mapping"], ws["detection_pipeline"]

    import torch
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    from fusion_lab.lidar_pcl import pcl_from_range_image

    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    frame_start, frame_end = int(cfg.get("frame_start", 0)), int(cfg.get("frame_end", 20))

    frames, cam_calib = [], None
    for cnt, frame in enumerate(WaymoDataFileReader(str(tfrecord))):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if cam_calib is None:
            c = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            cam_calib = {"intrinsic": list(c.intrinsic), "width": c.width, "height": c.height,
                         "transform": list(c.extrinsic.transform)}
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        lidar_z = [list(map(float, d[1:])) for d in detections
                   if all(lo <= v <= hi for v, (lo, hi) in
                          zip(d[1:3], (det_cfg.lim_x, det_cfg.lim_y)))]
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)
        gt = [(l.box.center_x, l.box.center_y, l.box.center_z) for l in labels]
        group = next((g for g in frame.camera_labels
                      if g.name == dataset_pb2.CameraName.FRONT), None)
        cam = None if group is None else [
            (l.box.center_x, l.box.center_y) for l in group.labels
            if l.type == label_pb2.Label.Type.TYPE_VEHICLE]
        frames.append({"frame": cnt, "lidar_z": lidar_z, "gt": gt, "cam": cam})
        print(f"cached frame {cnt}", flush=True)

    cache = {"segment": cfg["segment"], "calib": cam_calib, "frames": frames}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(cache))
    return cache


def _camera_sensor(calib: dict, cam_mod, yaw_deg: float) -> Sensor:
    """FRONT camera whose believed extrinsic is rotated by ``yaw_deg`` about its z axis."""
    ns = SimpleNamespace(intrinsic=calib["intrinsic"], width=calib["width"],
                         height=calib["height"],
                         extrinsic=SimpleNamespace(transform=calib["transform"]))
    sensor = Sensor("camera", ns, cam_mod)
    a = np.deg2rad(yaw_deg)
    offset = np.eye(4)
    offset[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    sensor.veh_to_sens = np.asmatrix(offset @ np.asarray(sensor.veh_to_sens))
    sensor.sens_to_veh = np.asmatrix(np.linalg.inv(np.asarray(sensor.veh_to_sens)))
    return sensor


class RecordingFilter(Filter):
    """Filter that logs innovation and d² of every accepted update before applying it."""

    def __init__(self, kalman_mod, assoc_mod):
        super().__init__(kalman_mod)
        self._assoc = assoc_mod
        self.camera, self.lidar = [], []

    def update(self, track, meas):
        gamma = np.asarray(self._k.innovation(track.x, meas)).reshape(-1)
        d2 = self._assoc.mahalanobis_distance(track, meas)
        (self.camera if meas.sensor.name == "camera" else self.lidar).append((gamma, d2))
        super().update(track, meas)


def replay(cache: dict, ws: dict, yaw_deg: float | None, seed: int = 0) -> dict:
    """Replay run_lab's per-frame loop. ``yaw_deg=None`` means lidar-only mode."""
    kalman, assoc, cam_mod = ws["kalman"], ws["association"], ws["camera_fusion"]
    rng = np.random.default_rng(seed)
    lidar = Sensor("lidar", None, cam_mod)
    camera = None if yaw_deg is None else _camera_sensor(cache["calib"], cam_mod, yaw_deg)
    kf = RecordingFilter(kalman, assoc)
    manager = TrackManager(ws["track_management"])
    gate_cam = chi2.ppf(tracking_params.gating_threshold, 2)
    totals = {"matches": 0, "sum_sq_err": 0.0, "ghosts": 0, "misses": 0, "confirmed": 0}
    nearest_d2 = []  # per in-FOV confirmed track: min camera d² before gating

    for rec in cache["frames"]:
        cnt = rec["frame"]
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
            for track in manager.track_list:
                if track.state == "confirmed" and cam_obs and camera.in_fov(track.x):
                    nearest_d2.append(min(assoc.mahalanobis_distance(track, m) for m in cam_obs))
            assoc.associate_and_update(manager, cam_obs, kf, camera)
        labels = [SimpleNamespace(box=SimpleNamespace(center_x=x, center_y=y, center_z=z))
                  for x, y, z in rec["gt"]]
        counts = tracking_counts(manager.track_list, labels)
        for key in totals:
            totals[key] += counts[key]

    n = len(cache["frames"])
    row = {
        "mode": "lidar" if yaw_deg is None else "fused",
        "yaw_offset_deg": yaw_deg,
        "rmse": float(np.sqrt(totals["sum_sq_err"] / totals["matches"])) if totals["matches"] else None,
        "matches": totals["matches"], "sum_sq_err": totals["sum_sq_err"],
        "ghost_track_frames": totals["ghosts"], "missed_gt_frames": totals["misses"],
        "mean_confirmed_tracks": totals["confirmed"] / n,
        "lidar_updates": len(kf.lidar),
        "lidar_innov_norm_mean_m": float(np.mean([np.linalg.norm(g) for g, _ in kf.lidar])),
    }
    if camera is not None:
        g = np.array([g for g, _ in kf.camera]).reshape(-1, 2)
        d2 = np.array([d for _, d in kf.camera])
        nd = np.array(nearest_d2)
        row.update({
            "camera_updates": len(kf.camera),
            "cam_innov_u_mean_px": float(g[:, 0].mean()) if len(g) else None,
            "cam_innov_u_std_px": float(g[:, 0].std()) if len(g) else None,
            "cam_innov_v_mean_px": float(g[:, 1].mean()) if len(g) else None,
            "cam_d2_accepted_mean": float(d2.mean()) if len(d2) else None,
            "confirmed_in_fov_checks": int(len(nd)),
            "nearest_d2_median": float(np.median(nd)) if len(nd) else None,
            "frac_nearest_outside_gate": float((nd >= gate_cam).mean()) if len(nd) else None,
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True,
                        help="Pickle cache path; keep it outside the repo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_import_paths()
    cache = (pickle.loads(args.cache.read_bytes()) if args.cache.is_file()
             else build_cache(args.config, args.cache))
    ws = load_workspace()
    rows = [replay(cache, ws, None, args.seed)]
    rows += [replay(cache, ws, yaw, args.seed) for yaw in YAW_LEVELS_DEG]

    out = student_root() / "bonus"
    meta = {"segment": cache["segment"], "frames": [cache["frames"][0]["frame"],
            cache["frames"][-1]["frame"]], "seed": args.seed,
            "camera_gate_chi2_995_k2": float(chi2.ppf(tracking_params.gating_threshold, 2)),
            "focal_px": cache["calib"]["intrinsic"][:2], "rows": rows}
    (out / "calibration_study.json").write_text(json.dumps(meta, indent=2))
    with (out / "calibration_study.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
