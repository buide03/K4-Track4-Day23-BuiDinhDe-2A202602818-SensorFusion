# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Bui Dinh De
- MSSV: 2A202602818
- Email: buidinhde2003tkk@gmail.com
- Link repo (fork): https://github.com/buide03/K4-Track4-Day23-BuiDinhDe-2A202602818-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`):

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519/16/222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse` = 0.1503 m, `matches` = 502, `sum_sq_err` = 11.3436, `ghost_track_frames` = 0, `missed_gt_frames` = 239, `mean_confirmed_tracks` = 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse` = 0.1359 m, `matches` = 502, `sum_sq_err` = 9.2668, `ghost_track_frames` = 0, `missed_gt_frames` = 239, `mean_confirmed_tracks` = 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  - **Số ghép và vòng đời giống hệt nhau:** cả hai mode có `matches` = 502, `ghost_track_frames` = 0,
    `missed_gt_frames` = 239, `mean_confirmed_tracks` = 2.5226. So từng frame trong `grade_run.log`,
    `det_*`, `valid_gt`, `confirmed`, `matches`, `ghosts`, `misses` trùng nhau ở cả 199 frame. Lý do: track chỉ được
    khởi tạo, cộng/trừ score và xoá trong lượt LiDAR (`handle_updated_track` / `manage_tracks` bỏ qua camera),
    nên camera không thể thêm hay bớt track. Vì vậy hai RMSE được tính trên **cùng** tập 502 cặp và so sánh trực tiếp được.
  - **Khác biệt nằm ở độ chính xác vị trí:** `sum_sq_err` giảm từ 11.3436 xuống 9.2668, RMSE giảm từ
    0.1503 m xuống 0.1359 m (-0.0145 m, -9.6%). Fused có sai số nhỏ hơn ở 137 frame,
    lớn hơn ở 58 frame; 4 frame bằng nhau (frame 0–3, chưa có track confirmed nên `matches` = 0). Camera update
    (có 509 update được nhận) bổ sung ràng buộc theo hướng ngang/dọc của ảnh cho các track trong FOV camera,
    nhưng không đo được độ sâu. Mức chênh `rmse_fused − rmse_lidar` = -0.0145 m, nằm trong giới hạn ≤ 0.05 m của rubric.
  - **Ghost/miss do detector và confirmation quyết định, không do fusion:** `ghost_track_frames` = 0 dù detector có `fp` = 16,
    vì false positive không đủ hit để vượt `score > 0.8`. `missed_gt_frames` = 239 = 222 lần detector bỏ sót
    (`fn`, recall 0.7004) + 17 frame có detection nhưng track chưa confirmed hoặc chưa ghép được
    (`det_tp` − `matches` = 519−502 = 17). Ví dụ: hai xe ở frame 0 chỉ được confirm từ frame 4. Camera không giảm được số này.
  - **Giới hạn khi đọc kết quả:** phép đo camera là tâm hộp 2D ground truth cộng nhiễu (σ = 0.5 px), không phải output của một camera
    detector, nên RMSE fused tốt hơn **không** chứng minh camera perception tốt. Ngay ở calibration gốc, innovation camera đã có bias
    (γᵤ ≈ -11.9 px, γᵥ ≈ 7.6 px, std ≈ 27 px; xem Bonus), và innovation LiDAR trung bình tăng từ
    0.093 m (lidar-only) lên 0.157 m (fused). Nghĩa là hai sensor không hoàn toàn nhất quán, và mức cải thiện
    phụ thuộc vào calibration: chỉ cần lệch yaw 0.5° là fused đã tệ hơn lidar-only (RMSE 0.1936 m).

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   **LiDAR:** `z` là tâm hộp 3D `(x, y, z)`, cột 3×1, đơn vị mét. `R = diag(0.1², 0.1², 0.1²)` m²
   (`sigma_lidar_*` trong `tracking_params.py`). `h(x)` chỉ là phép biến đổi vị trí sang hệ sensor
   (`R_s p + t`), nên tuyến tính: `H` là ma trận hằng 3×6, gồm khối xoay và các cột vận tốc bằng 0.

   **Camera:** `z` là tâm hộp 2D `(u, v)`, cột 2×1, đơn vị pixel, tạo trong
   `build_camera_measurement` (`camera_fusion.py`). `R = diag(5², 5²)` px² (`sigma_cam_i/j`).
   `h(x)` là phép chiếu pinhole `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`
   (`camera_measurement_prediction`), phi tuyến vì chia cho độ sâu `x_s`. Vì vậy `H` là Jacobian 2×6
   tính lại tại mỗi `x` (`Sensor.get_H` trong `sensors.py`). Camera không đo được độ sâu,
   nên chủ yếu sửa vị trí theo phương ngang và dọc của ảnh.

   `ekf_update` (`kalman.py`) dùng chung một công thức cho cả hai sensor: `γ = z − h(x)`,
   `S = H P Hᵀ + R`, `K = P Hᵀ S⁻¹`. Khác biệt chỉ đến từ `meas.sensor.get_hx/get_H` và `meas.R`.
   Vì `S` có đơn vị m² ở LiDAR và px² ở camera, Kalman gain tự cân đối mức tin vào từng sensor.

2. Vì sao cần gating Mahalanobis trước khi gán?

   Khoảng cách Euclid không tính đến độ bất định và không so sánh được giữa m với px.
   `d² = γᵀ S⁻¹ γ` (`mahalanobis_distance` trong `association.py`) chuẩn hoá innovation bằng `S`:
   track mới có `P` lớn chấp nhận phép đo ở xa hơn, track đã hội tụ thì chặt hơn, và kết quả không có đơn vị.
   Nếu cặp ghép đúng thì `d² ~ χ²(k)` với `k = dim_meas`, nên `chi2_gate` loại cặp có
   `d² ≥ chi2.ppf(0.995, k)`, tức 12.84 với LiDAR (k=3) và 10.60 với camera (k=2).
   Chỉ khoảng 0.5% cặp đúng bị loại nhầm.

   Gating đứng trước bước gán greedy (`association_cost_matrix` đặt `inf` cho cặp bị loại), vì
   greedy luôn lấy ô nhỏ nhất. Không có cổng thì một false positive ở xa vẫn bị ghép vào track nào đó,
   kéo track lệch đi và che mất việc sinh track mới. Trước cả Mahalanobis còn có kiểm tra FOV
   (`meas.sensor.in_fov`): track ở sau camera không được chiếu, vì `h(x)` sẽ chia cho độ sâu ≤ 0.
   Kết quả lần chạy chấm: `ghost_track_frames = 0` ở cả hai mode dù detector có `fp = 16`.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   **Track-then-fuse.** Chỉ có một danh sách track. Mỗi frame, `run_lab.py` predict một lần cho mọi track,
   chạy AssocL (`associate_and_update(..., lidar_sensor)`), rồi ở mode fused chạy AssocC
   (`associate_and_update(..., camera_sensor)`) trên **cùng** các track đó. Không có bước nào gộp
   dữ liệu thô của hai sensor trước khi detect hay track.

   Trên `grade_run.log`: có 398 record = 199 frame × 2 mode, mỗi `(mode, frame)` đúng một dòng.
   Các record `lidar` đứng trước, `fused` sau. So từng frame giữa hai mode thì
   `det_tp/det_fp/det_fn/valid_gt/confirmed/matches/ghosts/misses` **trùng nhau ở cả 199 frame**;
   chỉ `sum_sq_err` khác. Điều này chứng tỏ detection và vòng đời track do LiDAR quyết định trước,
   còn camera chỉ được fuse vào trạng thái của các track đã có. Tổng cộng `matches` = 502,
   `ghost_track_frames` = 0, `missed_gt_frames` = 239 ở cả hai mode; RMSE lidar 0.1503 m, fused 0.1359 m.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Khi đó `veh_to_sens` hoặc intrinsics sai, nên `h(x)` chiếu track sang sai pixel một cách **có hệ thống**.
   Triệu chứng dự kiến: innovation camera `γ = z − h(x)` có trung bình khác 0 và cùng hướng qua nhiều
   frame và nhiều track, thay vì nhiễu trắng quanh 0 với độ lệch cỡ `sigma_cam = 5` px. `d²` tăng,
   nhiều cặp camera bị `chi2_gate` loại hơn (track ở mép FOV có thể bị gán sai). Những cặp vẫn qua cổng
   sẽ kéo state lệch khỏi vị trí LiDAR, nên LiDAR innovation ở frame sau cũng lớn hơn và RMSE fused
   tăng lên so với lidar. Thí nghiệm bonus (mục Bonus bên dưới) xác nhận điều này: lệch yaw 1° làm trung bình γᵤ dịch
   ≈ 30 px, 41% track bị gate loại, và RMSE fused tăng lên 0.2562 m. Lệch nhỏ (≤ 0.5°)
   lọt qua cổng; lệch lớn (≥ 2°) bị chặn gần hết nên tracker trở về gần lidar-only.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi một frame không có detection, `meas_list` rỗng nên không thể suy ra sensor từ `meas.sensor`.
   Nhưng lượt LiDAR rỗng vẫn phải chạy vòng đời: các track nằm trong FOV LiDAR bị tính là miss
   (`score −= 1/6`) và có thể bị xoá. Vì vậy `associate_and_update` luôn gọi
   `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`, và `manage_tracks`
   (`manager.py`) dựa vào `sensor.name` để biết đây là lượt LiDAR hay camera. Test
   `test_empty_lidar_frame_scores_then_deletes_exhausted_track` kiểm tra đúng trường hợp này.

   LiDAR cho vị trí 3D đầy đủ, nên một phép đo LiDAR chưa ghép đủ để khởi tạo `x`
   (`init_track_state_from_meas`), và hit/miss LiDAR là bằng chứng về sự tồn tại của xe. Camera chỉ có
   `(u, v)` mà không có độ sâu, nên không khởi tạo được track 3D. Trong lab, phép đo camera lại là
   tâm hộp ground truth cộng nhiễu, nên cho camera cộng điểm hay sinh track thì tracker sẽ dựa vào nhãn GT.
   Vì vậy `handle_updated_track` chỉ chấm điểm khi `sensor.name == "lidar"`, còn `manage_tracks` return
   ngay với camera: camera chỉ tinh chỉnh `x`, `P` qua `ekf_update`.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Theo `track_management.py`, với `window = 6`:
   - Khởi tạo: `score = 1/6`, `state = "initialized"`.
   - Hit LiDAR: `score = min(1, score + 1/6)`; track chưa confirmed thì chuyển sang `"tentative"`.
   - **Xác nhận:** `score > 0.8` thì thành `"confirmed"`, nghĩa là cần 1 lần sinh + 4 hit liên tiếp
     (5/6 ≈ 0.83). Log khớp với điều này: hai xe ở frame 0 có `confirmed = 0` ở frame 0–3 và
     `confirmed = 2` từ frame 4.
   - **Miss** chỉ được tính khi track nằm trong FOV LiDAR (`score −= 1/6`). Track đã confirmed **không bị
     hạ trạng thái** khi miss; chỉ score giảm.
   - **Xoá** khi một trong ba điều kiện đúng (OR): `P[0,0] > 9` hoặc `P[1,1] > 9` (độ lệch chuẩn vị trí
     > 3 m, track đã lạc); confirmed và `score < 0.6`; chưa confirmed và `score <= 0`.
     Ví dụ: một track confirmed có score 1.0 chịu được 2 miss liên tiếp (0.83, 0.67) và bị xoá ở miss thứ 3 (0.5).
     Một track mới bị miss ngay frame sau thì score về 0 và bị xoá.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- **Phân tích calibration camera (+4).** Script: `student/bonus/calibration_study.py`; số liệu:
  `student/bonus/calibration_study.json`, `student/bonus/calibration_study.csv`.

  *Cách làm:* quay `veh_to_sens` của camera FRONT thêm một góc yaw δ quanh trục z của camera, tức tracker
  tin rằng camera hướng lệch δ so với thực tế. Sau đó chạy lại vòng lặp tracking của `run_lab.run`
  trên **cùng** detection LiDAR (đã cache từ một lần chạy detector) và **cùng** nhiễu camera (seed 0).
  Các mức δ = 0, 0.1, 0.25, 0.5, 1, 2, 5°. Frame 0–198, segment như lần chấm điểm.
  Ở δ = 0, mode lidar-only và fused tái tạo đúng `metrics_lidar.json` / `metrics_fused.json`:
  matches/ghost/miss khớp tuyệt đối, RMSE và SSE chỉ lệch ~1e-15 do thứ tự cộng số thực.
  Cổng camera là `chi2.ppf(0.995, 2)` = 10.60; f = 2083.1 px.
  "median d² gần nhất" là d² trước gating, từ mỗi track confirmed trong FOV tới phép đo camera gần nhất;
  "% ngoài cổng" là tỉ lệ track mà ngay cả phép đo gần nhất cũng vượt ngưỡng.

| Mode / lệch yaw | RMSE (m) | matches | ghost | miss | camera update được nhận | γᵤ TB ± std (px) | γᵥ TB (px) | d² TB (cặp được nhận) | median d² gần nhất | % ngoài cổng | ‖γ LiDAR‖ TB (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar-only | 0.1503 | 502 | 0 | 239 | — | — | — | — | — | — | 0.093 |
| fused 0.0° | 0.1359 | 502 | 0 | 239 | 509 | -11.9 ± 27.1 | 7.6 | 2.25 | 1.64 | 1.2% | 0.157 |
| fused 0.1° | 0.1342 | 502 | 0 | 239 | 508 | -8.3 ± 27.2 | 7.5 | 1.69 | 1.01 | 1.4% | 0.142 |
| fused 0.25° | 0.1464 | 502 | 0 | 239 | 508 | -2.9 ± 27.0 | 7.4 | 1.43 | 0.91 | 1.4% | 0.134 |
| fused 0.5° | 0.1936 | 502 | 0 | 239 | 504 | 5.8 ± 26.8 | 7.3 | 2.37 | 2.02 | 2.0% | 0.154 |
| fused 1.0° | 0.2562 | 502 | 0 | 239 | 301 | 18.6 ± 33.3 | 9.1 | 5.56 | 8.35 | 41.2% | 0.184 |
| fused 2.0° | 0.1551 | 502 | 0 | 239 | 30 | -20.5 ± 83.3 | -12.1 | 1.56 | 42.28 | 97.5% | 0.095 |
| fused 5.0° | 0.1542 | 499 | 1 | 242 | 15 | -3.6 ± 84.7 | -17.8 | 2.96 | 68.04 | 99.6% | 0.095 |

  *Nhận xét:*
  - **Triệu chứng trên innovation là một độ lệch có hệ thống**, không phải nhiễu lớn hơn: khi δ tăng từ 0 lên 1°,
    trung bình γᵤ dịch từ -11.9 lên +18.6 px
    (≈ 30 px/°, gần với lý thuyết f·tan 1° ≈ 36 px/°),
    trong khi độ lệch chuẩn gần như không đổi (≈ 27 px). Phần dịch đo được nhỏ hơn lý thuyết vì
    chỉ tính trên các cặp được nhận và state đã bị camera kéo một phần. γᵥ gần như không đổi vì lệch yaw chỉ ảnh hưởng trục ngang.
  - **Ngay ở δ = 0 innovation camera đã có bias** (γᵤ ≈ -11.9 px, γᵥ ≈ 7.6 px)
    và std ≈ 27 px, lớn hơn nhiều so với nhiễu mô phỏng σ = 0.5 px (`run_lab.py`) và σ = 5 px trong `R`.
    Giả thuyết của tôi (chưa kiểm chứng): tâm hộp 2D ground truth không trùng với hình chiếu của tâm hộp 3D, cộng với
    sai số vị trí track. Điều này giải thích vì sao lệch nhỏ 0.1–0.25° (cùng chiều bù bias) lại cho d² nhỏ hơn
    và RMSE ở 0.1° (0.1342 m) hơi tốt hơn 0° (0.1359 m). Đây là hiệu ứng bù trừ, không phải
    calibration gốc sai.
  - **Gating không chặn được lệch nhỏ (≤ 0.5°):** d² vẫn dưới ngưỡng, gần như mọi camera update vẫn được nhận
    (504 ở 0.5° so với 509 ở 0°), nên bias đi thẳng vào state.
    RMSE tăng lên 0.1936 m ở 0.5°, tệ hơn lidar-only (0.1503 m), và innovation LiDAR trung bình
    cũng tăng (0.154 m so với 0.093 m ở lidar-only):
    camera kéo state lệch khỏi LiDAR, rồi LiDAR kéo lại ở frame sau.
  - **Vùng nguy hiểm nhất là 1°:** chỉ 41% track bị chặn, các cặp còn lại vẫn mang bias
    ~19 px nên RMSE xấu nhất: 0.2562 m (+70% so với lidar-only).
    Đây là mức vượt ngưỡng 0.05 m của rubric (fused − lidar).
  - **Gating chặn được lệch lớn (≥ 2°):** median d² = 42.3 và 68.0, vượt xa ngưỡng 10.60,
    nên 97.5% / 99.6% track bị loại; chỉ còn 30 / 15 update.
    Tracker gần như trở về lidar-only: RMSE 0.1551 / 0.1542 m, ‖γ LiDAR‖ ≈ 0.095 m.
    Các update còn lại ở 5° có std γᵤ ≈ 85 px, nhiều khả năng là ghép nhầm sang xe khác.
    Ở 5° lần đầu xuất hiện ảnh hưởng tới vòng đời: matches 499 thay vì 502, 1 ghost frame.
    Camera không đổi score trực tiếp, nhưng state bị kéo lệch làm LiDAR association ở frame sau thay đổi.
  - **Kết luận:** gating χ² chỉ bảo vệ khi lỗi calibration làm innovation **lớn hơn nhiều** so với `S`. Lỗi nhỏ có hệ thống
    nằm trong cổng và làm hỏng estimate một cách âm thầm. Dấu hiệu nên theo dõi là **trung bình** của innovation
    camera (khác 0, cùng hướng qua nhiều frame và track) cùng tỉ lệ cặp bị gate loại tăng dần, chứ không phải độ lớn của từng γ.
  - Trong lab, matches/ghost/miss gần như không đổi ở mọi mức vì vòng đời track chỉ do LiDAR quyết định;
    lỗi calibration chủ yếu thể hiện qua RMSE.

- **Trực quan hoá track và đo trên BEV (+3).** Script: `student/bonus/visualize_tracks.py`. Script phát lại vòng lặp
  tracking của `run_lab.run` (lidar-only và fused, calibration gốc, seed 0) trên cùng cache detection của bonus calibration
  và ghi lại state/covariance trước và sau mỗi camera update. Ảnh nằm trong `student/bonus/`:
  - `bev_overview.png`: frame 99, gồm GT (ô vuông), đo LiDAR (×), track confirmed của hai mode (cam = lidar-only,
    xanh = fused) kèm ellipse 2σ, và FOV camera FRONT (chấm tím). Ba xe ID 0, 1, 10 đều được bám ở cả hai mode
    (đúng với `ghost_track_frames = 0`). Ellipse chỉ rộng ~0.2 m nên có ô phóng to quanh ID 0: ellipse lidar-only
    gần tròn, còn ellipse fused hẹp lại theo phương ngang.
  - `camera_update_effect.png`: **tác dụng của một camera update** lên track ID 0 ở frame 99 (track có nhiều camera
    update nhất; frame được chọn là lần update ở giữa chuỗi, không chọn frame đẹp nhất). Sau update LiDAR,
    state nằm ngay trên đo LiDAR với ellipse gần tròn (σx = σy ≈ 9.9 cm). Camera update dịch state 14.9 cm về phía
    tia chiếu ngược của pixel đo (u = 529.5), lỗi XY tới GT giảm từ 16.0 xuống 11.8 cm. σy giảm từ 9.9 xuống 4.9 cm,
    còn σx (gần như dọc theo tia, tức độ sâu) chỉ giảm từ 9.9 xuống 9.7 cm. Ellipse posterior dài ra **dọc theo tia camera**:
    camera ràng buộc vị trí theo hướng vuông góc với tia nhưng không đo được độ sâu. Đây chính là cấu trúc của
    Jacobian `H` 2×6 (một pixel tương ứng với cả một tia).
  - `track_error_timeline.png`: cùng track ID 0 qua frame 0–198. Lỗi XY tới GT gần nhất trung bình là 13.3 cm (lidar-only)
    so với 12.7 cm (fused). Fused không tốt hơn ở mọi frame: frame 0–45 fused hơi tệ hơn (12.2 → 13.7 cm), frame 75–110 tốt hơn rõ (18.7 → 12.4 cm).
    σy cuối frame của fused ổn định ở ~4–5 cm so với ~9.9 cm của lidar-only, vì track này có camera update ở cả 199 frame.
    Lưu ý: σ nhỏ hơn không đồng nghĩa với lỗi nhỏ hơn tương ứng; innovation camera có bias (xem bảng calibration ở trên),
    nên filter fused "tự tin" hơn mức thực tế.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude, Anthropic).
- Dùng cho phần nào (hàm, câu hỏi, debug): giải thích toán EKF, gating χ², mô hình pinhole và vòng đời track; cùng viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); debug lỗi `float()` trên ma trận 1×1 ở `mahalanobis_distance`; kiểm tra invariant của artifact; điền phần số liệu từ `metrics.json`; **soạn nháp 6 câu trả lời Parts E–H và đoạn giải thích khác biệt hai mode**; bonus calibration: viết script `student/bonus/calibration_study.py`, chạy thí nghiệm và soạn nháp bảng + nhận xét từ `calibration_study.json`; bonus trực quan hoá: viết script `student/bonus/visualize_tracks.py`, tạo 3 ảnh BEV và soạn nháp chú thích. Tôi đã đọc, đối chiếu với code/log và chịu trách nhiệm giải thích.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` (128 passed); chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198; đối chiếu số trong báo cáo với `metrics.json` và `grade_run.log`; đối chiếu công thức với `docs/HUONG_DAN_KY_THUAT.md` và tính lại các ví dụ số trong test; bonus: kiểm tra mức lệch 0° tái tạo đúng `metrics_lidar.json`/`metrics_fused.json`, rồi chạy lại lệnh chấm chính thức sau thí nghiệm.

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [ ] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
