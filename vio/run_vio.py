"""Drive the ESKF with IMU samples and VO relative poses; modes for validation and the real run.

Modes
  imuonly   : IMU propagation only from the GT start (dead-reckoning sanity check)
  gtrel     : feed exact GT relative poses (filter plumbing sanity check)
  gtscale   : VO rotation + direction with GT-derived translation magnitude (4a, plumbing only)
  estscale  : VO relative poses in VO units, scale estimated by the filter (4b, the real result)
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import numpy as np                                                   # arrays

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend import OK                                           # status code
from vio.eskf import ESKF, FilterConfig                              # the filter


def gt_velocity(seq: Sequence, t_ns: int, h_ns: int = 10_000_000) -> np.ndarray:   # GT IMU velocity by central difference
    p, _ = seq.gt.interp(np.array([t_ns - h_ns, t_ns + h_ns]))       # positions around t
    return (p[1] - p[0]) / (2 * h_ns * 1e-9)                         # m/s


def run_filter(seq: Sequence, vo: dict, mode: str = "estscale", cfg: FilterConfig | None = None, verbose: bool = True):
    cfg = cfg or FilterConfig()                                      # configuration
    t_c, T_vo, st, mid = vo["t_ns"], vo["T_w_c"], vo["status"], vo["map_id"]   # VO output
    n = len(t_c)                                                     # number of frames
    ok_idx = np.where(st == OK)[0]                                   # trusted VO frames
    k0 = int(ok_idx[0])                                              # filter starts here
    t_start = int(max(t_c[k0], seq.gt.t_ns[0] + 20_000_000))         # inside the GT span
    gt_cam = seq.T_w_c_gt(t_c)                                       # oracle camera poses (GT)
    p_gt, R_gt = seq.gt.interp(np.array([t_start]))                  # GT IMU pose at start
    v0 = gt_velocity(seq, t_start)                                   # GT velocity at start
    est_scale = mode == "estscale"                                   # scale estimated only in 4b
    f = ESKF(p_gt[0], v0, R_gt.as_matrix()[0], seq.noise, seq.calib.T_cam_imu, cfg,
             s=1.0, estimate_scale=est_scale)                        # filter
    imu_t, imu_w, imu_a = seq.imu.t_ns, seq.imu.gyro, seq.imu.acc    # IMU arrays
    frame_ids = np.where(t_c >= t_start)[0]                          # frames processed by the filter
    ev_t = np.concatenate([imu_t[imu_t > t_start], t_c[frame_ids]])  # event times
    ev_f = np.concatenate([np.full(int((imu_t > t_start).sum()), -1), frame_ids])   # -1 = IMU sample
    order = np.argsort(ev_t, kind="stable")                          # chronological
    ev_t, ev_f = ev_t[order], ev_f[order]                            # sorted
    out_T = np.tile(np.eye(4), (n, 1, 1))                            # fused camera poses
    out_valid = np.zeros(n, bool)                                    # filter state exists
    out_s = np.full(n, np.nan)                                       # scale estimate
    out_ss = np.full(n, np.nan)                                      # scale sigma
    out_bg, out_ba = np.full((n, 3), np.nan), np.full((n, 3), np.nan)   # bias estimates
    clone_frame, clone_map = None, None                              # VO frame of the clone
    scale_ready = not est_scale                                      # 4a: no scale to initialise
    num = den = 0.0                                                  # scale-init accumulators
    n_pairs = 0                                                      # pairs used for scale init
    n_upd = n_skip = 0                                               # update counters
    since_clone = 0                                                  # OK frames since the last clone
    t_prev = t_start                                                 # current filter time
    for t_now, fr in zip(ev_t, ev_f):                                # event loop
        dt = (int(t_now) - t_prev) * 1e-9                            # step length
        if dt > 0:                                                   # propagate to the event time
            tm = 0.5 * (int(t_now) + t_prev)                         # midpoint time
            ia = int(np.clip(np.searchsorted(imu_t, tm) - 1, 0, len(imu_t) - 2))   # bracketing sample
            w = (tm - imu_t[ia]) / float(imu_t[ia + 1] - imu_t[ia])  # interpolation weight
            f.propagate(dt, imu_w[ia] * (1 - w) + imu_w[ia + 1] * w, imu_a[ia] * (1 - w) + imu_a[ia + 1] * w)   # IMU step
            t_prev = int(t_now)                                      # advance
        if fr < 0:                                                   # IMU sample only
            continue                                                 # next event
        k = int(fr)                                                  # camera frame index
        if mode != "imuonly" and st[k] == OK:                        # a trusted VO frame
            if f.has_clone() and clone_map != mid[k]:                # new map segment: relative link broken
                f.drop_clone()                                       # forget the clone
                if est_scale:                                        # scale is per map
                    scale_ready, num, den, n_pairs = False, 0.0, 0.0, 0   # re-initialise
                    f.set_scale(f.s, 1.0)                            # reset uncertainty
            since_clone += 1                                         # OK frames since the clone
            if f.has_clone() and since_clone < cfg.update_every:     # baseline still too short
                pass                                                 # propagate only
            elif f.has_clone():                                      # relative measurement available
                if mode == "gtrel":                                  # exact GT relative pose
                    Ti = np.linalg.inv(gt_cam[clone_frame]) @ gt_cam[k]   # T_ci_cj
                    R_m, t_m = Ti[:3, :3], Ti[:3, 3]                 # rotation, metric translation
                else:                                                # VO relative pose
                    Ti = np.linalg.inv(T_vo[clone_frame]) @ T_vo[k]  # T_ci_cj in VO units
                    R_m, t_m = Ti[:3, :3], Ti[:3, 3]                 # rotation, VO-unit translation
                    if mode == "gtscale":                            # GT magnitude, VO direction (crutch)
                        d = np.linalg.norm(gt_cam[k][:3, 3] - gt_cam[clone_frame][:3, 3])   # true displacement
                        nrm = np.linalg.norm(t_m)                    # VO displacement (VO units)
                        t_m = t_m / nrm * d if nrm > 1e-9 else t_m   # rescale
                if est_scale and not scale_ready:                    # collect scale-init statistics
                    _, t_pred = f.predict_relative()                 # IMU-predicted displacement
                    num += float(t_pred @ t_m)                       # <pred, vo>
                    den += float(t_m @ t_m)                          # <vo, vo>
                    n_pairs += 1                                     # count
                    ok_u, _ = f.update_relative(R_m, t_m, use_t=False)   # rotation-only update
                    if n_pairs >= cfg.n_scale_init and den > 1e-12 and num > 0:   # enough excitation
                        s0 = num / den                               # least-squares scale
                        f.set_scale(s0, cfg.scale_init_rel_sigma * s0)   # initialise
                        scale_ready = True                           # now use the full update
                else:                                                # full 6-D update
                    ok_u, _ = f.update_relative(R_m, t_m, use_t=True)   # update
                n_upd += int(ok_u)                                   # applied
                n_skip += int(not ok_u)                              # not applied
            if (not f.has_clone()) or since_clone >= cfg.update_every:   # (re)clone at update frames
                f.clone()                                            # new clone at this frame
                clone_frame, clone_map = k, int(mid[k])              # remember
                since_clone = 0                                      # restart counting
        out_T[k] = f.cam_pose()                                      # fused camera pose
        out_valid[k] = True                                          # valid
        out_s[k] = f.s if (est_scale and scale_ready) else np.nan    # scale (NaN until initialised)
        out_ss[k] = np.sqrt(f.P[15, 15]) if (est_scale and scale_ready) else np.nan   # scale sigma
        out_ba[k], out_bg[k] = f.ba, f.bg                            # biases
    out_valid &= (t_c >= seq.gt.t_ns[0]) & (t_c <= seq.gt.t_ns[-1])  # frames without ground truth are never scored
    if verbose:                                                      # summary
        print(f"[{mode}] frames {int(out_valid.sum())}, updates applied {n_upd}, "
              f"noise-inflated (chi2 gate exceeded) {f.n_gated}")    # counts
    return dict(t_ns=t_c, T_w_c=out_T, valid=out_valid, s=out_s, s_sigma=out_ss, ba=out_ba, bg=out_bg,
                gt_cam=gt_cam, n_upd=n_upd, n_gated=f.n_gated)       # results


def main():                                                          # CLI entry
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--dataset", default="tumvi_room1")              # dataset key
    ap.add_argument("--mode", default="estscale", choices=["imuonly", "gtrel", "gtscale", "estscale"])   # mode
    ap.add_argument("--frontend", default="lk", choices=["lk", "orb", "sift"])   # which VO output feeds the filter
    a = ap.parse_args()                                              # parse
    seq = Sequence(a.dataset)                                        # data
    from vo.run_vo import out_path                                   # VO result naming
    vo = dict(np.load(out_path(a.dataset, a.frontend)))              # VO output
    out = run_filter(seq, vo, a.mode)                                # run
    tag = "" if a.frontend == "lk" else f"{a.frontend}_"             # keep the LK file names unchanged
    np.savez(PROJECT_ROOT / "results" / f"vio_{tag}{a.mode}_{a.dataset}.npz", **out)   # save
    ok = out["valid"]                                                # valid frames
    err = np.linalg.norm(out["T_w_c"][ok, :3, 3] - out["gt_cam"][ok, :3, 3], axis=1)   # unaligned position error
    print(f"[{a.mode}] unaligned position RMSE vs GT: {np.sqrt(np.mean(err ** 2)):.4f} m; final error {err[-1]:.3f} m")   # summary


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
