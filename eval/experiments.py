"""Step 8: ablations and robustness experiments on one sequence. Every number is measured, nothing is hand-set.

A  update interval          (VO frames between filter updates)
B  translation noise model  (sigma_t_rel)
C  IMU noise inflation
D  VO outages               (random OK frames removed, 3 seeds each)
E  front-end ablations      (CLAHE off, smaller central mask, no depth gating)
"""
from __future__ import annotations                                   # forward type hints
import copy                                                          # shallow copies
import json                                                          # metrics file
from dataclasses import replace                                      # config variants
import numpy as np                                                   # arrays

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK, LOST, VoConfig                        # status codes and LK config (front-end ablations)
from vo.run_vo import run as run_vo                                  # front end runner
from vio.run_vio import run_filter                                   # filter driver
from vio.eskf import FilterConfig                                    # filter config
from eval.ate import ate_rmse                                        # metrics


def score(out: dict, vo: dict, common: np.ndarray, segs: list) -> dict:   # metrics of a filter run
    gt = out["gt_cam"]                                               # oracle camera poses
    p_f, p_g = out["T_w_c"][common, :3, 3], gt[common, :3, 3]        # positions on common frames
    e_seg = []                                                       # per-segment rigid alignment errors
    for s in segs:                                                   # same segments as vision-only
        sel = common & (vo["map_id"] == s)                           # frames
        e_seg.append(ate_rmse(out["T_w_c"][sel, :3, 3], gt[sel, :3, 3], "se3")[2])   # errors
    return {"unaligned_m": float(np.sqrt(np.mean(np.sum((p_f - p_g) ** 2, axis=1)))),
            "global_se3_m": ate_rmse(p_f, p_g, "se3")[0],
            "per_segment_se3_m": float(np.sqrt(np.mean(np.concatenate(e_seg) ** 2)))}   # three views of ATE


def vision_only(vo: dict, gt: np.ndarray, common: np.ndarray, segs: list) -> float:   # oracle Sim3 per segment
    e = [ate_rmse(vo["T_w_c"][common & (vo["map_id"] == s), :3, 3], gt[common & (vo["map_id"] == s), :3, 3], "sim3")[2] for s in segs]   # errors
    return float(np.sqrt(np.mean(np.concatenate(e) ** 2)))           # RMSE


def row(label: str, r: dict, extra: str = "") -> str:                # markdown row
    return f"| {label} | {r['unaligned_m']:.3f} | {r['global_se3_m']:.3f} | {r['per_segment_se3_m']:.3f} | {extra} |"   # text


def main(name: str = "tumvi_room1") -> dict:                         # run everything
    res = PROJECT_ROOT / "results"                                   # output folder
    seq = Sequence(name)                                             # data
    vo = dict(np.load(res / f"vo_{name}.npz"))                       # baseline VO output
    base = run_filter(seq, vo, "estscale", verbose=False)            # baseline filter
    common = (vo["status"] == OK) & base["valid"]                    # frames used in every comparison
    segs = [int(s) for s in np.unique(vo["map_id"][common]) if (common & (vo["map_id"] == s)).sum() >= 20]   # usable segments
    gt = base["gt_cam"]                                              # oracle poses
    out = {"baseline": score(base, vo, common, segs), "vision_only_sim3_per_segment_m": vision_only(vo, gt, common, segs)}   # metrics
    md = ["| setting | ATE unaligned [m] | ATE global SE3 [m] | ATE per-segment SE3 [m] | note |", "|---|---|---|---|---|"]   # table
    md.append(row(f"baseline (update_every={FilterConfig().update_every}, sigma_t_rel={FilterConfig().sigma_t_rel})",
                  out["baseline"]))                                  # baseline row
    md.append(f"| vision-only, oracle Sim3 per segment | - | - | {out['vision_only_sim3_per_segment_m']:.3f} | reference |")   # VO row
    print(md[2]); print(md[3])                                       # progress

    def sweep(tag, items, make_cfg=None, make_vo=None, make_seq=None):   # generic sweep
        for lab, val in items:                                       # each value
            cfg = make_cfg(val) if make_cfg else None                # filter config
            v = make_vo(val) if make_vo else vo                      # VO data variant
            s = make_seq(val) if make_seq else seq                   # sequence variant (noise)
            r = score(run_filter(s, v, "estscale", cfg, verbose=False), vo, common, segs)   # run and score
            out[f"{tag}:{lab}"] = r                                  # store
            md.append(row(f"{tag}: {lab}", r))                       # table row
            print(md[-1])                                            # progress

    base_cfg = FilterConfig()                                        # defaults
    sweep("A update_every", [(str(n), n) for n in (1, 2, 5, 10, 20)], make_cfg=lambda n: replace(base_cfg, update_every=n))   # A
    sweep("B sigma_t_rel", [(str(x), x) for x in (0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.6)], make_cfg=lambda x: replace(base_cfg, sigma_t_rel=x))   # B

    def noisy_seq(k):                                                # shallow copy with scaled IMU noise
        s = copy.copy(seq)                                           # shares the data arrays
        s.noise = {**seq.noise, **{n: seq.noise[n] * k for n in ("acc_nd", "acc_rw", "gyr_nd", "gyr_rw")}}   # inflate
        return s                                                     # modified view
    sweep("C imu_noise_x", [(str(k), k) for k in (0.5, 1.0, 2.0, 4.0)], make_seq=noisy_seq)   # C

    for p in (0.3, 0.6, 0.9):                                        # D: outages
        errs = []                                                    # per-seed numbers
        for seed in (0, 1, 2):                                       # three seeds
            rng = np.random.default_rng(seed)                        # reproducible
            v = {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in vo.items()}   # copy
            drop = (v["status"] == OK) & (rng.random(len(v["status"])) < p)   # frames to remove
            v["status"][drop] = LOST                                 # pretend VO failed there
            errs.append(score(run_filter(seq, v, "estscale", None, verbose=False), vo, common, segs))   # score
        r = {k: float(np.mean([e[k] for e in errs])) for k in errs[0]}   # mean over seeds
        r["std_per_segment_se3_m"] = float(np.std([e["per_segment_se3_m"] for e in errs]))   # spread
        out[f"D vo_dropped:{p}"] = r                                 # store
        md.append(row(f"D vo_dropped: {int(p * 100)}% (3 seeds)", r, f"std {r['std_per_segment_se3_m']:.3f}"))   # row
        print(md[-1])                                                # progress

    variants = {"E clahe_off": VoConfig(use_clahe=False), "E mask_radius_180": VoConfig(mask_radius_px=180.0),
                "E no_depth_gating": VoConfig(depth_ratio_max=1e9, depth_ratio_min=0.0)}   # front-end ablations
    for lab, vcfg in variants.items():                               # run VO variants
        v, _ = run_vo(name, None, vcfg, verbose=False)               # new VO trajectory
        okv = (v["status"] == OK) & base["valid"]                    # frames this variant tracked
        sg = [int(s) for s in np.unique(v["map_id"][okv]) if (okv & (v["map_id"] == s)).sum() >= 20]   # segments
        if not sg:                                                   # nothing usable
            md.append(f"| {lab} | - | - | - | no usable segment |")   # row
            continue                                                 # next
        f = score(run_filter(seq, v, "estscale", None, verbose=False), v, okv, sg)   # fused score on its own frames
        vo_only = vision_only(v, gt, okv, sg)                        # its vision-only score
        out[lab] = {**f, "vision_only_sim3_per_segment_m": vo_only, "ok_fraction": float(okv.mean()), "n_segments": len(sg)}   # store
        md.append(row(lab, f, f"VO-only {vo_only:.3f}; OK {100 * okv.mean():.0f}%; {len(sg)} segs"))   # row
        print(md[-1])                                                # progress

    (res / f"step8_experiments_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")   # save table
    with open(res / f"metrics_step8_{name}.json", "w", encoding="utf-8") as f:   # save metrics
        json.dump(out, f, indent=2)                                  # write
    return out                                                       # metrics


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
