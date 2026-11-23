#!/usr/bin/env python3
# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Part of cryo-savanna; a copy of it goes into every project that runs its membrain-seg jobs, and travels with
# it (with runner_common.py, whose helpers it uses).
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
"""Run MemBrain-seg (membrane segmentation of tomograms, https://github.com/teamtomo/membrain-seg) as a RELION
External job.

cryo-savanna's "Additional external jobs" menu makes External jobs whose executable is

    python3 .cryo-savanna/membrain_runner.py segment

(a copy of this file in the project, beside a copy of runner_common.py, the helpers it shares with the other
runners). RELION runs it as ``... --o <job dir> [--in_* F]... --j <threads> --<option> <value> ...``, the
options after ``--j`` being the job's settings (cryo-savanna writes them into "Additional arguments").
membrain-seg is not imported: its ``membrain`` program is run (``conda run -n <env>`` when an environment is
given), so this file needs nothing but the standard library (Python 3.6+: RELION may start it on a compute node
with the system python3).

segment: for each tomogram of a RELION 5 tomograms.star, ``membrain segment`` with the job's checkpoint, the
tomogram's voxel size (rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning, else its MRC header) as
--in-pixel-size, and optionally ``membrain skeletonize`` on its segmentation. The tomogram is linked as
inputs/<rlnTomoName>.mrc first (membrain-seg names its outputs after the tomogram's file), and what membrain-seg
writes is renamed to names that do not depend on the checkpoint, in segmentations/:

    <name>_segmented.mrc     the segmentation (with connected components: their labels), the tomogram's size
    <name>_scores.mrc        the membrane scores before the threshold (Store probabilities?)
    <name>_uncertainty.mrc   the variance of the test-time augmentations (Store uncertainty map?)
    <name>_skeleton.mrc      the skeleton of the segmentation (Skeletonize?)

One tomogram per GPU at a time (or a few: "Tomograms per GPU"), each program shown only its GPU
(CUDA_VISIBLE_DEVICES). membrain_seg.star (one row per tomogram: the tomograms.star columns, the files above and
the status of each tomogram) is rewritten as tomograms finish, so the results can be looked at while the job
runs; tomograms.star (the tomograms done, with the same columns but the status) is the job's output for later
jobs. Tomograms already done are skipped when the job runs again (continue).

Thresholds: each tomogram can have a segmentation threshold of its own (option tomo_thresholds, which the job's
Scores view sets; else the job's). membrain_seg.star keeps the threshold each segmentation was made with; on a
continue, a tomogram whose threshold changed and that has its score map is thresholded again from it
(``membrain thresholds``: no network, no GPU work), then its connected components (``membrain components``)
and skeleton are made again; without its score map it is segmented again. delete_scores() deletes the score maps
once they are not needed (the view's "Delete score maps…").

Like every External job it writes RELION_JOB_EXIT_SUCCESS / _FAILURE / _ABORTED and, before success,
RELION_OUTPUT_NODES.star. RELION_JOB_ABORT_NOW and SIGTERM/SIGINT/SIGHUP (scancel) stop membrain-seg (its
process group: SIGTERM, then SIGKILL).
"""

import os
import sys
import time

if __package__:  # imported from cryo-savanna
    from cryo_savanna import runner_common as C
else:  # run in a project: the copy of runner_common.py beside this one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import runner_common as C  # type: ignore[no-redef]

Usage, Aborted, say = C.Usage, C.Aborted, C.say
TOMO_NAME = C.TOMO_NAME
yes, text, number, words, num = C.yes, C.text, C.number, C.words, C.num
_f, _rel, _link, _touch = C._f, C._rel, C._link, C._touch

WHO = "membrain-seg"  # say(): " ++++ membrain-seg: ..."
SEGMENT = "segment"
MODES = (SEGMENT,)

# files of the job
SUMMARY = "membrain_seg.star"  # one row per tomogram, with its status
TOMOGRAMS = "tomograms.star"  # the tomograms done, with their segmentations: the job's output
SEG_DIR, INPUTS_DIR, LOG_DIR = "segmentations", "inputs", "logs"
SEGMENTED, SCORES, UNCERTAINTY, SKELETON = "_segmented.mrc", "_scores.mrc", "_uncertainty.mrc", "_skeleton.mrc"

# node types of the outputs (RELION_OUTPUT_NODES.star)
NODE_SUMMARY = "ProcessData.star.membrain.summary"
NODE_TOMOGRAMS = "TomogramGroupMetadata.star.relion.tomo.membrain"

# columns cryo-savanna adds (not RELION's: RELION keeps labels it does not know)
P = "savannaMembrain"
COL_SEGMENTATION, COL_SCORES = P + "Segmentation", P + "Scores"
COL_UNCERTAINTY, COL_SKELETON = P + "Uncertainty", P + "Skeleton"
COL_TOMOGRAM, COL_VOXEL = P + "Tomogram", P + "VoxelSize"
COL_THRESHOLD = P + "Threshold"  # the segmentation threshold a tomogram's segmentation was made with
COL_STATUS, COL_SECONDS, COL_LOG = P + "Status", P + "Seconds", P + "Log"

SKEL_METHODS = ("3D-NMS", "2D-skimage")
TRAINING_PIXEL = 10.0  # Å: what membrain-seg's models were trained at (--out-pixel-size)


# ---------------------------------------------------------------------------- arguments


def parse(argv):
    return C.parse(argv, MODES)


def conda_prefix(opts):
    """The start of every membrain-seg command: ``conda run`` in the job's environment (nothing: on PATH)."""
    return C.conda_run(opts, "membrain_env")


def checkpoint(opts):
    ckpt = text(opts, "ckpt_path")
    if not ckpt:
        raise Usage("give the model checkpoint (.ckpt) of membrain-seg (Segmentation tab)")
    if not os.path.isfile(ckpt):
        raise Usage("model checkpoint not found: %s" % ckpt)
    return ckpt


def ckpt_token(ckpt):
    """What membrain-seg puts in its output names for this checkpoint (its file name up to "-val_loss")."""
    return os.path.basename(ckpt).split("-val_loss")[0]


def window_size(opts):
    n = number(opts, "sliding_window_size", int, what="Sliding window size")
    if n is None:
        return None
    if n <= 0 or n % 32:
        raise Usage("Sliding window size: a multiple of 32, e.g. 160 or 128 (got %d)" % n)
    return n


def voxel_size(row, opts, tomogram):
    """The tomogram's voxel size (Å): the job's, else rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning,
    else its MRC header."""
    v = number(opts, "voxel_size", what="Tomogram voxel size")
    if v:
        return v
    px, binning = _f(row, "rlnTomoTiltSeriesPixelSize"), _f(row, "rlnTomoTomogramBinning")
    header = C.mrc_header(tomogram)[3]
    if px and binning:
        v = px * binning
        if header > 0 and abs(header - v) > 0.02 * v:
            say("WARNING: %s: rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning is %.3f Å, the tomogram's header "
                "says %.3f Å; %.3f Å is used (give the voxel size in the job if it is wrong)"
                % (row[TOMO_NAME], v, header, v))
        return v
    if header > 0:
        return header
    raise Usage("%s: no voxel size (rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning, MRC header): give it in "
                "the job" % row[TOMO_NAME])


def job_threshold(opts):
    """The job's segmentation threshold (membrain-seg's scores: its network's raw output, 0 by default)."""
    t = number(opts, "segmentation_threshold", what="Segmentation threshold")
    return 0.0 if t is None else t


def tomo_thresholds(opts):
    """{tomogram: threshold} of option tomo_thresholds ("name=value name=value ...": another threshold for some
    tomograms, set in the job's Scores view)."""
    out = {}
    for w in words(opts, "tomo_thresholds"):
        name, sep, v = w.rpartition("=")
        try:
            val = float(v)
        except ValueError:
            val = None
        if not sep or not name or val is None or val != val:
            raise Usage("Threshold of some tomograms: %r is not <tomogram>=<number>" % w)
        out[name] = val
    return out


def threshold_of(opts, name):
    return tomo_thresholds(opts).get(name, job_threshold(opts))


def threshold_name(threshold):
    """How membrain thresholds writes a threshold in its output's name (Python's float repr: 0.0, -1.5)."""
    return "%r" % float(threshold)


def segment_command(opts, tomogram, ckpt, voxel, out_dir, threshold=None):
    cmd = conda_prefix(opts) + ["membrain", "segment", "--tomogram-path", tomogram, "--ckpt-path", ckpt,
                                "--out-folder", out_dir, "--in-pixel-size", num(voxel)]
    out_px = number(opts, "out_pixel_size", what="Pixel size of the model") or TRAINING_PIXEL
    cmd += ["--out-pixel-size", num(out_px)]
    if yes(opts, "rescale_patches"):
        cmd.append("--rescale-patches")
    tta = opts.get("test_time_augmentation", "Yes") == "Yes"
    if not tta:
        cmd.append("--no-test-time-augmentation")
    threshold = job_threshold(opts) if threshold is None else threshold
    cmd += ["--segmentation-threshold", num(threshold)]
    window = window_size(opts)
    if window:
        cmd += ["--sliding-window-size", str(window)]
    if yes(opts, "store_probabilities"):
        cmd.append("--store-probabilities")
    if yes(opts, "store_uncertainty"):
        if not tta:
            raise Usage("the uncertainty map needs the test-time augmentation (Segmentation tab)")
        cmd.append("--store-uncertainty-map")
    if yes(opts, "store_connected_components"):
        cmd.append("--store-connected-components")
        thres = number(opts, "connected_component_thres", int, what="Smallest connected component")
        if thres:
            cmd += ["--connected-component-thres", str(thres)]
    return cmd + words(opts, "segment_args")


def thresholds_command(opts, scores, threshold, out_dir):
    """membrain thresholds: the segmentation again from a score map, at another threshold (no network: CPU)."""
    return conda_prefix(opts) + ["membrain", "thresholds", "--scoremap-path", scores, "--thresholds", num(threshold),
                                 "--out-folder", out_dir]


def components_command(opts, segmentation, out_dir):
    cmd = conda_prefix(opts) + ["membrain", "components", "--segmentation-path", segmentation, "--out-folder",
                                out_dir]
    thres = number(opts, "connected_component_thres", int, what="Smallest connected component")
    if thres:
        cmd += ["--connected-component-thres", str(thres)]
    return cmd


def skeleton_command(opts, segmentation, out_dir):
    method = text(opts, "skel_method") or SKEL_METHODS[0]
    if method not in SKEL_METHODS:
        raise Usage("Skeletonization method: %s (got %r)" % (" or ".join(SKEL_METHODS), method))
    cmd = conda_prefix(opts) + ["membrain", "skeletonize", "--label-path", segmentation, "--out-folder", out_dir,
                                "--method", method]
    batch = number(opts, "skel_batch_size", int, what="Skeletonization batch size")
    if batch:
        cmd += ["--batch-size", str(batch)]
    return cmd + words(opts, "skel_args")


# ---------------------------------------------------------------------------- the tomograms


def outputs(job_dir, name):
    """{what: path} of a tomogram's files in segmentations/, as the job names them."""
    d = os.path.join(job_dir, SEG_DIR)
    return {"segmentation": os.path.join(d, name + SEGMENTED), "scores": os.path.join(d, name + SCORES),
            "uncertainty": os.path.join(d, name + UNCERTAINTY), "skeleton": os.path.join(d, name + SKELETON)}


def wanted(opts):
    """The files a tomogram should have, besides its segmentation, for these options."""
    out = []
    if yes(opts, "store_probabilities"):
        out.append("scores")
    if yes(opts, "store_uncertainty"):
        out.append("uncertainty")
    if yes(opts, "do_skeleton"):
        out.append("skeleton")
    return out


def prepare_tomogram(row, opts, job_dir):
    """The tomogram to segment (linked as inputs/<name>.mrc), its voxel size and the files membrain-seg writes."""
    name = row[TOMO_NAME]
    tomogram = C.tomogram_of(row, opts)
    link = os.path.join(job_dir, INPUTS_DIR, name + ".mrc")
    tomo_in = link if _link(tomogram, link) else tomogram
    return {"tomogram": tomo_in, "source": tomogram, "stem": os.path.splitext(os.path.basename(tomo_in))[0],
            "voxel": voxel_size(row, opts, tomogram)}


def _move(src, dst):
    if os.path.exists(src):
        os.replace(src, dst)
        return True
    return False


def columns_of(opts):
    """The columns the job adds to the tomograms.star ones (membrain_seg.star adds those of its status too)."""
    out = [COL_SEGMENTATION, COL_THRESHOLD]
    out += [c for what, c in (("scores", COL_SCORES), ("uncertainty", COL_UNCERTAINTY), ("skeleton", COL_SKELETON))
            if what in wanted(opts)]
    return out + [COL_TOMOGRAM, COL_VOXEL]


def write_tables(job_dir, opts, columns, rows, state):
    """membrain_seg.star (every tomogram, with its status) and tomograms.star (those done)."""
    files = {COL_SEGMENTATION: "segmentation", COL_SCORES: "scores", COL_UNCERTAINTY: "uncertainty",
             COL_SKELETON: "skeleton"}
    own = columns_of(opts)
    status = [COL_STATUS, COL_SECONDS, COL_LOG]
    base = [c for c in columns if c not in own and c not in status]
    summary, done = [], []
    for r in rows:
        s = state[r[TOMO_NAME]]
        vals = dict(r)
        done_now = s["status"] == "done"
        for c in own:
            if c in files:
                vals[c] = _rel(s["files"][files[c]]) if done_now else ""
        used = s.get("threshold")
        vals[COL_THRESHOLD] = "" if used is None else num(used)
        vals.update({COL_TOMOGRAM: _rel(s["tomogram"]), COL_VOXEL: "%.4f" % s["voxel"], COL_STATUS: s["status"],
                     COL_SECONDS: "%.0f" % s.get("seconds", 0), COL_LOG: _rel(s["log"])})
        summary.append([vals.get(c, "") for c in base + own + status])
        if s["status"] == "done":
            done.append([vals.get(c, "") for c in base + own])
    general = {P + "TomogramsFile": text(opts, "in_tomograms"), P + "Checkpoint": text(opts, "ckpt_path"),
               P + "TomogramKind": text(opts, "tomo_kind") or "Reconstructed"}
    C.write_star(os.path.join(job_dir, SUMMARY), [("general", general, None, None),
                                                  ("global", None, base + own + status, summary)])
    C.write_star(os.path.join(job_dir, TOMOGRAMS), [("global", None, base + own, done)])


def previous_thresholds(job_dir, job_default):
    """{tomogram: the threshold its segmentation was made with}, from the job's last membrain_seg.star (a job of
    an earlier cryo-savanna, without the column: ``job_default``, the job's threshold)."""
    path = os.path.join(job_dir, SUMMARY)
    if not os.path.isfile(path):
        return {}
    try:
        b = C.loop_block(path, "global")
    except (Usage, OSError):
        return {}
    out = {}
    for r in b.dicts():
        v = _f(r, COL_THRESHOLD) if COL_THRESHOLD in b.columns else job_default
        if v is not None:
            out[r.get(TOMO_NAME, "")] = v
    return out


def score_maps(job_dir):
    """The score maps the job kept (segmentations/<name>_scores.mrc), by tomogram."""
    d = os.path.join(job_dir, SEG_DIR)
    if not os.path.isdir(d):
        return {}
    return dict((f[:-len(SCORES)], os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(SCORES))


def delete_scores(job_dir):
    """Deletes the job's score maps and takes their column out of membrain_seg.star and tomograms.star (the
    job's options say whether it writes them again: membrain_jobs.delete_score_maps). The bytes freed."""
    freed = 0
    for path in score_maps(job_dir).values():
        try:
            freed += os.path.getsize(path)
            os.remove(path)
        except OSError:
            pass
    for name in (SUMMARY, TOMOGRAMS):
        path = os.path.join(job_dir, name)
        if not os.path.isfile(path):
            continue
        blocks = C.read_star(path)
        out = []
        for b in blocks.values():
            if b.is_loop and COL_SCORES in b.columns:
                i = b.columns.index(COL_SCORES)
                out.append((b.name, b.pairs or None, b.columns[:i] + b.columns[i + 1:],
                            [r[:i] + r[i + 1:] for r in b.rows]))
            else:
                out.append((b.name, b.pairs or None, b.columns if b.is_loop else None, b.rows))
        C.write_star(path, out)
    return freed


# ---------------------------------------------------------------------------- segment


def run_segment(sup, job_dir, threads, opts):
    _, columns, rows = C.tomogram_rows(opts)
    for d in (SEG_DIR, INPUTS_DIR, LOG_DIR):
        p = os.path.join(job_dir, d)
        if not os.path.isdir(p):
            os.makedirs(p)
    ckpt = checkpoint(opts)
    token = ckpt_token(ckpt)
    extra = wanted(opts)
    seg_dir = os.path.join(job_dir, SEG_DIR)
    state, prepared = {}, {}
    used = previous_thresholds(job_dir, job_threshold(opts))
    skip = yes(opts, "skip_done")
    for r in rows:  # every tomogram's inputs, and the options, first: a wrong setting shows before hours on the GPU
        name = r[TOMO_NAME]
        f = prepared[name] = prepare_tomogram(r, opts, job_dir)
        files = outputs(job_dir, name)
        want = threshold_of(opts, name)
        segment_command(opts, f["tomogram"], ckpt, f["voxel"], seg_dir, want)
        if yes(opts, "do_skeleton"):
            skeleton_command(opts, files["segmentation"], seg_dir)
        if yes(opts, "store_connected_components"):
            components_command(opts, files["segmentation"], seg_dir)
        have = skip and os.path.isfile(files["segmentation"]) and \
            all(os.path.isfile(files[w]) for w in extra if w != "skeleton")
        same = name in used and abs(used[name] - want) < 1e-9
        # segmented before at another threshold, with its score map: thresholded again, not segmented again
        again = have and not same and os.path.isfile(files["scores"])
        segmented = have and same
        skeleton = yes(opts, "do_skeleton") and (not segmented or not os.path.isfile(files["skeleton"]))
        done = segmented and not skeleton
        state[name] = {"status": "done" if done else "pending", "tomogram": f["tomogram"], "voxel": f["voxel"],
                       "files": files, "log": os.path.join(job_dir, LOG_DIR, name + ".log"),
                       "segmented": segmented, "again": again, "threshold": used.get(name), "want": want}
        if done:
            say("%s: already done, skipped" % name)
        elif again:
            say("%s: thresholded again at %s (it was %s), from its score map" % (name, num(want), num(used[name])))
    voxels = sorted(set(round(f["voxel"], 2) for f in prepared.values()))
    out_px = number(opts, "out_pixel_size") or TRAINING_PIXEL
    far = [v for v in voxels if abs(v - out_px) > 0.4 * out_px]
    if far and not yes(opts, "rescale_patches"):
        say("WARNING: tomograms of %s Å, the model's pixel size is %s Å: switch on 'Rescale patches?' (or use "
            "tomograms binned closer to it)" % (", ".join(num(v) for v in far), num(out_px)))

    def tables():
        write_tables(job_dir, opts, columns, rows, state)

    def task(name):
        s, f = state[name], prepared[name]

        def factory(gpu):
            def gen():
                env = {"CUDA_VISIBLE_DEVICES": C.cuda_device(gpu)}
                s["status"], t0 = "running", time.time()
                tables()
                files = s["files"]
                if s["again"]:
                    rc = yield thresholds_command(opts, files["scores"], s["want"], seg_dir), s["log"], env
                    written = os.path.join(seg_dir, "%s_scores_threshold_%s.mrc" % (name, threshold_name(s["want"])))
                    if rc != 0 or not os.path.isfile(written):
                        C._show_tail(s["log"])
                        raise Usage("membrain thresholds exited with %s%s (log: %s)"
                                    % (rc, "" if rc != 0 else ", without writing %s" % _rel(written), s["log"]))
                    os.replace(written, files["segmentation"])
                    if yes(opts, "store_connected_components"):
                        rc = yield components_command(opts, files["segmentation"], seg_dir), s["log"], env
                        written = os.path.join(seg_dir, name + "_segmented_components.mrc")
                        if rc != 0 or not os.path.isfile(written):
                            C._show_tail(s["log"])
                            raise Usage("membrain components exited with %s%s (log: %s)"
                                        % (rc, "" if rc != 0 else ", without writing %s" % _rel(written), s["log"]))
                        os.replace(written, files["segmentation"])
                    s["threshold"] = s["want"]
                elif not s["segmented"]:
                    say("%s: segmenting on GPU %d" % (name, gpu))
                    s["threshold"] = None  # (its old segmentation is replaced)
                    rc = yield segment_command(opts, f["tomogram"], ckpt, f["voxel"], seg_dir, s["want"]), \
                        s["log"], env
                    written = os.path.join(seg_dir, "%s_%s_segmented.mrc" % (f["stem"], token))
                    if rc != 0 or not os.path.isfile(written):
                        C._show_tail(s["log"])
                        raise Usage("membrain segment exited with %s%s (log: %s)"
                                    % (rc, "" if rc != 0 else ", without writing %s" % _rel(written), s["log"]))
                    os.replace(written, files["segmentation"])
                    if f["stem"] != name:  # no link: membrain-seg named them after the tomogram's own file
                        _move(os.path.join(seg_dir, f["stem"] + SCORES), files["scores"])
                    _move(os.path.join(seg_dir, "%s_%s_uncertainty_map.mrc" % (f["stem"], token)),
                          files["uncertainty"])
                    for w in ("scores", "uncertainty"):
                        if w in extra and not os.path.isfile(files[w]):
                            raise Usage("membrain segment wrote no %s map (%s)" % (w, _rel(files[w])))
                    s["threshold"] = s["want"]
                if yes(opts, "do_skeleton"):  # (again after a new segmentation)
                    say("%s: skeletonizing on GPU %d" % (name, gpu))
                    rc = yield skeleton_command(opts, files["segmentation"], seg_dir), s["log"], env
                    written = os.path.join(seg_dir, name + "_segmented_skel.mrc")
                    if rc != 0 or not os.path.isfile(written):
                        C._show_tail(s["log"])
                        raise Usage("membrain skeletonize exited with %s%s (log: %s)"
                                    % (rc, "" if rc != 0 else ", without writing %s" % _rel(written), s["log"]))
                    os.replace(written, files["skeleton"])
                s["seconds"] = time.time() - t0
                return True
            return gen()
        return name, factory

    ids = C.gpu_ids(opts)
    per_gpu = number(opts, "per_gpu", int, what="Tomograms per GPU") or 1
    if per_gpu < 1:
        raise Usage("Tomograms per GPU: 1 or more")
    pool = C.Pool("segment", [g for _ in range(per_gpu) for g in ids])
    for r in rows:
        if state[r[TOMO_NAME]]["status"] != "done":
            pool.add(*task(r[TOMO_NAME]))
    tables()
    say("%d tomogram(s) to segment, GPU(s) %s, %d at a time" % (len(pool.pending), " ".join(map(str, ids)),
                                                                len(pool.slots or ())))

    def changed(_pool, name, result):
        s = state[name]
        stopping = sup.stopped is not None or sup.aborted  # killed by the abort: not a failure of its own
        s["status"] = "done" if result else ("aborted" if stopping else "failed")
        say("%s: %s in %.1f min" % (name, s["status"], s.get("seconds", 0) / 60))
        tables()

    try:
        C.run_pools(sup, [pool], changed)
    except Aborted:
        for s in state.values():
            if s["status"] == "running":
                s["status"] = "aborted"
        tables()
        raise
    failed = [r[TOMO_NAME] for r in rows if state[r[TOMO_NAME]]["status"] != "done"]
    if failed:
        raise Usage("%d tomogram(s) failed: %s. Continue the job to run only those again."
                    % (len(failed), " ".join(failed)))
    return [(os.path.join(job_dir, TOMOGRAMS), NODE_TOMOGRAMS), (os.path.join(job_dir, SUMMARY), NODE_SUMMARY)]


# ---------------------------------------------------------------------------- main


def run(argv):
    mode, job_dir, threads, opts = parse(argv)
    C.WHO = WHO
    for f in (C.SUCCESS, C.FAILURE, C.ABORTED):
        if os.path.exists(os.path.join(job_dir, f)):
            os.remove(os.path.join(job_dir, f))
    sup = C.Supervisor(job_dir, hint=" (is membrain-seg's conda environment right?)")
    say("cryo-savanna membrain-seg %s, job %s" % (mode, job_dir))
    try:
        C.load_modules(opts)
        nodes = run_segment(sup, job_dir, threads, opts)
    except Aborted:
        _touch(os.path.join(job_dir, C.ABORTED))
        return 128 + sup.stopped if sup.stopped else 1
    except Usage as e:
        print("ERROR: " + str(e), file=sys.stderr, flush=True)
        _touch(os.path.join(job_dir, C.FAILURE))
        return 1
    C.write_output_nodes(job_dir, [(_rel(p), t) for p, t in nodes])
    say("done")
    _touch(os.path.join(job_dir, C.SUCCESS))
    return 0


def main():
    C.main(run)


if __name__ == "__main__":
    main()
