# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Failure hints: known messages in a failed job's logs (project/failure_hints.py), in the summaries and the
Overview."""

import pytest

from cryo_savanna.project import failure_hints as F
from cryo_savanna.project.snapshot import ProjectLoader
from cryo_savanna.project.summary import SummaryReader

# one log per hint, with the text the program really writes (RELION 5.1.1's src/error.h and REPORT_ERRORs,
# CUDA, Open MPI, SLURM, cryo-savanna's runners and SLURM monitor)
LOGS = {
    "gpu_oom": ("run.err", "in: /src/acc/cuda/cuda_settings.h, line 67\nERROR: \n\nYou ran out of memory on the GPU(s).\n"),
    "gpu_ids": ("run.err", "ERROR: \n\nThere was an issue with the GPU-ids. Either \n"),
    "gpu_mpi": ("run.err", "ERROR: \nERROR: not enough MPI nodes specified for the GPU IDs.\n"),
    "gpu_kernel": ("run.err", "ERROR: \n\nA GPU-function failed to execute.\n"),
    "time_limit": ("run.err", "slurmstepd: error: *** JOB 4242 ON gpu07 CANCELLED AT 2026-11-16T10:00:00 DUE TO TIME LIMIT ***\n"),
    "ram_oom": ("run.err", "slurmstepd: error: Detected 1 oom_kill event in StepId=4242.batch.\n"),
    "node_fail": ("run.err", "\n ++++ cryo-savanna: SLURM job 4242 ended NODE_FAIL, without RELION's exit file: "
                             "RELION_JOB_EXIT_FAILURE written, Mon Nov 16 10:00:00 2026\n"),
    "disk_full": ("run.err", "write error: No space left on device\n"),
    "permission": ("run.err", "Cannot open output file: Class3D/job012/run_it001_data.star: Permission denied\n"),
    "mpi_minimum": ("run.err", "ERROR: \nMlOptimiserMpi::initialiseWorkLoad: at least 3 MPI processes are required "
                               "when splitting data into random halves\n"),
    "mpi_slots": ("run.out", "There are not enough slots available in the system to satisfy the 5\nslots that were "
                             "requested by the application:\n"),
    "local_mpi": ("run.err", "You're submitting a local job with 40 parallel MPI processes. That's more than allowed "
                             "by the RELION_ERROR_LOCAL_MPI environment variable.\n"),
    "mask_size": ("run.err", "ERROR: \nERROR: Solvent mask does not have the same size as the reference vol.\n"),
    "pixel_size": ("run.err", "ERROR: \nCalculated and user-defined pixel sizes are different (> 1% of error)!\n"),
    "zero_weights": ("run.err", "\n fn_img= 000017@Extract/job007/Movies/mic_003.mrcs\n adaptive_fraction= 0.999\n"
                                " min_diff2= 1.7e+38\nin: /src/acc/acc_ml_optimiser_impl.h, line 2255\nERROR: \n"
                                "No orientation was found as better than any other.\n\nA particle image was compared "
                                "to the reference and resulted in all-zero\n"),
    "no_particles": ("run.err", "ERROR: \nERROR: there are no particles in Select/job020/particles.star\n"),
    "python_module": ("run.err", "Traceback (most recent call last):\nModuleNotFoundError: No module named 'napari'\n"),
    "cannot_start": ("run.err", "cannot start pytom_match_template.py: [Errno 2] No such file or directory: "
                                "'pytom_match_template.py' (is pytom's conda environment right?)\n"),
    "missing_file": ("run.err", "ERROR: \nMetaDataTable::read: File Extract/job007/particles.star does not exist\n"),
    "segfault": ("run.out", "[gpu07:12345] *** Process received signal ***\n[gpu07:12345] Signal: Segmentation fault (11)\n"),
    "mpi_abort": ("run.out", "MPI_ABORT was invoked on rank 2 in communicator MPI_COMM_WORLD\nwith errorcode 1.\n"),
}


def job(tmp_path, logs: dict[str, str]):
    d = tmp_path / "Class3D" / "job012"
    d.mkdir(parents=True)
    for name, text in logs.items():
        (d / name).write_text(text)
    return d


def test_every_hint_has_a_sample_log():
    assert set(LOGS) == {h.id for h in F.HINTS}


@pytest.mark.parametrize("hint_id", sorted(LOGS))
def test_each_log_gives_its_hint(tmp_path, hint_id):
    name, text = LOGS[hint_id]
    found = F.diagnose(job(tmp_path, {name: text}))
    assert [d.hint.id for d in found] == [hint_id]
    assert found[0].file == name and found[0].line in text and found[0].text == F.BY_ID[hint_id].text


def test_a_clean_log_gives_nothing(tmp_path):
    d = job(tmp_path, {"run.out": " + Reading particles ...\n Expectation iteration 3 of 25\n", "run.err": ""})
    assert F.diagnose(d) == []
    assert F.diagnose(tmp_path / "nowhere") == []  # no logs at all


def test_relion_error_when_nothing_is_known(tmp_path):
    """RELION's own message, the line after its "ERROR:" (an "ERROR:" in front of it again is dropped)."""
    d = job(tmp_path, {"run.err": "in: /src/postprocessing.cpp, line 1188\nERROR: \nERROR: mask and map have "
                                  "different sizes\n=== Backtrace  ===\n",
                       "run.out": "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"})
    [diag] = F.diagnose(d)
    assert diag.hint is F.RELION_ERROR and diag.line == "mask and map have different sizes"
    assert diag.text == "RELION stopped with this error: mask and map have different sizes"
    assert (diag.file, diag.line_no) == ("run.err", 3)


def test_specific_hints_hide_general_ones(tmp_path):
    d = job(tmp_path, {"run.err": LOGS["gpu_oom"][1] + "Segmentation fault (core dumped)\n",
                       "run.out": LOGS["mpi_abort"][1]})
    assert [x.hint.id for x in F.diagnose(d)] == ["gpu_oom"]  # not the crash it ends in, not MPI's abort
    d = job(tmp_path / "b", {"run.err": LOGS["python_module"][1] + "No such file or directory\n"})
    assert [x.hint.id for x in F.diagnose(d)] == ["python_module"]


def test_at_most_three_most_specific_first(tmp_path):
    text = "".join(LOGS[k][1] for k in ("permission", "disk_full", "time_limit", "gpu_oom", "pixel_size"))
    found = F.diagnose(job(tmp_path, {"run.err": text}))
    assert [x.hint.id for x in found] == ["gpu_oom", "time_limit", "disk_full"]


def test_line_numbers_count_the_whole_file(tmp_path):
    """Only the end of a long run.out is read; the line number is still that of the whole file."""
    head = "".join(f" Expectation iteration {i}\n" for i in range(20000))  # well over TAIL_BYTES
    d = job(tmp_path, {"run.out": head + LOGS["mpi_slots"][1]})
    [diag] = F.diagnose(d)
    assert diag.line_no == 20001 and diag.line.startswith("There are not enough slots")


def test_only_failed_jobs_are_diagnosed(spa_project):
    snap = ProjectLoader(spa_project).refresh()
    reader = SummaryReader(spa_project)
    s = {n: reader.summarize(i, snap.pipeline) for n, i in snap.jobs.items()}
    failed = [n for n, i in snap.jobs.items() if i.status == "Failed"]
    assert failed and all(s[n].failure for n in failed)
    assert s["LocalRes/job017/"].failure[0].line == "mask and map have different sizes"
    assert not any(s[n].failure for n in snap.jobs if n not in failed)


def test_overview_and_log_link(qtbot, tmp_path, spa_project):
    pytest.importorskip("pytestqt")
    from qtpy.QtCore import QSettings, QUrl
    from cryo_savanna.ui.main_window import MainWindow
    w = MainWindow(spa_project, poll_s=1000.0, settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    qtbot.addWidget(w)
    w.load_now()
    w.load_summaries_now()
    w.select_job("LocalRes/job017/")
    html = w.info.overview.toHtml()
    assert "Likely cause" in html and "mask and map have different sizes" in html and "run.err, line 3" in html
    card = w.graph.nodes["LocalRes/job017/"]
    assert "Likely cause: RELION stopped with this error" in card.toolTip()
    w.logs.setCurrentIndex(0)
    w.info._on_link(QUrl("log:0"))
    assert w.logs.currentWidget() is w.logs.views["run.err"]
    assert w.logs.views["run.err"].textCursor().selectedText() == "mask and map have different sizes"
