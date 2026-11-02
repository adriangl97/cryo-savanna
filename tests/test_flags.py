# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Flags of the programs behind "Additional arguments:" (no Qt)."""

import stat
from pathlib import Path

import pytest

from cryo_savanna.env.relion_env import RelionEnv
from cryo_savanna.jobdefs.flags import (
    JOB_PROGRAMS,
    FlagSource,
    command_flags,
    parse_args,
    parse_help,
    program_for,
    set_arg,
)
from cryo_savanna.jobdefs.model import JobDefs

HELP = Path(__file__).parent / "data" / "help-5.1.1"
BUNDLED = FlagSource(which=lambda p: None)


def test_parses_every_help_dump():
    for f in HELP.glob("*.txt"):
        text = f.read_text()
        flags = parse_help(text)
        listed = [ln for ln in text.splitlines() if ln.lstrip().startswith("--") and " : " in ln]
        hidden = sum(ln.split()[0] in ("--version", "--theme", "--list_themes", "--pipeline_control") for ln in listed)
        assert len(flags) == len(listed) - hidden, f.name
        assert len({fl.name for fl in flags}) == len(flags), f.name


def test_flag_details():
    refine = {f.name: f for f in parse_help((HELP / "relion_refine.txt").read_text())}
    assert refine["--iter"].default == "-1" and refine["--iter"].section == "General options"
    assert refine["--dont_skip_gridding"].is_switch and refine["--ctf"].section == "Corrections"
    assert refine["--i"].default == "" and not refine["--i"].is_switch
    subtomo = {f.name: f for f in parse_help((HELP / "relion_tomo_subtomo.txt").read_text())}
    assert subtomo["--o"].default is None  # compulsory: no default
    ctf = {f.name: f for f in parse_help((HELP / "relion_ctf_refine.txt").read_text())}
    assert ctf["--fit_mode"].help.endswith("during the per-micrograph CTF refinement.")  # continued line
    assert parse_help((HELP / "relion_run_motioncorr.txt").read_text()) == []  # it fails without --use_own


def test_bundled_help_for_every_job_program():
    for program in JOB_PROGRAMS:
        pf = BUNDLED.flags(program)
        assert pf is not None and len(pf.flags) > 5, program
    mc = BUNDLED.flags("relion_run_motioncorr")  # rebuilt from the source
    assert mc.get("--use_own").is_switch and mc.get("--eer_grouping").default == "40"
    assert BUNDLED.flags("relion_python_tomo_pick") is None


def test_every_variant_has_a_program():
    for v in JobDefs.load().variants.values():
        assert program_for(v.label, v.default_values(RelionEnv({}))), v.key


def test_program_follows_the_options():
    assert program_for("relion.class2d", {}) == "relion_refine"
    assert program_for("relion.autopick", {"do_amyloid": "No"}) == "relion_autopick"
    assert program_for("relion.autopick", {"do_amyloid": "Yes"}) == "relion_find_amyloid"
    assert program_for("relion.select", {"do_class_ranker": "Yes"}) == "relion_class_ranker"
    assert program_for("relion.select", {"do_split": "Yes", "do_class_ranker": "Yes"}) == "relion_star_handler"
    assert program_for("relion.select", {}) == "relion_display"
    assert program_for("relion.localres", {"do_relion_locres": "No"}) == "ResMap"
    assert program_for("relion.importtomo", {"do_coords": "Yes"}) == "relion_tomo_import_coordinates"


def test_editing_the_arguments_text():
    assert parse_args('--gpu "0:1" --skip_gridding --pad 1') == {"--gpu": ["0:1"], "--skip_gridding": [], "--pad": ["1"]}
    text = '--skip_gridding --iter 5 "keep me"'
    assert set_arg(text, "--iter", "10") == '--skip_gridding --iter 10 "keep me"'
    assert set_arg(text, "--iter", None) == '--skip_gridding "keep me"'
    assert set_arg(text, "--skip_gridding", None, switch=True) == '--iter 5 "keep me"'
    assert set_arg(text, "--pad", "2") == text + " --pad 2"
    assert set_arg("", "--maxsig", "a b") == "--maxsig 'a b'"
    assert set_arg("--iter 5 --iter 6", "--iter", "7") == "--iter 7"  # one left
    assert set_arg("free text", "--x", "", switch=True) == "free text --x"
    # RELION runs the arguments through the shell: a value it would act on is quoted
    assert set_arg("", "--i", "Movies/*.tif") == "--i 'Movies/*.tif'"
    assert set_arg("", "--o", "a;b") == "--o 'a;b'" and set_arg("", "--o", "$HOME") == "--o '$HOME'"
    assert set_arg("", "--i", '"Movies/*.tif"') == '--i "Movies/*.tif"'  # quoted by hand: as typed
    assert parse_args(set_arg("", "--sym", "it's"))["--sym"] == ["it's"]  # an open quote is quoted too


def test_command_flags_from_note():
    note = (" ++++ Executing new job on Tue Oct 28 10:00:00 2026\n ++++ with the following command(s):\n"
            "`which relion_refine_mpi` --o Class2D/job001/run --iter 25 --K 50 --ctf --skip_gridding --pad 2 "
            "--pipeline_control Class2D/job001/\n ++++ \n"
            " ++++ Executing new job on Wed Oct 29 10:00:00 2026\n ++++ with the following command(s):\n"
            "`which relion_refine_mpi` --o Class2D/job001/run --iter 30 --K 50 --ctf --iter 40\n"
            "`which relion_align_symmetry` --sym C2\n ++++ \n")
    flags = command_flags(note, other_args="--iter 40", program="relion_refine")
    assert flags == {"--o": "Class2D/job001/run", "--iter": "30", "--K": "50", "--ctf": ""}  # the last run only
    assert "--sym" in command_flags(note)


def test_live_help_is_used_and_cached(tmp_path):
    exe = tmp_path / "relion_run_motioncorr"
    calls = tmp_path / "calls"
    exe.write_text(f"""#!/bin/sh
echo "$@" >> {calls}
case "$*" in *--use_own*) ;; *) echo "ERROR: choose" >&2; exit 1;; esac
echo "+++ RELION: command line arguments (with defaults for optional ones between parantheses) +++"
echo "====== General ===== "
echo "                     --only_live (3) : a flag only this build has"
""")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    src = FlagSource(which=lambda p: str(exe) if p == exe.name else None)
    assert src.needs_run("relion_run_motioncorr")
    pf = src.flags("relion_run_motioncorr")
    assert [f.name for f in pf.flags] == ["--only_live"] and str(exe) in pf.source
    assert not src.needs_run("relion_run_motioncorr")
    src.flags("relion_run_motioncorr")
    assert calls.read_text().splitlines() == ["--help", "--use_own --help"]  # retried once, then cached


def test_broken_live_help_falls_back_to_bundled(tmp_path):
    exe = tmp_path / "relion_refine"
    exe.write_text("#!/bin/sh\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    pf = FlagSource(which=lambda p: str(exe)).flags("relion_refine")
    assert pf.get("--iter") is not None and "bundled" in pf.source


@pytest.mark.parametrize("program", ["relion_manualpick", "relion_display"])
def test_gui_programs_are_not_started(program):
    src = FlagSource(which=lambda p: pytest.fail("must not look for " + p))
    assert not src.needs_run(program) and src.flags(program) is not None
