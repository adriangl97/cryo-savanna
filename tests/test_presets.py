# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
"""Setting presets: the store (project/presets.py), and the job editor's Setting presets button and dialogs."""

import json
import multiprocessing

import pytest

from cryo_savanna.jobdefs.model import OptionDef
from cryo_savanna.project.presets import PresetError, PresetStore, config_root, file_name, is_input


def test_config_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYO_SAVANNA_CONFIG", str(tmp_path / "c"))
    assert config_root() == tmp_path / "c"
    monkeypatch.delenv("CRYO_SAVANNA_CONFIG")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "x"))
    assert config_root() == tmp_path / "x" / "cryo-savanna"


def test_file_names():
    assert file_name("relion.class3d") == "relion.class3d.json"
    assert file_name("relion.class3d:tomo") == "relion.class3d__tomo.json"
    assert file_name("cli:relion_image_handler") == "cli__relion_image_handler.json"
    assert file_name("weird/key name") == "weird_key_name.json"


def test_is_input():
    def opt(kind):
        return OptionDef(key="k", kind=kind, label="", default="")
    assert all(is_input(opt(k)) for k in ("inputnode", "filename", "folder"))
    assert not any(is_input(opt(k)) for k in ("any", "radio", "boolean", "slider"))


def test_save_get_rename_duplicate_delete(tmp_path):
    s = PresetStore(tmp_path)
    s.save("relion.class3d", "big", {"nr_classes": "8", "tau_fudge": "4"}, "5.1.1")
    s.save("relion.class3d", "Alpha", {"nr_classes": "3"})
    assert s.names("relion.class3d") == ["Alpha", "big"]  # case not counted
    assert s.names("relion.class2d") == []
    p = s.get("relion.class3d", "big")
    assert p.options == {"nr_classes": "8", "tau_fudge": "4"} and p.relion_version == "5.1.1" and p.saved
    s.set_options("relion.class3d", "big", {"nr_classes": "10"})
    assert s.get("relion.class3d", "big").options == {"nr_classes": "10"}
    assert s.get("relion.class3d", "big").relion_version == "5.1.1"  # kept
    s.rename("relion.class3d", "big", "K10")
    assert s.names("relion.class3d") == ["Alpha", "K10"]
    with pytest.raises(PresetError):
        s.rename("relion.class3d", "Alpha", "K10")  # taken
    assert s.duplicate("relion.class3d", "K10") == "K10 (copy)"
    assert s.duplicate("relion.class3d", "K10") == "K10 (copy 2)"
    s.delete("relion.class3d", "K10 (copy)")
    assert s.names("relion.class3d") == ["Alpha", "K10", "K10 (copy 2)"]
    data = json.loads((tmp_path / "relion.class3d.json").read_text())
    assert data["format"] == 1 and set(data["presets"]) == {"Alpha", "K10", "K10 (copy 2)"}


def test_names_are_checked(tmp_path):
    s = PresetStore(tmp_path)
    for bad in ("", "   ", "two\nlines", "x" * 121):
        with pytest.raises(PresetError):
            s.save("relion.class2d", bad, {})
    assert s.save("relion.class2d", "  padded  ", {}).name == "padded"


def test_unknown_keys_and_damaged_files(tmp_path):
    s = PresetStore(tmp_path)
    (tmp_path / "relion.class2d.json").write_text("{not json")
    assert s.names("relion.class2d") == []  # read as empty, written again whole
    s.save("relion.class2d", "a", {"nr_classes": "50", "an_option_of_relion_6": "x"})
    assert s.get("relion.class2d", "a").options["an_option_of_relion_6"] == "x"  # kept


def _save_ten(root: str, i: int) -> None:
    s = PresetStore(root)
    for j in range(10):
        s.save("relion.class2d", f"p{i}-{j}", {"k": str(j)})


def test_two_windows_saving_at_once_lose_nothing(tmp_path):
    """Separate processes, as two cryo-savanna windows are (the file lock is per process)."""
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_save_ten, args=(str(tmp_path), i)) for i in range(4)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(60)
    assert all(pr.exitcode == 0 for pr in procs)
    assert len(PresetStore(tmp_path).names("relion.class2d")) == 40


# ---------------------------------------------------------------------------- the job editor and dialogs

def _editor(qtbot, spa_project):
    pytest.importorskip("pytestqt")
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.project.pipeline import Pipeline
    from cryo_savanna.project.snapshot import job_type_names
    from cryo_savanna.ui.job_editor import JobEditor
    defs = _defs()
    e = JobEditor(RelionEnv({}), job_type_names(defs), flag_source=FlagSource(which=lambda p: None),
                  relion_version=defs.version)
    qtbot.addWidget(e)
    e.set_project(spa_project, Pipeline.read(spa_project))
    return e


def _defs():
    from cryo_savanna.jobdefs.model import JobDefs
    return JobDefs.load()


def _accept_save(monkeypatch, name, change=None):
    """The save dialog, answered: this name, and the ticks as they are (or changed by ``change``)."""
    from qtpy.QtWidgets import QDialog
    from cryo_savanna.ui import presets_dialog

    def fake_exec(dlg):
        dlg.name.setText(name)
        if change is not None:
            change(dlg)
        return QDialog.Accepted
    monkeypatch.setattr(presets_dialog.SavePresetDialog, "exec", fake_exec)


def test_save_ticks_changed_options_but_not_inputs(qtbot, spa_project, monkeypatch):
    e = _editor(qtbot, spa_project)
    e.new_job(_defs().variant("relion.class2d"))
    e.widgets["fn_img"].set_value("Extract/job007/particles.star")  # an input: changed, not ticked
    e.widgets["nr_classes"].set_value("100")
    e.widgets["tau_fudge"].set_value("3")
    _accept_save(monkeypatch, "K100")
    assert e.save_preset_as() == "K100"
    p = e.presets.get("relion.class2d", "K100")
    assert p.options == {"nr_classes": "100", "tau_fudge": "3"} and p.relion_version == e.relion_version
    assert e.loaded_preset == "K100" and "saved" in e.status.text()


def test_load_keeps_inputs_and_resets_the_rest(qtbot, spa_project):
    e = _editor(qtbot, spa_project)
    e.presets.save("relion.class2d", "K100", {"nr_classes": "100", "an_old_option": "1"})
    e.new_job(_defs().variant("relion.class2d"))
    e.widgets["fn_img"].set_value("Extract/job007/particles.star")
    e.widgets["nr_iter_grad"].set_value("300")  # not in the preset: back to the default
    assert e.load_preset("K100")
    assert e.widgets["nr_classes"].value() == "100"
    assert e.widgets["fn_img"].value() == "Extract/job007/particles.star"
    assert e.widgets["nr_iter_grad"].value() == e.options["nr_iter_grad"].default
    assert "1 option" in e.status.text() and "an_old_option" in e.status.text()
    assert not e.load_preset("nope")


def test_a_preset_with_an_input_sets_it(qtbot, spa_project, monkeypatch):
    e = _editor(qtbot, spa_project)
    e.new_job(_defs().variant("relion.class2d"))
    e.widgets["fn_img"].set_value("Extract/job007/particles.star")

    def tick_input(dlg):
        row = dlg.keys.index("fn_img")
        from qtpy.QtCore import Qt
        dlg.table.item(row, 0).setCheckState(Qt.Checked)
    _accept_save(monkeypatch, "with input", tick_input)
    e.save_preset_as()
    e.widgets["fn_img"].set_value("Select/job009/particles.star")
    e.load_preset("with input")
    assert e.widgets["fn_img"].value() == "Extract/job007/particles.star"


def test_update_saves_the_same_options_with_new_values(qtbot, spa_project):
    e = _editor(qtbot, spa_project)
    e.presets.save("relion.class2d", "K", {"nr_classes": "100"})
    e.new_job(_defs().variant("relion.class2d"))
    e.load_preset("K")
    e.widgets["nr_classes"].set_value("150")
    e.widgets["tau_fudge"].set_value("4")  # not in the preset: not added
    assert e.update_preset()
    assert e.presets.get("relion.class2d", "K").options == {"nr_classes": "150"}


def test_menu_and_modes(qtbot, spa_project):
    from cryo_savanna.project.jobstar import JobStar
    e = _editor(qtbot, spa_project)
    assert not e.presets_btn.isEnabled()  # no job type yet
    e.presets.save("relion.class2d", "K", {"nr_classes": "100"})
    e.new_job(_defs().variant("relion.class2d"))
    assert e.presets_btn.isEnabled()
    e._fill_presets_menu()
    texts = [a.text() for a in e.presets_menu.actions()]
    assert texts[0] == "K" and "Save current settings as…" in texts and "Manage presets…" in texts
    assert not any(t.startswith("Update") for t in texts)
    e.load_preset("K")
    e._fill_presets_menu()
    assert any(a.text() == "Update “K” with the current settings" for a in e.presets_menu.actions())
    js = JobStar.read(spa_project / "Class2D/job006/job.star")
    e.continue_job("Class2D/job006/", js, _defs().variant("relion.class2d"))
    assert not e.presets_btn.isEnabled() and not e.load_preset("K")
    e.overwrite_job("Class2D/job006/", js, _defs().variant("relion.class2d"))
    assert e.presets_btn.isEnabled() and e.loaded_preset is None


def test_presets_of_cli_and_pytom_forms(qtbot, spa_project):
    from cryo_savanna.env.relion_env import RelionEnv
    from cryo_savanna.jobdefs.cli_tools import CliCatalog
    from cryo_savanna.jobdefs.flags import FlagSource
    from cryo_savanna.jobdefs.pytom_jobs import PytomCatalog
    e = _editor(qtbot, spa_project)
    cli = CliCatalog(_defs(), RelionEnv({}), FlagSource(which=lambda p: None)).variant("relion_image_handler")
    e.new_job(cli)
    key = next(k for k, o in e.options.items() if o.kind == "any" and not k.startswith(("nr_", "do_", "other")))
    e.presets.save(cli.key, "p", {key: "7"})
    assert e.load_preset("p") and e.widgets[key].value() == "7"
    assert (e.presets.root / "cli__relion_image_handler.json").is_file()
    pytom = PytomCatalog(_defs(), RelionEnv({})).variant("match")
    e.new_job(pytom)
    e.presets.save(pytom.key, "q", {"angular_search": "7.00"})
    assert e.load_preset("q") and e.values()["angular_search"] == "7.00"


def test_manage_dialog_edits_renames_and_deletes(qtbot, tmp_path, monkeypatch):
    pytest.importorskip("pytestqt")
    from qtpy.QtWidgets import QInputDialog, QMessageBox
    from cryo_savanna.ui.presets_dialog import ManagePresetsDialog
    v = _defs().variant("relion.class2d")
    store = PresetStore(tmp_path)
    store.save(v.key, "K", {"nr_classes": "100", "an_old_option": "1"})
    store.save(v.key, "L", {"tau_fudge": "4"})
    d = ManagePresetsDialog(None, store, v.key, "2D classification", v.options, select="K")
    qtbot.addWidget(d)
    assert d.current() == "K" and d.table.rowCount() == 2 and "an_old_option" in d.note.text()
    d.table.item(0, 2).setText("120")  # edited in the table: saved at once
    assert store.get(v.key, "K").options["nr_classes"] == "120"
    d.table.selectRow(1)
    d._remove_option()
    assert store.get(v.key, "K").options == {"nr_classes": "120"}
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: (next(x for x in a[3] if "(tau_fudge)" in x), True))
    d._add_option()
    assert store.get(v.key, "K").options == {"nr_classes": "120", "tau_fudge": v.options["tau_fudge"].default}
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("K new", True))
    d._rename()
    assert store.names(v.key) == ["K new", "L"] and d.current() == "K new"
    d._duplicate()
    assert d.current() == "K new (copy)"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    d._delete()
    assert store.names(v.key) == ["K new", "L"]
