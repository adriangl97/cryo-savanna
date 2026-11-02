# Copyright (C) 2026 Adrián González-López
# SPDX-License-Identifier: GPL-2.0-or-later
import os

import pytest

from cryo_savanna.project.star import (
    StarBlock,
    StarError,
    escape_string,
    format_block,
    format_star,
    format_value,
    parse_star_text,
    read_star,
    tokenize_line,
    write_star,
)


@pytest.mark.parametrize(
    "line, tokens",
    [
        ("a b   c", ["a", "b", "c"]),
        ("a b # comment", ["a", "b"]),
        ("a#b c", ["a#b", "c"]),  # '#' inside a token is not a comment
        ('"with space" x', ["with space", "x"]),
        ("'single quoted' x", ["single quoted", "x"]),
        ('""', [""]),
        ('"say "hi\a" now" x', ['say "hi" now', "x"]),
        ('"a"b" c', ['a"b', "c"]),  # a quote not followed by whitespace does not close the token
    ],
)
def test_tokenize(line, tokens):
    assert tokenize_line(line) == tokens


def test_tokenize_unclosed_quote():
    with pytest.raises(StarError):
        tokenize_line('"never closed')


@pytest.mark.parametrize("value", ["", "plain", "with space", 'he said "hi" ok', '"quoted"', "tab\there", "'x'"])
def test_escape_round_trip(value):
    assert tokenize_line(escape_string(value) + " next") == [value, "next"]


def test_escape_matches_relion():
    assert escape_string("") == '""'
    assert escape_string("abc") == "abc"
    assert escape_string("a b") == '"a b"'
    assert escape_string('x" y') == '"x\a" y"'


def test_format_value_numbers():
    assert format_value(True) == "           1"
    assert format_value(7) == "           7"
    assert format_value(1.5) == "    1.500000"
    assert format_value(-1.5) == "    -1.50000"  # RELION uses %12.5f for negative values
    assert format_value(0.0001) == "1.000000e-04"
    assert format_value(123456.0) == "1.234560e+05"
    assert format_value(0.0) == "    0.000000"


def test_list_block_layout_matches_relion():
    # Layout of the data_job block of a job.star written by RELION 5.1
    b = StarBlock.from_dict("job", {"rlnJobTypeLabel": "relion.class2d", "rlnJobIsContinue": False, "rlnJobIsTomo": False})
    assert format_block(b) == (
        "\n# version 50001\n\ndata_job\n\n"
        "_rlnJobTypeLabel             relion.class2d\n"
        "_rlnJobIsContinue                       0\n"
        "_rlnJobIsTomo                           0\n"
        " \n"
    )


def test_loop_block_layout_matches_relion():
    b = StarBlock.from_records("joboptions_values", ["rlnJobOptionVariable", "rlnJobOptionValue"],
                               [["do_queue", "No"], ["other_args", ""], ["fn_img", "Extract/job005/particles.star"]])
    assert format_block(b) == (
        "\n# version 50001\n\ndata_joboptions_values\n\nloop_ \n"
        "_rlnJobOptionVariable #1 \n_rlnJobOptionValue #2 \n"
        "  do_queue         No \n"
        "other_args         \"\" \n"
        "    fn_img Extract/job005/particles.star \n"
        " \n"
    )


def test_empty_block_is_not_written():
    assert format_block(StarBlock.from_records("x", ["a"], [])) == ""


def test_parse_round_trip():
    blocks = [
        StarBlock.from_dict("general", {"rlnFinalResolution": 3.2, "rlnMaskName": "Mask Create/mask.mrc"}),
        StarBlock.from_records("fsc", ["rlnSpectralIndex", "rlnResolution", "rlnName"],
                               [[1, 0.01, "a"], [2, 0.02, "b c"], [3, 0.03, '']]),
    ]
    text = "".join(format_block(b) for b in blocks)
    parsed = parse_star_text(text)
    assert list(parsed) == ["general", "fsc"]
    assert parsed["general"].is_loop is False
    assert parsed["general"].as_dict() == {"rlnFinalResolution": "3.200000", "rlnMaskName": "Mask Create/mask.mrc"}
    assert parsed["general"].version == 50001
    assert parsed["fsc"].column("rlnName") == ["a", "b c", ""]
    assert parsed["fsc"].column("rlnSpectralIndex") == ["1", "2", "3"]


def test_parse_selected_blocks_and_underscore_values():
    text = "data_a\nloop_\n_rlnX\n_rlnY\n1 _value\n2 x\n\ndata_b\n_rlnZ 5\n"
    parsed = parse_star_text(text, blocks=["a"])
    assert list(parsed) == ["a"]
    assert parsed["a"].rows == [["1", "_value"], ["2", "x"]]


def test_parse_wrong_column_count():
    with pytest.raises(StarError):
        parse_star_text("data_a\nloop_\n_rlnX\n_rlnY\n1 2 3\n")


def test_list_block_label_followed_by_a_tab():
    parsed = parse_star_text('data_a\n\n_rlnName\t"a b"\n_rlnX 3\n_rlnEmpty\n')["a"]
    assert parsed.as_dict() == {"rlnName": "a b", "rlnX": "3", "rlnEmpty": ""}


def test_values_starting_with_a_hash_survive_and_short_two_column_rows_are_read():
    from cryo_savanna.project.jobstar import JobStar
    js = JobStar("relion.external", options={"fn_exe": "a.sh", "other_args": "#test"})
    assert '"#test"' in js.to_text()
    assert parse_star_text(js.to_text())["joboptions_values"].rows[1] == ["other_args", "#test"]
    # RELION's readStarLoop accepts a missing value in a two-column loop ("fn_mtf <empty>")
    text = "data_t\nloop_\n_rlnJobOptionVariable #1\n_rlnJobOptionValue #2\nfn_mtf\nangpix 1.0\n"
    assert parse_star_text(text)["t"].rows == [["fn_mtf", ""], ["angpix", "1.0"]]
    with pytest.raises(StarError):
        parse_star_text("data_t\nloop_\n_a #1\n_b #2\n_c #3\n1 2\n")


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write any file")
@pytest.mark.parametrize("atomic", [True, False])
def test_a_file_this_user_may_not_write_is_not_replaced(tmp_path, atomic):
    """A rename would replace it (the directory is writable) and take it from its owner's RELION."""
    path = tmp_path / "default_pipeline.star"
    path.write_text("theirs\n")
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="Not writable"):
        write_star(path, [StarBlock.from_dict("g", {"rlnX": 1})], atomic=atomic)
    assert path.read_text() == "theirs\n" and list(tmp_path.iterdir()) == [path]


def test_a_symlink_is_written_through(tmp_path):
    """As RELION writes it: the file the link points to changes, the link stays a link."""
    real, link = tmp_path / "real.star", tmp_path / "default_pipeline.star"
    write_star(real, [StarBlock.from_dict("job", {"a": 1})])
    os.symlink(real.name, link)
    write_star(link, [StarBlock.from_dict("job", {"a": 2})])
    assert link.is_symlink() and read_star(real)["job"].get("a") == "2"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["default_pipeline.star", "real.star"]


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "x.star"
    write_star(path, [StarBlock.from_dict("job", {"a": 1})])

    def full_disk(*args):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(os, "fsync", full_disk)
    with pytest.raises(OSError):
        write_star(path, [StarBlock.from_dict("job", {"a": 2})])
    assert [p.name for p in tmp_path.iterdir()] == ["x.star"] and read_star(path)["job"].get("a") == "1"


@pytest.mark.parametrize("value", ["_mic001.mrc", "data_001.mrc", "loop_x", ";note", "#1"])
def test_a_row_that_would_not_read_as_values_is_quoted(value):
    """RELION writes these bare at the start of a row, and readers (RELION's too) then take the row for a
    label, a new block or loop, or a comment: quoted, they read back as the values they are."""
    b = StarBlock.from_records("t", ["rlnA", "rlnB"], [[value, "z"], ["y", value]])
    text = format_star([b])
    assert parse_star_text(text)["t"].rows == [[value, "z"], ["y", value]]
    # quoted where a row starts; elsewhere as RELION writes it (a "#" value is quoted anywhere already)
    assert text.count(f'"{value}"') == (2 if value.startswith("#") else 1)
