"""User-facing CLI behavior: discover, inspect, fit, and consume saved results."""

import json
import shlex
from pathlib import Path

import numpy as np
import pytest
import trimesh

from assetspheres.cli import build_parser, main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf"


@pytest.mark.parametrize(
    "command", [[], ["fit"], ["inspect"], ["methods"], ["report"], ["export"], ["view"]]
)
def test_help_has_consistent_program_name_and_success_exit(command, capsys):
    with pytest.raises(SystemExit) as outcome:
        main([*command, "--help"])
    assert outcome.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: assetspheres" in help_text
    assert "(default: None)" not in help_text
    if command == ["fit"]:
        for group in [
            "input geometry:",
            "output files:",
            "sphere fitting:",
            "CPU and CUDA execution:",
            "voxelization:",
            "experimental refined method:",
        ]:
            assert group in help_text
        assert "PER LINK" in help_text and "default: 32" in help_text
        assert "outputs/<input stem>" in help_text


def test_methods_json_lists_supported_choices_without_human_text(capsys):
    assert main(["methods", "--json"]) == 0
    output = capsys.readouterr()
    assert not output.err
    description = json.loads(output.out)
    assert description["default"] == "voxel"
    methods = {method["name"]: method for method in description["methods"]}
    assert set(methods) == {
        "voxel",
        "surface",
        "cover",
        "cover-tight",
        "cover-opt",
        "slab",
        "refined",
        "lula",
    }
    assert methods["voxel"]["closed_mesh_required"] is True
    assert methods["surface"]["closed_mesh_required"] is False
    assert methods["cover-opt"]["closed_mesh_required"] is False
    assert methods["cover-opt"]["devices"] == ["cpu"]
    assert methods["refined"]["devices"] == ["cpu", "cuda"]
    assert "Isaac Sim" in methods["lula"]["dependencies"]


def test_methods_text_explains_coverage_and_runtime_scope(capsys):
    assert main(["methods"]) == 0
    output = capsys.readouterr().out
    assert "Method" in output and "Coverage" in output
    assert "conditional solid enclosure" in output
    assert "not a check" in output


def test_inspect_json_has_link_geometry_and_unit_information(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["inspect", str(EXAMPLE), "--json"]) == 0
    output = capsys.readouterr()
    assert not output.err
    inspection = json.loads(output.out)
    assert inspection["source"] == str(EXAMPLE)
    assert inspection["units"] == "metre"
    assert inspection["sphere_frame"] == "link_local"
    assert inspection["link_count"] == inspection["tree_link_count"] == 2
    links = {link["name"]: link for link in inspection["links"]}
    assert links["base"]["watertight"] is True
    assert links["base"]["triangle_count"] == 12
    assert links["base"]["body_count"] == 1
    assert np.allclose(links["base"]["extents_m"], [0.2, 0.18, 0.12])
    assert links["arm"]["geometry_types"] == ["cylinder", "sphere"]
    assert links["arm"]["body_count"] == 2
    assert not (tmp_path / "outputs").exists()


def test_inspect_respects_selected_link_and_geometry(capsys):
    assert main(["inspect", str(EXAMPLE), "--link", "arm", "--geometry", "visual", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["link_count"] == 1
    assert inspection["tree_link_count"] == 2
    assert inspection["links"][0]["name"] == "arm"
    assert inspection["links"][0]["geometry_source"] == "visual"


def test_inspect_mesh_scale_and_human_output(tmp_path, capsys):
    source = tmp_path / "millimetre-part.stl"
    trimesh.creation.box(extents=(100, 200, 300)).export(source)
    assert main(["inspect", str(source), "--scale", ".001", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["format"] == "mesh"
    assert np.allclose(inspection["links"][0]["extents_m"], [0.1, 0.2, 0.3])
    assert main(["inspect", str(source), "--scale", ".001"]) == 0
    text = capsys.readouterr().out
    assert "millimetre-part" in text and "Size x/y/z (m)" in text
    assert "closed" in text and "metre" in text


def test_fit_uses_default_output_and_prints_shell_safe_next_commands(tmp_path, monkeypatch, capsys):
    work_directory = tmp_path / "workspace with spaces"
    work_directory.mkdir()
    monkeypatch.chdir(work_directory)
    assert (
        main(
            [
                "fit",
                str(EXAMPLE),
                "--method",
                "surface",
                "--count",
                "2",
                "--link-count",
                "arm=3",
                "--samples",
                "128",
                "--no-glb",
            ]
        )
        == 0
    )
    output_directory = work_directory / "outputs" / "two_link"
    result_path = output_directory / "spheres.json"
    result = json.loads(result_path.read_text())
    assert [len(spheres) for spheres in result["links"].values()] == [2, 3]
    assert result["config"]["voxelizer"] == "subdivide"
    assert not (output_directory / "preview.glb").exists()
    output = capsys.readouterr().out
    report_command = next(
        line.partition(": ")[2] for line in output.splitlines() if line.startswith("Report: ")
    )
    assert shlex.split(report_command) == ["assetspheres", "report", str(result_path)]
    view_command = next(
        line.partition(":")[2].strip() for line in output.splitlines() if line.startswith("View:")
    )
    assert shlex.split(view_command) == ["assetspheres", "view", str(result_path)]
    assert main(["report", str(result_path), "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert reports["arm"]["sphere_count"] == 3


def test_default_output_never_silently_replaces_existing_results(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    arguments = [
        "fit",
        str(EXAMPLE),
        "--method",
        "surface",
        "--count",
        "2",
        "--samples",
        "128",
        "--no-glb",
    ]
    assert main(arguments) == 0
    result_path = tmp_path / "outputs" / "two_link" / "spheres.json"
    original = result_path.read_bytes()
    capsys.readouterr()
    assert main(arguments) == 2
    assert "--force" in capsys.readouterr().err
    assert result_path.read_bytes() == original
    assert main([*arguments, "--force"]) == 0


def test_explicit_output_and_legacy_flags_remain_supported(tmp_path, capsys):
    output = tmp_path / "chosen-output"
    assert (
        main(
            [
                "fit",
                str(EXAMPLE),
                "-o",
                str(output),
                "--method",
                "slab",
                "--count",
                "3",
                "--link",
                "arm",
                "--workers",
                "1",
                "--executor",
                "thread",
                "--samples",
                "128",
                "--no-glb",
            ]
        )
        == 0
    )
    result = json.loads((output / "spheres.json").read_text())
    assert list(result["links"]) == ["arm"]
    capsys.readouterr()
    assert (
        main(["export", str(output / "spheres.json"), "-o", str(tmp_path / "export"), "--no-glb"])
        == 0
    )
    assert (tmp_path / "export" / "spheres.urdf").is_file()


@pytest.mark.parametrize(
    "override,expected",
    [
        (["--link-count", "arm"], "Expected NAME=VALUE"),
        (["--link-count", "arm=abc"], "Expected an integer"),
        (["--link-count", "arm=2", "--link-count", "arm=3"], "Duplicate override"),
        (["--package-root", "robot"], "Expected NAME=VALUE"),
    ],
)
def test_assignment_errors_are_concise_and_actionable(override, expected, capsys):
    assert main(["fit", str(EXAMPLE), "--method", "surface", *override]) == 2
    error = capsys.readouterr().err
    assert expected in error
    assert "Traceback" not in error


def test_missing_file_message_and_parser_error_codes(capsys):
    assert main(["inspect", "missing.urdf"]) == 2
    error = capsys.readouterr().err
    assert "missing.urdf" in error and "Check the input/result path" in error
    assert "Traceback" not in error
    with pytest.raises(SystemExit) as outcome:
        main(["fit", str(EXAMPLE), "--method", "unknown"])
    assert outcome.value.code == 2
    with pytest.raises(SystemExit) as outcome:
        main(["export", "saved.json"])
    assert outcome.value.code == 2


@pytest.mark.parametrize("face", ["1 2 99", "-10 -2 -1"])
def test_invalid_obj_face_indices_have_a_concise_cli_error(tmp_path, capsys, face):
    source = tmp_path / "invalid.obj"
    source.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nf {face}\n")
    assert main(["inspect", str(source), "--json"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "Invalid triangle mesh" in output.err and str(source) in output.err
    assert "Traceback" not in output.err


def test_new_voxelizer_options_are_explicit_and_default_to_legacy_backend():
    parser = build_parser()
    defaults = parser.parse_args(["fit", str(EXAMPLE)])
    assert defaults.voxelizer == "subdivide" and defaults.voxel_solid_policy == "single"
    selected = parser.parse_args(
        ["fit", str(EXAMPLE), "--voxelizer", "warp", "--voxel-solid-policy", "union-components"]
    )
    assert selected.voxelizer == "warp" and selected.voxel_solid_policy == "union-components"
