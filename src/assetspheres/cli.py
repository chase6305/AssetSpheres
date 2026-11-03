"""Command-line asset inspection, fitting, and result management.

Parsing, configuration construction, and command execution are kept separate so
new fitting options do not turn the entry point into one large conditional.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .asset import Asset, load_asset
from .export import load_result_asset, read_result, write_outputs
from .fit import fit_asset
from .model import FitConfig, FitResult


class CommandHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show option defaults and preserve the command examples' line breaks."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


@dataclass(frozen=True)
class MethodInfo:
    name: str
    closed_mesh_required: bool
    devices: tuple[str, ...]
    coverage: str
    dependencies: str


METHODS = (
    MethodInfo("voxel", True, ("cpu", "cuda"), "Approximate filled volume", "Core; torch for CUDA"),
    MethodInfo("surface", False, ("cpu",), "Sampled surface neighborhoods", "Core"),
    MethodInfo("cover", False, ("cpu",), "Complete triangle surface", "Core"),
    MethodInfo("cover-tight", False, ("cpu",), "Tighter complete triangle surface", "Core"),
    MethodInfo("slab", False, ("cpu",), "Complete slices; conditional solid enclosure", "Core"),
    MethodInfo(
        "refined",
        True,
        ("cpu", "cuda"),
        "Sampled coverage/excess-volume trade-off",
        "Core; torch for CUDA; rtree for mesh labels",
    ),
    MethodInfo(
        "lula", True, ("cpu",), "Native installed Lula generator", "Isaac Sim Lula bindings"
    ),
)


def _parse_overrides(values: list[str] | None, *, integer: bool = False) -> dict:
    """Parse repeatable NAME=VALUE options without silently replacing duplicates."""
    overrides = {}
    for value in values or []:
        name, separator, setting = value.partition("=")
        if not separator or not name or not setting:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        if name in overrides:
            raise ValueError(f"Duplicate override for {name!r}")
        if integer:
            try:
                setting = int(setting)
            except ValueError as error:
                raise ValueError(f"Expected an integer in {name}=N, got {setting!r}") from error
        overrides[name] = setting
    return overrides


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def _print_fit_summary(result: FitResult) -> None:
    rows = []
    for link_name, spheres in result.links.items():
        report = result.reports.get(link_name, {})
        metrics_stale = report.get("quality_metrics_stale", False)
        coverage = report.get("surface_coverage")
        gap = report.get("surface_gap_p95_m")
        rows.append(
            (
                link_name,
                "manual" if metrics_stale else report.get("effective_method", "?"),
                str(len(spheres)),
                f"{coverage:.1%}" if coverage is not None and not metrics_stale else "n/a",
                f"{gap:.6f}" if gap is not None and not metrics_stale else "n/a",
            )
        )
        if metrics_stale:
            print(
                f"  {link_name}: Manually edited; saved quality metrics are stale", file=sys.stderr
            )
            continue
        for warning in report.get("warnings", []):
            print(f"  {link_name}: {warning}", file=sys.stderr)
    _print_table(("Link", "Method", "Spheres", "Surface", "Gap p95 (m)"), rows)


def _add_package_argument(parser) -> None:
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="ROS package directory; repeat for multiple packages",
    )


def _add_asset_arguments(parser) -> None:
    parser.add_argument(
        "--geometry",
        choices=["auto", "collision", "visual"],
        default="auto",
        help="auto prefers collision and uses visual only when collision is absent",
    )
    parser.add_argument(
        "--link",
        action="append",
        metavar="NAME",
        help="Load only this link; repeat to select multiple links",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Standalone mesh unit scale; use 0.001 for millimetres",
    )
    _add_package_argument(parser)


def _add_output_arguments(parser, *, required: bool) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=required,
        default=argparse.SUPPRESS,
        help="Output directory" if required else "Output directory; omitted: outputs/<input stem>",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    parser.add_argument("--no-glb", action="store_true", help="Skip the static GLB preview")


def _add_command(commands, name: str, help_text: str, handler, *, epilog: str | None = None):
    command = commands.add_parser(
        name,
        help=help_text,
        description=help_text,
        formatter_class=CommandHelpFormatter,
        epilog=epilog,
    )
    command.set_defaults(handler=handler)
    return command


def _add_fit_command(commands) -> None:
    command = _add_command(
        commands,
        "fit",
        "Fit collision spheres to a URDF or triangle mesh",
        _run_fit_command,
        epilog=(
            "Examples:\n"
            "  assetspheres fit robot.urdf\n"
            "  assetspheres fit robot.urdf --method slab --workers 4\n"
            "  assetspheres fit part.stl --scale 0.001 -o outputs/part\n"
            "Run 'assetspheres methods' to compare fitting methods."
        ),
    )
    command.add_argument("input", type=Path, help="URDF or standalone triangle mesh path")
    _add_asset_arguments(command.add_argument_group("input geometry"))
    _add_output_arguments(command.add_argument_group("output files"), required=False)
    fitting = command.add_argument_group("sphere fitting")
    fitting.add_argument(
        "--method",
        choices=[method.name for method in METHODS],
        default="voxel",
        help="Fitting algorithm; see 'assetspheres methods'",
    )
    fitting.add_argument("--count", type=int, default=32, help="Maximum sphere count PER LINK")
    fitting.add_argument(
        "--link-count",
        action="append",
        default=[],
        metavar="LINK=N",
        help="Override one link's sphere budget; repeatable",
    )
    fitting.add_argument("--padding", type=float, default=0.0, help="Radius offset in metres")
    fitting.add_argument(
        "--surface-radius",
        type=float,
        default=0.01,
        help="Fixed sphere radius for the surface method, in metres",
    )
    fitting.add_argument(
        "--samples", type=int, default=6000, help="Fitting/evaluation sample budget"
    )
    fitting.add_argument("--seed", type=int, default=0, help="Nonnegative random seed")
    fitting.add_argument(
        "--non-watertight",
        choices=["error", "surface", "cover"],
        default="error",
        help="Explicit fallback when a volume method receives an open mesh",
    )
    parallel = command.add_argument_group("CPU and CUDA execution")
    parallel.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="voxel/refined tensor device; CUDA requires compatible PyTorch",
    )
    parallel.add_argument(
        "--workers", type=int, default=1, help="CPU link workers; CUDA fitting uses one GPU owner"
    )
    parallel.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="CPU worker type; processes use spawn",
    )
    voxelization = command.add_argument_group("voxelization")
    voxelization.add_argument(
        "--resolution", type=int, default=48, help="Voxel cells along the longest mesh axis"
    )
    voxelization.add_argument(
        "--voxel-size", type=float, help="Explicit voxel pitch in metres; overrides resolution"
    )
    voxelization.add_argument(
        "--voxelizer",
        choices=["subdivide", "open3d", "warp"],
        default="subdivide",
        help="subdivide: original CPU path; open3d: CPU; warp: CPU/CUDA (optional runtimes)",
    )
    voxelization.add_argument(
        "--voxel-solid-policy",
        choices=["single", "union-components"],
        default="single",
        help="Solid interpretation for optional geometry-query voxelizers",
    )
    refinement = command.add_argument_group("experimental refined method")
    refinement.add_argument(
        "--iterations", type=int, default=120, help="Center/radius optimizer iterations"
    )
    refinement.add_argument(
        "--refine-samples", type=int, default=2048, help="Training points per sample group"
    )
    refinement.add_argument(
        "--refine-reference",
        choices=["voxel", "mesh"],
        default="voxel",
        help="Training labels; mesh needs rtree and a single closed body",
    )
    refinement.add_argument(
        "--outside-weight",
        type=float,
        default=10.0,
        help="Exterior penalty; higher values discourage excess volume",
    )


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="assetspheres",
        description="Turn URDF/mesh assets into link-local collision spheres (metres).",
        formatter_class=CommandHelpFormatter,
        epilog=(
            "Start with:\n"
            "  assetspheres methods\n"
            "  assetspheres inspect robot.urdf\n"
            "  assetspheres fit robot.urdf --method slab"
        ),
    )
    parser.add_argument("--version", action="version", version=f"AssetSpheres {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_fit_command(commands)
    inspect = _add_command(
        commands, "inspect", "Inspect asset links and geometry before fitting", _run_inspect_command
    )
    inspect.add_argument("input", type=Path, help="URDF or standalone mesh path")
    _add_asset_arguments(inspect)
    inspect.add_argument("--json", action="store_true", help="Write only a JSON inspection report")
    methods = _add_command(
        commands, "methods", "Compare algorithms, coverage, and dependencies", _run_methods_command
    )
    methods.add_argument("--json", action="store_true", help="Write only JSON method descriptions")
    report = _add_command(
        commands, "report", "Print quality metrics from a saved result", _run_report_command
    )
    report.add_argument("result", type=Path, help="Saved spheres.json or edited result")
    report.add_argument(
        "--json", action="store_true", help="Write only the complete stored reports"
    )
    export = _add_command(
        commands, "export", "Export an existing or manually edited JSON result", _run_export_command
    )
    export.add_argument("result", type=Path, help="Saved result JSON")
    export.add_argument("--asset", type=Path, help="Override the original asset path")
    _add_output_arguments(export, required=True)
    _add_package_argument(export)
    view = _add_command(
        commands, "view", "Inspect and edit a result using the optional Viser UI", _run_view_command
    )
    view.add_argument("result", type=Path, help="Saved result JSON")
    view.add_argument("--asset", type=Path, help="Override the original asset path")
    view.add_argument("--host", default="127.0.0.1", help="Viewer listen address")
    view.add_argument("--port", type=int, default=8890, help="Viewer listen port")
    view.add_argument("--save", type=Path, help="Edited JSON path; omitted: spheres.edited.json")
    _add_package_argument(view)
    return parser


def _load_input_asset(arguments: argparse.Namespace) -> Asset:
    return load_asset(
        arguments.input,
        geometry=arguments.geometry,
        link_names=arguments.link,
        package_roots=_parse_overrides(arguments.package_root),
        scale=arguments.scale,
    )


def _build_fit_configuration(arguments: argparse.Namespace) -> FitConfig:
    return FitConfig(
        method=arguments.method,
        count=arguments.count,
        resolution=arguments.resolution,
        voxel_size=arguments.voxel_size,
        padding=arguments.padding,
        surface_radius=arguments.surface_radius,
        samples=arguments.samples,
        seed=arguments.seed,
        non_watertight=arguments.non_watertight,
        device=arguments.device,
        iterations=arguments.iterations,
        refine_samples=arguments.refine_samples,
        refine_reference=arguments.refine_reference,
        outside_weight=arguments.outside_weight,
        voxelizer=arguments.voxelizer,
        voxel_solid_policy=arguments.voxel_solid_policy,
    )


def _print_next_commands(result_path: Path) -> None:
    print("Report: " + shlex.join(["assetspheres", "report", str(result_path)]))
    print("View:   " + shlex.join(["assetspheres", "view", str(result_path)]))


def _run_fit_command(arguments: argparse.Namespace) -> None:
    configuration = _build_fit_configuration(arguments)
    asset = _load_input_asset(arguments)
    output_directory = getattr(arguments, "output", Path("outputs") / arguments.input.stem)
    print(f"Fitting {len(asset.links)} link(s) with {configuration.method}...", flush=True)
    result = fit_asset(
        asset,
        configuration,
        link_counts=_parse_overrides(arguments.link_count, integer=True),
        workers=arguments.workers,
        executor=arguments.executor,
    )
    output_paths = write_outputs(
        asset, result, output_directory, force=arguments.force, include_glb=not arguments.no_glb
    )
    _print_fit_summary(result)
    print(f"Wrote {len(output_paths)} files to {output_paths[0].parent}")
    _print_next_commands(output_paths[0])


def _inspect_asset(asset: Asset) -> dict[str, Any]:
    return {
        "source": str(asset.path),
        "format": "urdf" if asset.urdf is not None else "mesh",
        "units": "metre",
        "sphere_frame": "link_local",
        "mesh_scale": asset.mesh_scale,
        "link_count": len(asset.links),
        "tree_link_count": len(asset.zero_pose_transforms),
        "links": [
            {
                "name": link_name,
                "geometry_source": geometry.geometry_source,
                "geometry_types": list(geometry.geometry_types or ("mesh",)),
                "triangle_count": len(geometry.mesh.faces),
                "vertex_count": len(geometry.mesh.vertices),
                "body_count": int(geometry.mesh.body_count),
                "watertight": bool(geometry.mesh.is_watertight),
                "extents_m": geometry.mesh.extents.tolist(),
                "bounds_m": geometry.mesh.bounds.tolist(),
            }
            for link_name, geometry in asset.links.items()
        ],
    }


def _run_inspect_command(arguments: argparse.Namespace) -> None:
    inspection = _inspect_asset(_load_input_asset(arguments))
    if arguments.json:
        print(json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False))
        return
    print(f"Asset: {inspection['source']}")
    print(f"Geometry links: {inspection['link_count']}  |  Units: metre  |  Frame: link-local")
    rows = [
        (
            link["name"],
            link["geometry_source"],
            str(link["triangle_count"]),
            str(link["body_count"]),
            "closed" if link["watertight"] else "open",
            " × ".join(f"{extent:.6g}" for extent in link["extents_m"]),
        )
        for link in inspection["links"]
    ]
    _print_table(("Link", "Geometry", "Triangles", "Bodies", "Mesh", "Size x/y/z (m)"), rows)
    print("Closed topology alone does not prove a mesh has no self-intersections.")


def _run_methods_command(arguments: argparse.Namespace) -> None:
    if arguments.json:
        print(
            json.dumps(
                {"default": "voxel", "methods": [asdict(method) for method in METHODS]},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    rows = [
        (
            method.name,
            "closed" if method.closed_mesh_required else "open/closed",
            "/".join(method.devices),
            method.coverage,
        )
        for method in METHODS
    ]
    _print_table(("Method", "Mesh", "Device", "Coverage"), rows)
    print("\nDevices are supported backends, not a check of this machine's availability.")
    print("CUDA tensor stages need compatible PyTorch; optional voxelizers have separate runtimes.")
    print(
        "Native lula needs Isaac Sim bindings; refined mesh labels need the quality extra (rtree)."
    )
    print("Slab solid enclosure assumes the input is a complete bounded-solid boundary.")


def _run_report_command(arguments: argparse.Namespace) -> None:
    result = read_result(arguments.result)
    if arguments.json:
        print(json.dumps(result.reports, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        _print_fit_summary(result)


def _load_saved_asset(arguments: argparse.Namespace, result: FitResult) -> Asset:
    return load_result_asset(
        result, path=arguments.asset, package_roots=_parse_overrides(arguments.package_root)
    )


def _run_export_command(arguments: argparse.Namespace) -> None:
    result = read_result(arguments.result)
    output_paths = write_outputs(
        _load_saved_asset(arguments, result),
        result,
        arguments.output,
        force=arguments.force,
        include_glb=not arguments.no_glb,
    )
    print(f"Wrote {len(output_paths)} files to {output_paths[0].parent}")
    _print_next_commands(output_paths[0])


def _run_view_command(arguments: argparse.Namespace) -> None:
    from .viewer import serve

    result = read_result(arguments.result)
    save_path = arguments.save or arguments.result.with_name("spheres.edited.json")
    serve(
        _load_saved_asset(arguments, result),
        result,
        host=arguments.host,
        port=arguments.port,
        save_path=save_path,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except FileNotFoundError as error:
        missing_path = error.filename or str(error)
        print(
            f"assetspheres: File not found: {missing_path}. Check the input/result path.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError, ImportError) as error:
        print(f"assetspheres: {error}", file=sys.stderr)
        return 2
    return 0
