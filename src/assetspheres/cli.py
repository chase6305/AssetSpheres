from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .asset import load_asset
from .export import load_result_asset, read_result, write_outputs
from .fit import fit_asset
from .model import FitConfig, FitResult


def _mapping(values: list[str] | None, *, integer: bool = False) -> dict:
    result = {}
    for value in values or []:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        if key in result:
            raise ValueError(f"Duplicate override for {key!r}")
        result[key] = int(item) if integer else item
    return result


def _summary(result: FitResult) -> None:
    print(f"{'Link':30} {'Method':9} {'Spheres':>7} {'Surface':>9} {'Gap p95 (m)':>12}")
    for name, spheres in result.links.items():
        report = result.reports.get(name, {})
        stale = report.get("quality_metrics_stale", False)
        coverage = report.get("surface_coverage")
        gap = report.get("surface_gap_p95_m")
        coverage_text = f"{coverage:.1%}" if coverage is not None and not stale else "n/a"
        gap_text = f"{gap:.6f}" if gap is not None and not stale else "n/a"
        method = "manual" if stale else report.get("effective_method", "?")
        print(f"{name:30} {method:9} {len(spheres):7} {coverage_text:>9} {gap_text:>12}")
        if stale:
            print(f"  {name}: Manually edited; saved quality metrics are stale", file=sys.stderr)
            continue
        for warning in report.get("warnings", []):
            print(f"  {name}: {warning}", file=sys.stderr)


def _package_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="ROS package directory; repeat for multiple packages",
    )


def _output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    parser.add_argument("--no-glb", action="store_true", help="Skip the static GLB preview")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit link-local collision spheres to URDF/mesh assets"
    )
    parser.add_argument("--version", action="version", version="AssetSpheres 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit a URDF or standalone triangle mesh")
    fit.add_argument("input", type=Path)
    _output_arguments(fit)
    _package_argument(fit)
    fit.add_argument("--method", choices=["voxel", "surface", "cover", "lula"], default="voxel")
    fit.add_argument("--count", type=int, default=32, help="Maximum spheres PER LINK (default: 32)")
    fit.add_argument(
        "--link-count",
        action="append",
        metavar="LINK=N",
        default=[],
        help="Override a link's sphere budget; repeatable",
    )
    fit.add_argument("--link", action="append", help="Only fit this link; repeatable")
    fit.add_argument(
        "--geometry",
        choices=["auto", "collision", "visual"],
        default="auto",
        help="auto uses collision geometry, or visual when collision is absent",
    )
    fit.add_argument(
        "--scale", type=float, default=1.0, help="Standalone mesh unit scale (mm: 0.001)"
    )
    fit.add_argument("--resolution", type=int, default=48, help="Voxels along the longest axis")
    fit.add_argument("--voxel-size", type=float, help="Explicit voxel pitch in metres")
    fit.add_argument(
        "--padding", type=float, default=0.0, help="Nonnegative radius offset in metres"
    )
    fit.add_argument("--surface-radius", type=float, default=0.01, help="Surface sphere radius (m)")
    fit.add_argument("--samples", type=int, default=6000)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument(
        "--non-watertight",
        choices=["error", "surface", "cover"],
        default="error",
        help="Explicit fallback for open meshes with a volume backend",
    )
    view = commands.add_parser("view", help="Inspect and edit a result with the optional Viser UI")
    view.add_argument("result", type=Path)
    view.add_argument("--asset", type=Path, help="Override the original asset path")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8890)
    view.add_argument("--save", type=Path, help="Edited JSON path (default: spheres.edited.json)")
    _package_argument(view)
    export = commands.add_parser("export", help="Export an existing or manually edited JSON result")
    export.add_argument("result", type=Path)
    export.add_argument("--asset", type=Path)
    _output_arguments(export)
    _package_argument(export)
    report = commands.add_parser("report", help="Print saved quality metrics")
    report.add_argument("result", type=Path)
    report.add_argument("--json", action="store_true", help="Print the complete stored reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fit":
            cfg = FitConfig(
                method=args.method,
                count=args.count,
                resolution=args.resolution,
                voxel_size=args.voxel_size,
                padding=args.padding,
                surface_radius=args.surface_radius,
                samples=args.samples,
                seed=args.seed,
                non_watertight=args.non_watertight,
            )
            asset = load_asset(
                args.input,
                geometry=args.geometry,
                link_names=args.link,
                package_roots=_mapping(args.package_root),
                scale=args.scale,
            )
            print(f"Fitting {len(asset.links)} link(s) with {args.method}...", flush=True)
            result = fit_asset(asset, cfg, link_counts=_mapping(args.link_count, integer=True))
            paths = write_outputs(
                asset, result, args.output, force=args.force, include_glb=not args.no_glb
            )
            _summary(result)
            print(f"Wrote {len(paths)} files to {paths[0].parent}")
        elif args.command == "report":
            result = read_result(args.result)
            if args.json:
                print(json.dumps(result.reports, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                _summary(result)
        elif args.command in {"view", "export"}:
            result = read_result(args.result)
            asset = load_result_asset(
                result, path=args.asset, package_roots=_mapping(args.package_root)
            )
            if args.command == "export":
                paths = write_outputs(
                    asset, result, args.output, force=args.force, include_glb=not args.no_glb
                )
                print(f"Wrote {len(paths)} files to {paths[0].parent}")
            else:
                from .viewer import serve

                serve(
                    asset,
                    result,
                    host=args.host,
                    port=args.port,
                    save_path=args.save or args.result.with_name("spheres.edited.json"),
                )
    except (ValueError, OSError, ImportError) as exc:
        print(f"assetspheres: {exc}", file=sys.stderr)
        return 2
    return 0
