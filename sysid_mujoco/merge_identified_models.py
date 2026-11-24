"""Merge independently identified joint parameters into one MuJoCo XML.

The single-joint CMA-ES runs intentionally produce one XML per joint.  This
utility combines selected ``best_params.json`` files without rerunning an
optimizer and records the exact source files used for the merged model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-xml", type=Path, required=True)
    parser.add_argument("--best-params", type=Path, nargs="+", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write identified_fixed_base.xml and merged_params.json here. "
            "This preserves the original fixed-base workflow."
        ),
    )
    output.add_argument(
        "--output-xml",
        type=Path,
        help=(
            "Write the merged model to this exact path and place its manifest "
            "beside it as <stem>_params.json. This is useful when relative mesh "
            "paths require the output to remain beside the source model."
        ),
    )
    return parser.parse_args()


def merge_parameter_files(paths: list[Path]) -> tuple[str, dict[str, dict[str, float]]]:
    robot: str | None = None
    merged: dict[str, dict[str, float]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_robot = payload.get("robot")
        if robot is None:
            robot = source_robot
        elif source_robot != robot:
            raise ValueError(f"Robot mismatch in {path}: {source_robot!r} != {robot!r}")
        for joint, values in payload["parameters"].items():
            if joint in merged:
                raise ValueError(f"Joint {joint!r} appears in more than one parameter file.")
            merged[joint] = {name: float(value) for name, value in values.items()}
    if robot is None or not merged:
        raise ValueError("No identified parameters were loaded.")
    return robot, merged


def write_merged_xml(
    base_xml: Path,
    output_xml: Path,
    parameters: dict[str, dict[str, float]],
) -> None:
    tree = ET.parse(base_xml)
    root = tree.getroot()
    found: set[str] = set()
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name not in parameters:
            continue
        found.add(name)
        for attribute, value in parameters[name].items():
            joint.set(attribute, f"{value:.12g}")
    missing = sorted(set(parameters) - found)
    if missing:
        raise ValueError(f"Identified joints are missing from the base XML: {missing}")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_xml is not None:
        output_xml = args.output_xml.expanduser().resolve()
        manifest_path = output_xml.with_name(f"{output_xml.stem}_params.json")
    else:
        output_dir = args.output_dir.expanduser().resolve()
        output_xml = output_dir / "identified_fixed_base.xml"
        manifest_path = output_dir / "merged_params.json"
    return output_xml, manifest_path


def main() -> None:
    args = parse_args()
    base_xml = args.base_xml.expanduser().resolve()
    parameter_paths = [path.expanduser().resolve() for path in args.best_params]
    if not base_xml.is_file():
        raise FileNotFoundError(base_xml)
    for path in parameter_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    robot, parameters = merge_parameter_files(parameter_paths)
    output_xml, manifest_path = resolve_output_paths(args)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    write_merged_xml(base_xml, output_xml, parameters)
    manifest = {
        "robot": robot,
        "base_xml": str(base_xml),
        "sources": [str(path) for path in parameter_paths],
        "parameters": parameters,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged {len(parameters)} joints into {output_xml}")
    print(f"Wrote parameter manifest to {manifest_path}")


if __name__ == "__main__":
    main()
