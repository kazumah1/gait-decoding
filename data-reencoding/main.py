from __future__ import annotations

import csv
import re
from itertools import zip_longest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = REPO_ROOT / "data" / "RepositoryData"
OUTPUT_PATH = REPO_ROOT / "data" / "gait_decoding_combined.csv"
TRIAL_DIR_PATTERN = re.compile(r"^(SL\d{2})-(T\d{2})$")
JOINT_LABEL_PATTERN = re.compile(r"\(([^)]+)\)")
EOG_HEADERS = {
    "TP9": "eog_TP9_upper_vertical",
    "TP10": "eog_TP10_lower_vertical",
    "FT9": "eog_FT9_left_horizontal",
    "FT10": "eog_FT10_right_horizontal",
}


def split_tab_fields(line: str) -> list[str]:
    fields = [field.strip() for field in line.rstrip("\n").split("\t")]
    if fields and fields[-1] == "":
        fields.pop()
    return fields


def parse_channel_labels(impedance_path: Path) -> list[str]:
    labels: list[str] = []
    found_table = False

    with impedance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Phys. Chn."):
                found_table = True
                continue
            if not found_table or not line.strip():
                continue

            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if not parts or not parts[0].startswith("#"):
                continue

            label = parts[1]
            if label not in {"Ref", "Gnd"}:
                labels.append(label)

    if len(labels) != 64:
        raise ValueError(f"Expected 64 channel labels in {impedance_path}, found {len(labels)}")

    return labels


def parse_joint_labels(joints_path: Path) -> list[str]:
    with joints_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    match = JOINT_LABEL_PATTERN.search(header)
    if match is None:
        raise ValueError(f"Could not parse joint labels from {joints_path}")

    labels = match.group(1).split()
    if len(labels) != 12:
        raise ValueError(f"Expected 12 joint labels in {joints_path}, found {len(labels)}")

    return labels


def build_header(channel_labels: list[str], joint_labels: list[str]) -> list[str]:
    header = ["subject_id", "trial_id", "time_seconds"]

    for label in channel_labels:
        header.append(EOG_HEADERS.get(label, f"eeg_{label}"))

    for label in joint_labels:
        header.append(f"joint_{label}")

    return header


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")

    trial_dirs = sorted(
        path for path in INPUT_ROOT.iterdir() if path.is_dir() and TRIAL_DIR_PATTERN.fullmatch(path.name)
    )
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found in {INPUT_ROOT}")

    canonical_channels = parse_channel_labels(trial_dirs[0] / "impedances-before.txt")
    canonical_joints = parse_joint_labels(trial_dirs[0] / "joints.txt")

    total_rows = 0
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(build_header(canonical_channels, canonical_joints))

        for trial_dir in trial_dirs:
            match = TRIAL_DIR_PATTERN.fullmatch(trial_dir.name)
            if match is None:
                raise ValueError(f"Unexpected directory name: {trial_dir.name}")

            subject_id, trial_id = match.groups()
            print(f"Processing {trial_dir.name}...")

            channel_labels = parse_channel_labels(trial_dir / "impedances-before.txt")
            joint_labels = parse_joint_labels(trial_dir / "joints.txt")
            if channel_labels != canonical_channels:
                raise ValueError(f"Channel labels do not match canonical order in {trial_dir}")
            if joint_labels != canonical_joints:
                raise ValueError(f"Joint labels do not match canonical order in {trial_dir}")

            eeg_path = trial_dir / "eeg.txt"
            joints_path = trial_dir / "joints.txt"

            with eeg_path.open("r", encoding="utf-8") as eeg_file, joints_path.open(
                "r", encoding="utf-8"
            ) as joints_file:
                eeg_header = eeg_file.readline().strip()
                if eeg_header != "64 channels":
                    raise ValueError(f"Unexpected EEG header in {eeg_path}: {eeg_header}")

                joints_file.readline()
                joints_file.readline()

                for row_number, (eeg_line, joints_line) in enumerate(
                    zip_longest(eeg_file, joints_file), start=1
                ):
                    if eeg_line is None or joints_line is None:
                        raise ValueError(f"Row count mismatch between {eeg_path} and {joints_path}")

                    eeg_fields = split_tab_fields(eeg_line)
                    joint_fields = split_tab_fields(joints_line)

                    if len(eeg_fields) != 65:
                        raise ValueError(
                            f"Expected 65 values in {eeg_path} at row {row_number}, found {len(eeg_fields)}"
                        )
                    if len(joint_fields) != 13:
                        raise ValueError(
                            f"Expected 13 values in {joints_path} at row {row_number}, found {len(joint_fields)}"
                        )
                    if eeg_fields[0] != joint_fields[0]:
                        raise ValueError(
                            f"Timestamp mismatch in {trial_dir} at row {row_number}: "
                            f"{eeg_fields[0]} != {joint_fields[0]}"
                        )

                    writer.writerow(
                        [subject_id, trial_id, eeg_fields[0], *eeg_fields[1:], *joint_fields[1:]]
                    )
                    total_rows += 1

    print(f"Wrote {total_rows} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
