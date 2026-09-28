import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_FILENAME = "summary.txt"


def _is_python_file(path: pathlib.Path) -> bool:
    return path.suffix == ".py" and "__pycache__" not in path.parts


def _has_main_signature(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return ('if __name__ == "__main__"' in text) or ("def main(" in text)


def _iter_project_python_files():
    for path in ROOT.rglob("*.py"):
        if _is_python_file(path):
            yield path


def _collect_files():
    src_root = ROOT / "src"
    src_files = sorted(
        [p for p in src_root.rglob("*.py") if _is_python_file(p)]
    ) if src_root.exists() else []

    main_files = sorted(
        [p for p in _iter_project_python_files() if _has_main_signature(p)]
    )

    selected = []
    seen = set()
    for path in src_files + main_files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        selected.append(path)
    return src_files, main_files, selected


def generate_summary():
    src_files, main_files, selected_files = _collect_files()
    out_path = ROOT / OUTPUT_FILENAME

    with out_path.open("w", encoding="utf-8") as outfile:
        outfile.write("Python Source Summary\n")
        outfile.write("=====================\n\n")
        outfile.write(f"Project root: {ROOT}\n")
        outfile.write(f"Included src files: {len(src_files)}\n")
        outfile.write(f"Detected main/entry files: {len(main_files)}\n")
        outfile.write(f"Total unique files included: {len(selected_files)}\n\n")

        for file_path in selected_files:
            rel_path = file_path.relative_to(ROOT)
            outfile.write(f"{'=' * 80}\n")
            outfile.write(f"FILE: {rel_path}\n")
            outfile.write(f"{'=' * 80}\n\n")
            try:
                outfile.write(file_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                outfile.write(f"ERROR READING FILE: {exc}\n")
            outfile.write("\n\n")

    print(f"Success! Wrote {len(selected_files)} files to '{out_path}'.")


if __name__ == "__main__":
    generate_summary()
