"""
Timestamped run directories, and the README.md that documents each one.

Every diagnostic run gets its own folder rather than overwriting a shared
output file. The motivation is that results on this project have repeatedly
been invalidated by a change somewhere else -- the Nelder-Mead simplex fix, the
Euler-Maruyama discretization bug, the KDE round-trip removal -- and a bare
loss_comparison.png on disk carries no record of which code produced it. A run
folder carries its config, its dataset, its git commit and its results together,
so an old result stays interpretable instead of merely stale.

Folder name: dd-mm-y-hh-mm-<n_genes>, where y is the last digit of the year.
    21-09-6-15-42-8   = 21 September 2026, 15:42, 8 genes
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def run_stamp(n_genes: int, when: datetime | None = None) -> str:
    """dd-mm-y-hh-mm-<n_genes>, y being the year's last digit."""
    now = when or datetime.now()
    return f"{now:%d-%m}-{now.year % 10}-{now:%H-%M}-{n_genes}"


def make_run_dir(base: Path, n_genes: int, *, when: datetime | None = None) -> Path:
    """Create base/<stamp>/ and return it. Never overwrites an existing run."""
    d = Path(base) / run_stamp(n_genes, when)
    if d.exists():                      # same minute, same gene count
        for suffix in "bcdefghij":
            alt = d.with_name(d.name + suffix)
            if not alt.exists():
                d = alt
                break
    d.mkdir(parents=True)
    return d


def git_commit() -> str:
    """Short SHA + dirty marker, so a run records the code that produced it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


class Readme:
    """
    Accumulates markdown sections, written out once at the end.

    Sections are appended in call order; write() joins them. Kept deliberately
    dumb -- no templating engine, no escaping rules to remember.
    """

    def __init__(self, title: str):
        self._parts: list[str] = [f"# {title}\n"]

    def text(self, body: str) -> "Readme":
        self._parts.append(body.strip() + "\n")
        return self

    def section(self, heading: str, body: str = "") -> "Readme":
        self._parts.append(f"## {heading}\n")
        if body:
            self._parts.append(body.strip() + "\n")
        return self

    def kv(self, pairs: dict) -> "Readme":
        """
        Two-column table. Values are str()'d, so anything goes -- including
        pipes, which must be escaped or they read as cell separators and
        silently shred the table (stat names like "mean |edge|" do this).
        """
        def esc(v):
            return str(v).replace("|", "\\|")

        rows = "\n".join(f"| {esc(k)} | {esc(v)} |" for k, v in pairs.items())
        self._parts.append(f"| | |\n|---|---|\n{rows}\n")
        return self

    def table(self, df, *, floatfmt: str = "{:.4f}") -> "Readme":
        """
        A pandas DataFrame as a markdown table.

        Hand-rolled rather than df.to_markdown(), which needs `tabulate` --
        not worth a dependency for eight lines. Numbers are right-aligned so
        columns of metrics stay readable.
        """
        cols = list(df.columns)

        def cell(v):
            return floatfmt.format(v) if isinstance(v, float) else str(v)

        head = "| " + " | ".join(cols) + " |"
        rule = "|" + "|".join("---:" if df[c].dtype.kind in "fiu" else "---"
                              for c in cols) + "|"
        rows = ["| " + " | ".join(cell(v) for v in r) + " |"
                for r in df.itertuples(index=False, name=None)]
        self._parts.append("\n".join([head, rule, *rows]) + "\n")
        return self

    def image(self, path: str, alt: str = "") -> "Readme":
        """Relative path, so the README renders inside its own run folder."""
        self._parts.append(f"![{alt or path}]({path})\n")
        return self

    def code(self, body: str, lang: str = "") -> "Readme":
        self._parts.append(f"```{lang}\n{body.strip()}\n```\n")
        return self

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.write_text("\n".join(self._parts))
        return path
