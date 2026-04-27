"""
Genera un snapshot compacto del estado del proyecto para usar con Claude.

Uso:
    python scripts/context_snapshot.py

Salida:
    docs/context_snapshots/snapshot_YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "context_snapshots"

ARCHIVOS_CLAVE = [
    "README.md",
    "CLAUDE.md",
    "CLAUDE_CONTEXT.md",
    "docs/DESIGN_DECISIONS.md",
    "docs/THEORY_TO_CODE.md",
    "docs/RUN_GUIDE.md",
    "configs/pipeline.yaml",
]


def ejecutar(comando: list[str]) -> str:
    try:
        resultado = subprocess.run(
            comando,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return resultado.stdout.strip() or resultado.stderr.strip()
    except Exception as exc:
        return f"No se pudo ejecutar {' '.join(comando)}: {exc}"


def leer_archivo(rel_path: str, max_chars: int = 6000) -> str:
    path = ROOT / rel_path
    if not path.exists():
        return f"[NO EXISTE] {rel_path}"

    texto = path.read_text(encoding="utf-8", errors="replace")
    if len(texto) > max_chars:
        return texto[:max_chars] + "\n\n[TRUNCADO]"
    return texto


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = OUT_DIR / f"snapshot_{timestamp}.md"

    git_status = ejecutar(["git", "status", "--short"])
    git_branch = ejecutar(["git", "branch", "--show-current"])
    git_commit = ejecutar(["git", "rev-parse", "--short", "HEAD"])

    partes = [
        "# Snapshot de contexto para Claude",
        "",
        f"- Fecha: {datetime.now().isoformat(timespec='seconds')}",
        f"- Rama git: {git_branch or 'desconocida'}",
        f"- Commit: {git_commit or 'desconocido'}",
        "",
        "## Estado git",
        "```txt",
        git_status or "Sin cambios detectados.",
        "```",
        "",
        "## Estructura principal",
        "```txt",
        ejecutar(["git", "ls-files"]),
        "```",
        "",
    ]

    for archivo in ARCHIVOS_CLAVE:
        partes.extend(
            [
                f"## {archivo}",
                "```txt",
                leer_archivo(archivo),
                "```",
                "",
            ]
        )

    salida.write_text("\n".join(partes), encoding="utf-8")
    print(f"Snapshot generado: {salida}")


if __name__ == "__main__":
    main()