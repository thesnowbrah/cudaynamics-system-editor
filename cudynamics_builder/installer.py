from __future__ import annotations

import difflib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .generator import GeneratedProject
from .model import SystemSpec


class InstallError(RuntimeError):
    pass


@dataclass(slots=True)
class InstallPlan:
    writes: dict[Path, str | bytes]
    diff: str


@dataclass(slots=True, frozen=True)
class LibrarySystem:
    system_id: str
    display_name: str
    directory: Path


@dataclass(slots=True)
class RemovalPlan:
    updated_files: dict[Path, str]
    directories: list[Path]
    system_ids: list[str]
    diff: str


def locate_default_project(builder_root: Path) -> Path:
    candidate = builder_root.parent / "cudaynamics-master"
    return candidate if candidate.is_dir() else builder_root.parent


def ensure_unique_system_identity(
    project_root: str | Path, spec: SystemSpec, language: str = "en"
) -> list[str]:
    """Make the generated folder ID and visible system name unique in a project."""
    systems = Path(project_root).resolve() / "systems"
    if not systems.is_dir():
        return []  # build_plan will produce the detailed invalid-project error.

    existing_ids = {path.name.casefold() for path in systems.iterdir() if path.is_dir()}
    existing_names: set[str] = set()
    for description in systems.glob("*/*.txt"):
        try:
            first_line = description.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            continue
        if first_line.startswith("Name:"):
            existing_names.add(first_line.partition(":")[2].strip().casefold())

    messages: list[str] = []
    original_id = spec.system_id
    if original_id.casefold() in existing_ids:
        suffix = 2
        candidate = f"{original_id}{suffix}"
        while candidate.casefold() in existing_ids:
            suffix += 1
            candidate = f"{original_id}{suffix}"
        spec.system_id = candidate
        messages.append(
            f"ID {original_id!r} уже существует и заменён на {candidate!r}."
            if language == "ru" else
            f"ID {original_id!r} already exists and was changed to {candidate!r}."
        )

    original_name = spec.display_name
    if original_name.casefold() in existing_names:
        suffix = 2
        candidate = f"{original_name} {suffix}"
        while candidate.casefold() in existing_names:
            suffix += 1
            candidate = f"{original_name} {suffix}"
        spec.display_name = candidate
        messages.append(
            f"Название {original_name!r} уже существует и заменено на {candidate!r}."
            if language == "ru" else
            f"Name {original_name!r} already exists and was changed to {candidate!r}."
        )
    return messages


def list_library_systems(project_root: str | Path) -> list[LibrarySystem]:
    root = Path(project_root).resolve()
    systems = root / "systems"
    if not systems.is_dir():
        raise InstallError("Selected folder is not a CUDAynamics project: systems directory is missing")
    result: list[LibrarySystem] = []
    for directory in sorted((path for path in systems.iterdir() if path.is_dir()), key=lambda p: p.name.casefold()):
        description = directory / f"{directory.name}.txt"
        display_name = directory.name
        if description.is_file():
            try:
                for line in description.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    if line.startswith("Name:"):
                        display_name = line.partition(":")[2].strip() or directory.name
                        break
            except OSError:
                pass
        result.append(LibrarySystem(directory.name, display_name, directory))
    return result


def _remove_registration(text: str, system_id: str, kind: str) -> str:
    escaped = re.escape(system_id)
    if kind == "headers":
        pattern = rf'^\s*#include\s+"systems/{escaped}/{escaped}\.h"\s*\r?\n?'
    elif kind == "main":
        pattern = rf'^\s*addKernel\(\s*{escaped}\s*\);\s*\r?\n?'
    elif kind == "vcx":
        pattern = rf'^\s*<CudaCompile Include="systems\\{escaped}\\{escaped}\.cu"\s*/>\s*\r?\n?'
    elif kind == "filters":
        pattern = rf'^\s*<CudaCompile Include="systems\\{escaped}\\{escaped}\.cu"(?:\s*/>|>.*?</CudaCompile>)\s*\r?\n?'
    else:
        raise ValueError(kind)
    return re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL if kind == "filters" else re.MULTILINE)


def build_removal_plan(project_root: str | Path, system_ids: list[str]) -> RemovalPlan:
    root = Path(project_root).resolve()
    required = {
        "systemsHeaders.h": "headers",
        "main.cpp": "main",
        "CUDAynamics.vcxproj": "vcx",
        "CUDAynamics.vcxproj.filters": "filters",
    }
    missing = [name for name in (*required, "systems") if not (root / name).exists()]
    if missing:
        raise InstallError("Selected folder is not a CUDAynamics project. Missing: " + ", ".join(missing))
    unique_ids = list(dict.fromkeys(system_ids))
    if not unique_ids:
        raise InstallError("Select at least one system to delete")
    directories = [root / "systems" / sid for sid in unique_ids]
    absent = [path.name for path in directories if not path.is_dir()]
    if absent:
        raise InstallError("System directories were not found: " + ", ".join(absent))

    updated_files: dict[Path, str] = {}
    diffs: list[str] = []
    for name, kind in required.items():
        path = root / name
        original = path.read_text(encoding="utf-8-sig")
        updated = original
        for sid in unique_ids:
            updated = _remove_registration(updated, sid, kind)
        updated_files[path] = updated
        diffs.extend(difflib.unified_diff(
            original.splitlines(), updated.splitlines(), fromfile=name, tofile=name, lineterm=""
        ))
    for directory in directories:
        diffs.append(f"Delete directory (backed up): systems/{directory.name}")
    return RemovalPlan(updated_files, directories, unique_ids, "\n".join(diffs))


def apply_removal_plan(project_root: str | Path, plan: RemovalPlan) -> Path:
    root = Path(project_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = root / ".cudynamics-editor-backups" / f"removal-{stamp}"
    moved: list[tuple[Path, Path]] = []
    try:
        for path in plan.updated_files:
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for directory in plan.directories:
            destination = backup / directory.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(destination))
            moved.append((directory, destination))
        for path, content in plan.updated_files.items():
            temporary = path.with_suffix(path.suffix + ".editor-tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
    except Exception as exc:
        for original, saved in reversed(moved):
            if saved.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(saved), str(original))
        for path in plan.updated_files:
            saved = backup / path.relative_to(root)
            if saved.exists():
                shutil.copy2(saved, path)
        raise InstallError(f"Removal failed and changes were rolled back: {exc}") from exc
    return backup


def _insert_before_group_end(text: str, marker: str, addition: str) -> str:
    position = text.find(marker)
    if position < 0:
        raise InstallError(f"В файле проекта не найден маркер {marker!r}")
    end = text.find("</ItemGroup>", position)
    if end < 0:
        raise InstallError("Повреждён XML проекта: нет </ItemGroup>")
    return text[:end] + addition + text[end:]


def build_plan(
    project_root: str | Path,
    spec: SystemSpec,
    generated: GeneratedProject,
    source_document: str | Path | None = None,
) -> InstallPlan:
    root = Path(project_root).resolve()
    required = ["systemsHeaders.h", "main.cpp", "CUDAynamics.vcxproj", "CUDAynamics.vcxproj.filters", "systems"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise InstallError("Это не корень CUDAynamics. Не найдены: " + ", ".join(missing))
    target_dir = root / "systems" / spec.system_id
    if target_dir.exists():
        raise InstallError(f"Система {spec.system_id!r} уже существует")

    originals = {name: (root / name).read_text(encoding="utf-8-sig") for name in required[:-1]}
    sid = spec.system_id
    updated: dict[str, str] = {}

    headers = originals["systemsHeaders.h"]
    include = f'#include "systems/{sid}/{sid}.h"'
    if include in headers:
        raise InstallError("Заголовок системы уже зарегистрирован")
    updated["systemsHeaders.h"] = headers.rstrip() + f"\n{include}\n"

    main = originals["main.cpp"]
    anchor = "    //selectKernel(kernels.begin()->first);"
    if anchor not in main:
        anchor = "    selectKernel(lorenz);"
    if anchor not in main:
        raise InstallError("Не найдено место регистрации addKernel в main.cpp")
    updated["main.cpp"] = main.replace(anchor, f"    addKernel({sid});\n{anchor}", 1)

    vcx = originals["CUDAynamics.vcxproj"]
    updated["CUDAynamics.vcxproj"] = _insert_before_group_end(
        vcx, '<CudaCompile Include="systems\\', f'    <CudaCompile Include="systems\\{sid}\\{sid}.cu" />\n  '
    )
    filters = originals["CUDAynamics.vcxproj.filters"]
    updated["CUDAynamics.vcxproj.filters"] = _insert_before_group_end(
        filters, '<CudaCompile Include="systems\\', f'    <CudaCompile Include="systems\\{sid}\\{sid}.cu" />\n  '
    )

    writes: dict[Path, str | bytes] = {root / name: content for name, content in updated.items()}
    for relative, content in generated.files.items():
        writes[root / Path(relative)] = content

    source_relative: Path | None = None
    if source_document:
        source_path = Path(source_document)
        if not source_path.is_file():
            raise InstallError(f"Исходный файл не найден: {source_path}")
        source_relative = Path("systems") / sid / f"{sid}{source_path.suffix.lower()}"
        writes[root / source_relative] = source_path.read_bytes()

    diffs = []
    for name, new in updated.items():
        diffs.extend(difflib.unified_diff(originals[name].splitlines(), new.splitlines(), fromfile=name, tofile=name, lineterm=""))
    for relative, content in generated.files.items():
        diffs.extend(difflib.unified_diff([], content.splitlines(), fromfile="/dev/null", tofile=relative, lineterm=""))
    if source_relative is not None:
        diffs.append(f"Binary source file: {source_relative.as_posix()}")
    return InstallPlan(writes, "\n".join(diffs))


def apply_plan(project_root: str | Path, plan: InstallPlan) -> Path:
    root = Path(project_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = root / ".cudynamics-builder-backups" / stamp
    changed_existing = [path for path in plan.writes if path.exists()]
    created = [path for path in plan.writes if not path.exists()]
    try:
        for path in changed_existing:
            relative = path.relative_to(root)
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for path, content in plan.writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".builder-tmp")
            if isinstance(content, bytes):
                temporary.write_bytes(content)
            else:
                temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
    except Exception as exc:
        for path in created:
            if path.exists():
                path.unlink()
        for path in changed_existing:
            saved = backup / path.relative_to(root)
            if saved.exists():
                shutil.copy2(saved, path)
        raise InstallError(f"Установка отменена и изменения откачены: {exc}") from exc
    return backup
