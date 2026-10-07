from __future__ import annotations

import difflib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .installer import InstallError
from .parser import RESERVED_IDENTIFIERS


INTERNAL_PARAMETERS = {"method", "COUNT", "signal", "symmetry"}
RANGE_KINDS = ("Fixed", "Linear", "Step", "Random", "Normal")


@dataclass(slots=True)
class RenameAttribute:
    role: str
    original: str
    edited: str


@dataclass(slots=True)
class ConfigAttribute:
    role: str
    source_name: str
    config_name: str
    range_kind: str
    minimum: str
    maximum: str
    step: str
    count: str
    mean: str
    deviation: str
    line_index: int


@dataclass(slots=True)
class EditableSystem:
    root: Path
    original_id: str
    system_id: str
    display_name: str
    variables: list[RenameAttribute]
    parameters: list[RenameAttribute]
    config_attributes: list[ConfigAttribute]
    methods: list[str]
    default_method: str
    method_line_index: int
    cu_text: str
    h_text: str
    txt_lines: list[str]


@dataclass(slots=True)
class SystemEditPlan:
    root: Path
    original_id: str
    system_id: str
    directory_files: dict[Path, str | bytes]
    registration_files: dict[Path, str]
    diff: str


def _enum_names(source: str, enum_name: str) -> list[str]:
    match = re.search(rf"enum\s+{re.escape(enum_name)}\s*\{{([^}}]*)\}}", source, re.DOTALL)
    if not match:
        raise InstallError(f"Could not find enum {enum_name!r} in the CUDA source")
    body = re.sub(r"//[^\r\n]*", "", match.group(1))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return [item.strip() for item in body.split(",") if item.strip()]


def load_editable_system(project_root: str | Path, system_id: str) -> EditableSystem:
    root = Path(project_root).resolve()
    directory = root / "systems" / system_id
    cu = directory / f"{system_id}.cu"
    header = directory / f"{system_id}.h"
    description = directory / f"{system_id}.txt"
    missing = [path.name for path in (cu, header, description) if not path.is_file()]
    if missing:
        raise InstallError("System cannot be edited; required files are missing: " + ", ".join(missing))
    cu_text = cu.read_text(encoding="utf-8-sig")
    h_text = header.read_text(encoding="utf-8-sig")
    txt_lines = description.read_text(encoding="utf-8-sig").splitlines()
    variable_names = _enum_names(cu_text, "variables")
    parameter_names = [name for name in _enum_names(cu_text, "parameters") if name not in INTERNAL_PARAMETERS]
    display_name = system_id
    for line in txt_lines:
        if line.startswith("Name:"):
            display_name = line.partition(":")[2].strip() or system_id
            break

    config_attributes: list[ConfigAttribute] = []
    methods: list[str] = []
    default_method = ""
    method_line_index = -1
    role_indices = {"var": 0, "param": 0}
    source_by_role = {"var": variable_names, "param": parameter_names}
    for line_index, line in enumerate(txt_lines):
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["enum", "method"]:
            method_line_index = line_index
            for token in parts[2:]:
                if len(token) >= 2 and token[0] in "01":
                    methods.append(token[1:])
                    if token[0] == "1" and not default_method:
                        default_method = token[1:]
            continue
        if len(parts) < 9 or parts[0] not in source_by_role or parts[2] not in RANGE_KINDS:
            continue
        role = parts[0]
        source_names = source_by_role[role]
        index = role_indices[role]
        source_name = source_names[index] if index < len(source_names) else parts[1]
        role_indices[role] += 1
        config_attributes.append(ConfigAttribute(
            role, source_name, parts[1], parts[2], parts[3], parts[4], parts[5],
            parts[6], parts[7], parts[8], line_index,
        ))
    if methods and not default_method:
        default_method = methods[0]
    return EditableSystem(
        root, system_id, system_id, display_name,
        [RenameAttribute("var", name, name) for name in variable_names],
        [RenameAttribute("param", name, name) for name in parameter_names],
        config_attributes, methods, default_method, method_line_index,
        cu_text, h_text, txt_lines,
    )


def validate_renames(model: EditableSystem) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", model.system_id):
        problems.append("System ID must be a valid C/C++ identifier")
    elif model.system_id in RESERVED_IDENTIFIERS and model.system_id != model.original_id:
        problems.append(f"System ID {model.system_id!r} is reserved by CUDAynamics")
    attributes = [*model.variables, *model.parameters]
    edited = [item.edited for item in attributes]
    for item in attributes:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.edited):
            problems.append(f"Attribute {item.edited!r} is not a valid C/C++ identifier")
        if item.edited in RESERVED_IDENTIFIERS and item.edited != item.original:
            problems.append(f"Attribute name {item.edited!r} is reserved by CUDAynamics")
    duplicates = sorted({name for name in edited if edited.count(name) > 1})
    if duplicates:
        problems.append("Duplicate attribute names: " + ", ".join(duplicates))
    return problems


def make_edit_identity_unique(model: EditableSystem) -> list[str]:
    systems = model.root / "systems"
    existing_ids = {
        path.name.casefold() for path in systems.iterdir()
        if path.is_dir() and path.name.casefold() != model.original_id.casefold()
    }
    existing_names: set[str] = set()
    for path in systems.glob("*/*.txt"):
        if path.parent.name.casefold() == model.original_id.casefold():
            continue
        try:
            first = next((line for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.startswith("Name:")), "")
        except OSError:
            continue
        if first:
            existing_names.add(first.partition(":")[2].strip().casefold())
    messages: list[str] = []
    base_id = model.system_id
    if base_id.casefold() in existing_ids:
        number = 2
        while f"{base_id}{number}".casefold() in existing_ids:
            number += 1
        model.system_id = f"{base_id}{number}"
        messages.append(f"ID {base_id!r} already exists and was changed to {model.system_id!r}.")
    base_name = model.display_name
    if base_name.casefold() in existing_names:
        number = 2
        while f"{base_name} {number}".casefold() in existing_names:
            number += 1
        model.display_name = f"{base_name} {number}"
        messages.append(f"Name {base_name!r} already exists and was changed to {model.display_name!r}.")
    return messages


def _rewrite_enum(source: str, enum_name: str, mapping: dict[str, str]) -> str:
    pattern = rf"(enum\s+{re.escape(enum_name)}\s*\{{)([^}}]*)(\}})"
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return source
    body = match.group(2)
    body = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda token: mapping.get(token.group(0), token.group(0)),
        body,
    )
    return source[:match.start(2)] + body + source[match.end(2):]


def _rewrite_cuda(source: str, variable_mapping: dict[str, str], parameter_mapping: dict[str, str]) -> str:
    source = _rewrite_enum(source, "variables", variable_mapping)
    source = _rewrite_enum(source, "parameters", parameter_mapping)
    source = re.sub(
        r"\b(V|Vnext)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        lambda match: f"{match.group(1)}({variable_mapping.get(match.group(2), match.group(2))})",
        source,
    )
    source = re.sub(
        r"attributes::variables::([A-Za-z_][A-Za-z0-9_]*)",
        lambda match: "attributes::variables::" + variable_mapping.get(match.group(1), match.group(1)),
        source,
    )
    source = re.sub(
        r"\bP\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        lambda match: f"P({parameter_mapping.get(match.group(1), match.group(1))})",
        source,
    )
    source = re.sub(
        r"attributes::parameters::([A-Za-z_][A-Za-z0-9_]*)",
        lambda match: "attributes::parameters::" + parameter_mapping.get(match.group(1), match.group(1)),
        source,
    )
    return source


def _replace_system_id(source: str, old: str, new: str, *, header_include: bool = False) -> str:
    source = re.sub(rf"(#define\s+name\s+){re.escape(old)}\b", rf"\g<1>{new}", source)
    if header_include:
        source = source.replace(f'#include "{old}.h"', f'#include "{new}.h"')
    return source


def build_system_edit_plan(model: EditableSystem) -> SystemEditPlan:
    problems = validate_renames(model)
    if problems:
        raise InstallError("\n".join(problems))
    root = model.root
    old_id, new_id = model.original_id, model.system_id
    old_dir, new_dir = root / "systems" / old_id, root / "systems" / new_id
    if old_id.casefold() != new_id.casefold() and new_dir.exists():
        raise InstallError(f"Target system directory already exists: {new_dir}")
    variable_mapping = {item.original: item.edited for item in model.variables if item.original != item.edited}
    parameter_mapping = {item.original: item.edited for item in model.parameters if item.original != item.edited}
    cu_text = _replace_system_id(
        _rewrite_cuda(model.cu_text, variable_mapping, parameter_mapping), old_id, new_id,
        header_include=True,
    )
    h_text = _replace_system_id(model.h_text, old_id, new_id)

    txt_lines = list(model.txt_lines)
    for index, line in enumerate(txt_lines):
        if line.startswith("Name:"):
            txt_lines[index] = f"Name: {model.display_name}"
            break
    effective_names = {**variable_mapping, **parameter_mapping}
    for attribute in model.config_attributes:
        config_name = effective_names.get(attribute.source_name, attribute.config_name)
        txt_lines[attribute.line_index] = " ".join((
            attribute.role, config_name, attribute.range_kind, attribute.minimum,
            attribute.maximum, attribute.step, attribute.count, attribute.mean,
            attribute.deviation,
        ))
    if model.methods:
        if model.default_method not in model.methods:
            raise InstallError("Default method must be one of the methods in enum method")
        txt_lines[model.method_line_index] = "enum method " + " ".join(
            ("1" if method == model.default_method else "0") + method
            for method in model.methods
        )
    txt_text = "\n".join(txt_lines) + "\n"

    directory_files: dict[Path, str | bytes] = {}
    for source_path in old_dir.iterdir():
        if source_path.name == f"{old_id}.cu":
            directory_files[Path(f"{new_id}.cu")] = cu_text
        elif source_path.name == f"{old_id}.h":
            directory_files[Path(f"{new_id}.h")] = h_text
        elif source_path.name == f"{old_id}.txt":
            directory_files[Path(f"{new_id}.txt")] = txt_text
        elif source_path.is_file():
            name = f"{new_id}{source_path.suffix.lower()}" if source_path.stem == old_id else source_path.name
            directory_files[Path(name)] = source_path.read_bytes()

    registration_files: dict[Path, str] = {}
    replacements = {
        root / "systemsHeaders.h": [(f"systems/{old_id}/{old_id}.h", f"systems/{new_id}/{new_id}.h")],
        root / "main.cpp": [(f"addKernel({old_id});", f"addKernel({new_id});")],
        root / "CUDAynamics.vcxproj": [(f"systems\\{old_id}\\{old_id}.cu", f"systems\\{new_id}\\{new_id}.cu")],
        root / "CUDAynamics.vcxproj.filters": [(f"systems\\{old_id}\\{old_id}.cu", f"systems\\{new_id}\\{new_id}.cu")],
    }
    diffs: list[str] = []
    for path, pairs in replacements.items():
        if not path.is_file():
            raise InstallError(f"Required registration file is missing: {path.name}")
        original = path.read_text(encoding="utf-8-sig")
        updated = original
        for old, new in pairs:
            updated = updated.replace(old, new)
        if old_id != new_id and updated == original:
            raise InstallError(
                f"Could not find registration for system {old_id!r} in {path.name}"
            )
        registration_files[path] = updated
        if updated != original:
            diffs.extend(difflib.unified_diff(original.splitlines(), updated.splitlines(), fromfile=path.name, tofile=path.name, lineterm=""))
    diffs.append(f"Edit system directory: systems/{old_id} -> systems/{new_id}")
    return SystemEditPlan(root, old_id, new_id, directory_files, registration_files, "\n".join(diffs))


def apply_system_edit_plan(plan: SystemEditPlan) -> Path:
    root = plan.root
    old_dir = root / "systems" / plan.original_id
    new_dir = root / "systems" / plan.system_id
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = root / ".cudynamics-editor-backups" / f"edit-{stamp}"
    staging = root / "systems" / f".{plan.system_id}.editor-tmp-{stamp}"
    moved_old = False
    installed_new = False
    try:
        staging.mkdir(parents=False)
        for relative, content in plan.directory_files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            else:
                destination.write_text(content, encoding="utf-8", newline="\n")
        for path in plan.registration_files:
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        saved_dir = backup / "systems" / plan.original_id
        saved_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(saved_dir))
        moved_old = True
        shutil.move(str(staging), str(new_dir))
        installed_new = True
        for path, content in plan.registration_files.items():
            temporary = path.with_suffix(path.suffix + ".editor-tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
    except Exception as exc:
        if installed_new and new_dir.exists():
            shutil.rmtree(new_dir)
        if staging.exists():
            shutil.rmtree(staging)
        saved_dir = backup / "systems" / plan.original_id
        if moved_old and saved_dir.exists() and not old_dir.exists():
            shutil.move(str(saved_dir), str(old_dir))
        for path in plan.registration_files:
            saved = backup / path.relative_to(root)
            if saved.exists():
                shutil.copy2(saved, path)
        raise InstallError(f"Editing failed and changes were rolled back: {exc}") from exc
    return backup
