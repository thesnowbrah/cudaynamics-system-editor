from __future__ import annotations

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from .model import AttributeDefaults, Equation, EquationKind, METHODS, SIGNAL_PARAMETERS, SystemSpec


class ParseError(ValueError):
    pass


_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)
_MATH_NAMES = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "atan2": sp.atan2, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log,
    "log10": lambda x: sp.log(x, 10), "log2": lambda x: sp.log(x, 2), "pow": sp.Pow,
    "abs": sp.Abs, "Abs": sp.Abs, "fabs": sp.Abs,
    "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling,
    "min": sp.Min, "Min": sp.Min, "max": sp.Max, "Max": sp.Max,
    "mod": sp.Mod, "fmod": sp.Mod, "sign": sp.sign, "sgn": sp.sign,
    "erf": sp.erf, "erfc": sp.erfc,
    "hypot": lambda x, y: sp.sqrt(x * x + y * y),
    "pi": sp.pi, "E": sp.E,
}

# Unscoped enum members generated inside namespace attributes share one C++
# namespace. ``h`` is additionally reserved by the metadata step parameter.
CUDA_MACRO_IDENTIFIERS = {
    "H", "name", "numb", "LOCAL_BUFFERS", "LOAD_PT_OMP", "LOAD_PT_CUDA",
    "mapPosition", "CUDA_marshal", "CUDA_kernel", "CUDA_ATTR_COUNT",
    "CUDA_SET_DEVICE", "CUDA_LASTERROR", "CUDA_SYNCHRONIZE", "CUDA_RESET",
    "FDS_ARGUMENTS", "RECORD_STEP", "NORMAL_STEP_IN_ANALYSIS_IF_HIRES",
    "LLE_STORE_TO_TEMP", "LLE_FILL", "LLE_MEAN_RESULT", "MAX_ENUMS",
    "MAX_ATTRIBUTES", "MAX_MAPS", "MAX_VARS_PARAMS", "USE_DOUBLE_PRECISION",
}
RESERVED_IDENTIFIERS = {
    "h", "method", "COUNT", "symmetry", "signal", "square", "sine", "triangle",
    "variables", "parameters", "methods", "waveforms",
    *METHODS.keys(),
    *CUDA_MACRO_IDENTIFIERS,
}


def sympify_expression(text: str) -> sp.Expr:
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    local_dict = {name: sp.Symbol(name) for name in identifiers if name not in _MATH_NAMES}
    local_dict.update(_MATH_NAMES)
    return parse_expr(text, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=False)


def _normalize_signal_calls(text: str) -> tuple[str, set[str]]:
    """Turn unknown one-argument functions such as F(t) into signal symbols."""
    names: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _MATH_NAMES:
            return match.group(0)
        names.add(safe_identifier(name))
        return safe_identifier(name)

    normalized = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*t\s*\)", replace, text)
    return normalized, names


def _strip_outer_groups(text: str) -> str:
    """Remove balanced {...} wrappers without touching groups inside an expression."""
    value = text.strip()
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        closes_at_end = False
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1].strip()
    return value


def _trim_unmatched_edge_groups(text: str) -> str:
    """Drop only surplus OCR braces at the outer edges of a fragment.

    pix2tex can split nested ``array`` wrappers across equation rows, leaving
    an opening brace on one row and the matching closing brace on another.
    Braces belonging to LaTeX commands remain balanced and are preserved.
    """
    value = text.strip()
    while value.startswith("{") and value.count("{") > value.count("}"):
        value = value[1:].lstrip()
    while value.endswith("}") and value.count("}") > value.count("{"):
        value = value[:-1].rstrip()
    return value


def _clean_latex_fragment(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)\b", "", text)
    text = text.replace("&", "")
    text = re.sub(r"\\(?:quad|qquad|,|;|:|!)", " ", text)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    return _strip_outer_groups(_trim_unmatched_edge_groups(text.strip()))


def _split_lines(text: str) -> list[str]:
    text = text.replace("\r", "\n").replace("−", "-")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    # System delimiters emitted by pix2tex are presentation, not equations.
    text = re.sub(r"\\left\s*\\?[\{\[\(]", "", text)
    text = re.sub(r"\\right\s*(?:\\?[\}\]\)]|\.)", "", text)
    environments = r"cases|aligned|alignedat|gathered|array|matrix|pmatrix|bmatrix|vmatrix"
    text = re.sub(rf"\\begin\s*\{{(?:{environments})\}}(?:\s*\{{[^{{}}]*\}})?", "", text)
    text = re.sub(rf"\\end\s*\{{(?:{environments})\}}", "", text)
    text = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", "\n", text)
    text = text.replace(";", "\n")
    lines = []
    for raw in text.splitlines():
        line = _clean_latex_fragment(raw).strip(" \t,$")
        line = _strip_outer_groups(line)
        if line:
            lines.append(line)
    return lines


def _latex_rhs_to_sympy(rhs: str) -> str:
    rhs = _clean_latex_fragment(rhs).replace("−", "-").replace("·", "*")
    if "\\" not in rhs and "{" not in rhs:
        return rhs
    try:
        from sympy.parsing.latex import parse_latex
        return str(parse_latex(rhs))
    except Exception:
        rhs = rhs.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
        rhs = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}", r"\1", rhs)
        for old, new in (("\\sin", "sin"), ("\\cos", "cos"), ("\\tan", "tan"),
                         ("\\arcsin", "asin"), ("\\arccos", "acos"), ("\\arctan", "atan"),
                         ("\\sinh", "sinh"), ("\\cosh", "cosh"), ("\\tanh", "tanh"),
                         ("\\sec", "sec"), ("\\csc", "csc"), ("\\cot", "cot"),
                         ("\\exp", "exp"), ("\\sqrt", "sqrt"), ("\\ln", "log"),
                         ("\\log", "log"), ("\\min", "min"), ("\\max", "max")):
            rhs = rhs.replace(old, new)
        rhs = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", rhs)
        return rhs.replace("{", "(").replace("}", ")")


def _parse_lhs(lhs: str) -> tuple[str, EquationKind]:
    raw = _clean_latex_fragment(lhs)
    raw = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{d\}", "d", raw)
    raw = re.sub(r"\\partial", "d", raw)
    raw = re.sub(r"\s+", "", raw)
    raw = _strip_outer_groups(raw)
    fraction = re.fullmatch(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", raw)
    if not fraction:
        # Accept decorative braces left around only the fraction by malformed
        # nested pix2tex arrays. No other non-brace text is ignored here.
        embedded = re.search(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", raw)
        if embedded and not (raw[:embedded.start()] + raw[embedded.end():]).strip("{} "):
            fraction = embedded
    if fraction:
        numerator = _strip_outer_groups(fraction.group(1)).replace("{", "").replace("}", "")
        denominator = _strip_outer_groups(fraction.group(2)).replace("{", "").replace("}", "")
        match = re.fullmatch(r"d([A-Za-z_][A-Za-z0-9_]*)", numerator)
        if match and denominator in {"dt", "dtime"}:
            return match.group(1), EquationKind.STATE
    patterns = (
        r"^d([A-Za-z_][A-Za-z0-9_]*)/dt$",
        r"^\\dot\{([A-Za-z_][A-Za-z0-9_]*)\}$",
        r"^\\dot([A-Za-z_][A-Za-z0-9_]*)$",
        r"^dot\(?([A-Za-z_][A-Za-z0-9_]*)\)?$",
        r"^([A-Za-z_][A-Za-z0-9_]*)['′]$",
    )
    for pattern in patterns:
        match = re.match(pattern, raw)
        if match:
            return match.group(1), EquationKind.STATE
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", raw)
    if match:
        return match.group(1), EquationKind.ALGEBRAIC
    raise ParseError(f"Не удалось распознать левую часть: {lhs!r}")


def safe_identifier(name: str) -> str:
    translit = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "σ": "sigma", "ρ": "rho", "μ": "mu", "ω": "omega"})
    name = re.sub(r"[^A-Za-z0-9_]", "_", name.translate(translit))
    if not name or name[0].isdigit():
        name = "v_" + name
    return name


def parse_system(text: str, system_id: str = "new_system", display_name: str = "New system") -> SystemSpec:
    equations: list[Equation] = []
    signal_names: set[str] = set()
    errors: list[str] = []
    for number, line in enumerate(_split_lines(text), 1):
        line = re.sub(r"^\s*[-•]\s*", "", line)
        if "=" not in line:
            errors.append(f"строка {number}: нет знака '='")
            continue
        lhs, rhs = line.split("=", 1)
        try:
            target, kind = _parse_lhs(lhs)
            expression_text, found_signals = _normalize_signal_calls(_latex_rhs_to_sympy(rhs))
            signal_names |= found_signals
            expression = str(sympify_expression(expression_text))
            equations.append(Equation(safe_identifier(target), expression, kind, line))
        except Exception as exc:
            errors.append(f"строка {number}: {exc}")
    if errors:
        raise ParseError("\n".join(errors))
    if not equations:
        raise ParseError("Система не содержит уравнений")
    targets = [e.target for e in equations]
    duplicates = sorted({x for x in targets if targets.count(x) > 1})
    if duplicates:
        raise ParseError("Повторно заданы переменные: " + ", ".join(duplicates))

    for signal_name in sorted(signal_names - set(targets)):
        equations.append(Equation(signal_name, "waveform(t)", EquationKind.SIGNAL, f"{signal_name}(t) (распознано как сигнал)"))
    targets = [e.target for e in equations]
    symbols = {
        str(s) for equation in equations if equation.kind != EquationKind.SIGNAL
        for s in sympify_expression(equation.expression).free_symbols
    }
    parameters = sorted(symbols - set(targets) - {"t"})
    if "t" in symbols and "t" not in targets:
        equations.append(Equation("t", "1", EquationKind.STATE, "dt/dt = 1 (добавлено автоматически)"))
    spec = SystemSpec(safe_identifier(system_id.lower()), display_name, equations, parameters)
    normalize_reserved_identifiers(spec)
    return spec


def algebraic_order(spec: SystemSpec) -> list[Equation]:
    remaining = {e.target: e for e in spec.equations if e.kind == EquationKind.ALGEBRAIC}
    available = set(spec.states) | set(spec.signals) | set(spec.markers) | set(spec.parameters)
    ordered: list[Equation] = []
    while remaining:
        ready = [name for name, eq in remaining.items() if {str(s) for s in sympify_expression(eq.expression).free_symbols} <= available]
        if not ready:
            raise ParseError("Циклическая зависимость алгебраических переменных: " + ", ".join(remaining))
        for name in ready:
            ordered.append(remaining.pop(name))
            available.add(name)
    return ordered


def validate_spec(spec: SystemSpec) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.system_id):
        problems.append("ID системы должен быть корректным идентификатором C/C++")
    if not spec.states:
        problems.append("Нужно хотя бы одно дифференциальное уравнение")
    targets = [equation.target for equation in spec.equations]
    duplicate_targets = sorted({name for name in targets if targets.count(name) > 1})
    if duplicate_targets:
        problems.append("Повторно заданы переменные: " + ", ".join(duplicate_targets))
    if len(spec.signals) > 1:
        problems.append("Текущий формат CUDAynamics поддерживает не более одного входного сигнала")
    if spec.signals and "t" not in spec.states:
        problems.append("Для входного сигнала требуется состояние времени t (dt/dt = 1)")
    if not spec.methods:
        problems.append("Выберите хотя бы один метод интегрирования")
    unsupported_methods = set(spec.methods) - set(METHODS)
    if unsupported_methods:
        problems.append("Неподдерживаемые методы: " + ", ".join(sorted(unsupported_methods)))
    if spec.default_method is not None and spec.default_method not in spec.methods:
        problems.append("Метод по умолчанию должен входить в список создаваемых методов")
    generated_parameter_count = len(spec.parameters) + 2  # method, COUNT
    generated_parameter_count += int("VariableSymmetryCD" in spec.methods)
    generated_parameter_count += int(bool(spec.signals))  # waveform enum selector
    if len(spec.variables) > 128 or generated_parameter_count > 128:
        problems.append("CUDAynamics допускает не более 128 переменных и 128 параметров (с учётом h, method и symmetry)")
    reserved_parameters = set(spec.parameters) & RESERVED_IDENTIFIERS
    if spec.signals:
        reserved_parameters -= set(SIGNAL_PARAMETERS)
    else:
        reserved_parameters |= set(spec.parameters) & set(SIGNAL_PARAMETERS)
    if reserved_parameters:
        problems.append("Зарезервированные имена параметров: " + ", ".join(sorted(reserved_parameters)))
    reserved_variables = set(spec.variables) & RESERVED_IDENTIFIERS
    if spec.signals:
        reserved_variables |= set(spec.variables) & set(SIGNAL_PARAMETERS)
    if reserved_variables:
        problems.append("Зарезервированные имена переменных: " + ", ".join(sorted(reserved_variables)))
    collisions = set(spec.variables) & set(spec.parameters)
    if collisions:
        problems.append("Имена одновременно используются как переменные и параметры: " + ", ".join(sorted(collisions)))
    known = set(spec.variables) | set(spec.parameters)
    for equation in spec.equations:
        if equation.kind == EquationKind.SIGNAL:
            continue
        try:
            unknown = {str(s) for s in sympify_expression(equation.expression).free_symbols} - known
            if unknown:
                problems.append(f"{equation.target}: неизвестные символы {', '.join(sorted(unknown))}")
        except Exception as exc:
            problems.append(f"{equation.target}: {exc}")
    try:
        algebraic_order(spec)
    except ParseError as exc:
        problems.append(str(exc))
    assignable_names = set(spec.states) | set(spec.algebraic) | set(spec.markers)
    for event_index, event in enumerate(spec.events, 1):
        if event.after_target is not None and event.after_target not in set(spec.variables):
            problems.append(
                f"Условие {event_index}: неизвестное уравнение для позиции {event.after_target!r}"
            )
        try:
            condition = sympify_expression(event.condition)
            if not isinstance(condition, (sp.logic.boolalg.Boolean, sp.core.relational.Relational)):
                problems.append(f"Условие {event_index} должно быть сравнением или логическим выражением")
            unknown = {str(s) for s in condition.free_symbols} - known
            if unknown:
                problems.append(f"Условие {event_index}: неизвестные символы {', '.join(sorted(unknown))}")
        except Exception as exc:
            problems.append(f"Условие {event_index}: {exc}")
        if not event.assignments:
            problems.append(f"Условие {event_index} не содержит присваиваний")
        for target, expression in event.assignments.items():
            if target not in assignable_names:
                problems.append(
                    f"Условие {event_index}: нельзя присваивать входному сигналу {target!r}"
                )
            try:
                unknown = {str(s) for s in sympify_expression(expression).free_symbols} - known
                if unknown:
                    problems.append(f"Условие {event_index}, {target}: неизвестные символы {', '.join(sorted(unknown))}")
            except Exception as exc:
                problems.append(f"Условие {event_index}, {target}: {exc}")
    return problems


def ensure_signal_support(spec: SystemSpec) -> None:
    """Add CUDAynamics' time state and standard waveform parameters when needed."""
    if not spec.signals:
        return
    if "t" not in spec.states:
        spec.equations.append(Equation("t", "1", EquationKind.STATE, "dt/dt = 1 (добавлено для сигнала)"))
    spec.parameters = [
        name for name in spec.parameters
        if name not in spec.signals and name not in SIGNAL_PARAMETERS
    ]
    defaults = {"Idc": 0.0, "Iamp": 1.0, "Ifreq": 1.0, "Idel": 0.0, "Idf": 0.5}
    for name in SIGNAL_PARAMETERS:
        if name not in spec.parameters:
            spec.parameters.append(name)
        if name not in spec.parameter_defaults:
            value = defaults[name]
            spec.parameter_defaults[name] = AttributeDefaults(value=value, minimum=value)
    spec.ensure_defaults()


def _custom_identifier(name: str, occupied: set[str]) -> str:
    candidate = f"{name}_custom"
    suffix = 2
    while candidate in occupied or candidate in RESERVED_IDENTIFIERS:
        candidate = f"{name}_custom{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _rename_expression(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    expression = sympify_expression(text)
    replacements = {sp.Symbol(old): sp.Symbol(new) for old, new in mapping.items()}
    return str(expression.xreplace(replacements))


def _canonical_time_equation(equation: Equation) -> bool:
    if equation.target != "t" or equation.kind != EquationKind.STATE:
        return False
    try:
        return sp.simplify(sympify_expression(equation.expression) - 1) == 0
    except Exception:
        return False


def _materialize_event_targets(spec: SystemSpec) -> None:
    """Turn unknown assignment targets into persistent, non-integrated variables.

    A marker keeps its previous value until a conditional event changes it.  In
    contrast to an algebraic variable it has no formula that would overwrite
    the event assignment on every integration stage.
    """
    known = set(spec.variables)
    for event in spec.events:
        for target in event.assignments:
            if target not in known:
                spec.equations.append(
                    Equation(target, target, EquationKind.MARKER, "создано условным оператором")
                )
                known.add(target)


def normalize_reserved_identifiers(spec: SystemSpec) -> list[str]:
    """Rename user identifiers that collide with generated CUDAynamics names.

    Renaming is symbolic and therefore also updates nested equations and event
    expressions. A canonical ``dt/dt = 1`` remains the internal time state.
    """
    warnings: list[str] = []
    prior_mapping = {
        old: new for old, new in spec.identifier_renames.items()
        # Once internal time exists, newly entered "t" means that time. The
        # original user state remains available under the displayed t_custom.
        if not (old == "t" and "t" in spec.states)
    }
    if prior_mapping:
        for equation in spec.equations:
            if equation.kind != EquationKind.SIGNAL:
                equation.expression = _rename_expression(equation.expression, prior_mapping)
        for event in spec.events:
            event.condition = _rename_expression(event.condition, prior_mapping)
            event.assignments = {
                prior_mapping.get(target, target): _rename_expression(expression, prior_mapping)
                for target, expression in event.assignments.items()
            }
            if event.after_target is not None:
                event.after_target = prior_mapping.get(event.after_target, event.after_target)
    _materialize_event_targets(spec)
    all_symbols = {
        str(symbol)
        for equation in spec.equations if equation.kind != EquationKind.SIGNAL
        for symbol in sympify_expression(equation.expression).free_symbols
    }
    occupied = {equation.target for equation in spec.equations} | all_symbols | set(spec.parameters)
    target_mapping: dict[str, str] = {}
    for equation in spec.equations:
        collision = equation.target in RESERVED_IDENTIFIERS
        if equation.target == "t" and spec.signals and not _canonical_time_equation(equation):
            collision = True
        if equation.target in SIGNAL_PARAMETERS and spec.signals:
            collision = True
        if collision and equation.target not in target_mapping:
            renamed = _custom_identifier(equation.target, occupied)
            target_mapping[equation.target] = renamed
            warnings.append(
                f"Переменная {equation.target!r} переименована в {renamed!r}: имя занято CUDAynamics."
            )

    if target_mapping:
        for equation in spec.equations:
            old_target = equation.target
            equation.target = target_mapping.get(old_target, old_target)
            if equation.kind != EquationKind.SIGNAL:
                equation.expression = _rename_expression(equation.expression, target_mapping)
        for event in spec.events:
            event.condition = _rename_expression(event.condition, target_mapping)
            event.assignments = {
                target_mapping.get(target, target): _rename_expression(expression, target_mapping)
                for target, expression in event.assignments.items()
            }
            if event.after_target is not None:
                event.after_target = target_mapping.get(event.after_target, event.after_target)
        for old, new in target_mapping.items():
            if old in spec.variable_defaults and new not in spec.variable_defaults:
                spec.variable_defaults[new] = spec.variable_defaults.pop(old)

    # User-written t without a target denotes independent time, as before.
    symbols_after_targets = {
        str(symbol)
        for equation in spec.equations if equation.kind != EquationKind.SIGNAL
        for symbol in sympify_expression(equation.expression).free_symbols
    }
    if "t" in symbols_after_targets and "t" not in spec.variables:
        spec.equations.append(Equation("t", "1", EquationKind.STATE, "dt/dt = 1 (добавлено автоматически)"))

    ensure_signal_support(spec)

    symbols = {
        str(symbol)
        for equation in spec.equations if equation.kind != EquationKind.SIGNAL
        for symbol in sympify_expression(equation.expression).free_symbols
    }
    for event in spec.events:
        symbols |= {str(symbol) for symbol in sympify_expression(event.condition).free_symbols}
        for expression in event.assignments.values():
            symbols |= {str(symbol) for symbol in sympify_expression(expression).free_symbols}

    parameter_names = symbols - set(spec.variables) - {"t"}
    parameter_mapping: dict[str, str] = {}
    for name in sorted(parameter_names):
        is_signal_control = bool(spec.signals) and name in SIGNAL_PARAMETERS
        if name in RESERVED_IDENTIFIERS and not is_signal_control:
            renamed = _custom_identifier(name, occupied)
            parameter_mapping[name] = renamed
            warnings.append(f"Параметр {name!r} переименован в {renamed!r}: имя занято CUDAynamics.")

    if parameter_mapping:
        for equation in spec.equations:
            if equation.kind != EquationKind.SIGNAL:
                equation.expression = _rename_expression(equation.expression, parameter_mapping)
        for event in spec.events:
            event.condition = _rename_expression(event.condition, parameter_mapping)
            event.assignments = {
                target: _rename_expression(expression, parameter_mapping)
                for target, expression in event.assignments.items()
            }
        for old, new in parameter_mapping.items():
            if old in spec.parameter_defaults and new not in spec.parameter_defaults:
                spec.parameter_defaults[new] = spec.parameter_defaults.pop(old)

    renamed_symbols = {parameter_mapping.get(name, name) for name in parameter_names}
    ordinary = sorted(renamed_symbols - set(SIGNAL_PARAMETERS))
    spec.parameters = ordinary + (list(SIGNAL_PARAMETERS) if spec.signals else [])
    spec.identifier_renames.update(target_mapping)
    spec.identifier_renames.update(parameter_mapping)
    spec.rename_warnings = warnings
    spec.ensure_defaults()
    return warnings


def parse_assignments(text: str) -> dict[str, str]:
    """Parse ``v = c; u = d`` into validated symbolic assignment strings."""
    result: dict[str, str] = {}
    for item in re.split(r"[;\n]+", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ParseError(f"В присваивании нет '=': {item!r}")
        target, expression = item.split("=", 1)
        target = safe_identifier(target.strip())
        if target in result:
            raise ParseError(f"Переменная {target!r} присваивается дважды в одном условии")
        result[target] = str(sympify_expression(expression.strip()))
    if not result:
        raise ParseError("Добавьте хотя бы одно присваивание, например v = c; u = d")
    return result


def pretty_equations(spec: SystemSpec, language: str = "ru") -> str:
    rows = []
    for equation in spec.equations:
        if equation.kind == EquationKind.STATE:
            lhs = f"d{equation.target}/dt"
        elif equation.kind == EquationKind.SIGNAL:
            lhs = f"{equation.target} ({'сигнал' if language == 'ru' else 'signal'})"
        elif equation.kind == EquationKind.MARKER:
            lhs = f"{equation.target} ({'задаётся условием' if language == 'ru' else 'set by condition'})"
        else:
            lhs = equation.target
        rhs = ("выбранная форма(t)" if language == "ru" else "selected waveform(t)") if equation.kind == EquationKind.SIGNAL else sp.sstr(sympify_expression(equation.expression))
        rows.append(f"{lhs} = {rhs}")
    for event in spec.events:
        assignments = "; ".join(f"{name} = {value}" for name, value in event.assignments.items())
        if language == "ru":
            position = f"после {event.after_target}" if event.after_target else "в конце шага"
        else:
            position = f"after {event.after_target}" if event.after_target else "at end of step"
        rows.append(f"if {event.condition}: {assignments} ({position})")
    return "\n".join(rows)
