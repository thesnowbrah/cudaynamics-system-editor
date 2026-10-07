from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class EquationKind(str, Enum):
    STATE = "state"
    ALGEBRAIC = "algebraic"
    SIGNAL = "signal"
    MARKER = "marker"


@dataclass(slots=True)
class Equation:
    target: str
    expression: str
    kind: EquationKind
    source: str = ""


@dataclass(slots=True)
class ConditionalEvent:
    """A post-step state reset guarded by a boolean condition."""

    condition: str
    assignments: dict[str, str] = field(default_factory=dict)
    after_target: str | None = None


@dataclass(slots=True)
class AttributeDefaults:
    value: float = 0.0
    range_kind: str = "Fixed"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    count: int = 100
    mean: float = 0.0
    deviation: float = 0.0


METHODS = {
    "ExplicitEuler": "Явный Эйлер",
    "SemiExplicitEuler": "Полуявный Эйлер",
    "ImplicitEuler": "Неявный Эйлер (Ньютон)",
    "ExplicitMidpoint": "Явная средняя точка",
    "ImplicitMidpoint": "Неявная средняя точка (Ньютон)",
    "ExplicitRungeKutta4": "Рунге—Кутта 4",
    "ExplicitDormandPrince8": "Dormand—Prince 8 (фиксированный шаг)",
    "VariableSymmetryCD": "Variable Symmetry CD",
}

SIGNAL_PARAMETERS = ("Idc", "Iamp", "Ifreq", "Idel", "Idf")


def time_to_steps(duration: float, step: float, *, allow_zero: bool = False) -> int:
    """Convert user-facing simulation time to an integer CUDAynamics step count."""
    if not math.isfinite(duration) or not math.isfinite(step):
        raise ValueError("Время и шаг должны быть конечными числами")
    if step <= 0:
        raise ValueError("Шаг h должен быть больше нуля")
    if duration < 0 or (duration == 0 and not allow_zero):
        qualifier = "неотрицательным" if allow_zero else "больше нуля"
        raise ValueError(f"Время должно быть {qualifier}")
    if duration == 0:
        return 0
    ratio = duration / step
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return max(1, nearest)
    return max(1, math.ceil(ratio))


@dataclass(slots=True)
class SystemSpec:
    system_id: str = "new_system"
    display_name: str = "New system"
    equations: list[Equation] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    variable_defaults: dict[str, AttributeDefaults] = field(default_factory=dict)
    parameter_defaults: dict[str, AttributeDefaults] = field(default_factory=dict)
    methods: list[str] = field(default_factory=lambda: ["ExplicitEuler", "ExplicitRungeKutta4"])
    default_method: str | None = None
    step: float = 0.01
    steps: int = 1000
    transient: int = 0
    execute_on_launch: bool = False
    symmetry_default: float = 0.0
    events: list[ConditionalEvent] = field(default_factory=list)
    rename_warnings: list[str] = field(default_factory=list)
    identifier_renames: dict[str, str] = field(default_factory=dict)

    @property
    def states(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.STATE]

    @property
    def algebraic(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.ALGEBRAIC]

    @property
    def signals(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.SIGNAL]

    @property
    def markers(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.MARKER]

    @property
    def variables(self) -> list[str]:
        return self.states + [x for x in self.algebraic if x not in self.states] + [
            x for x in self.signals if x not in self.states and x not in self.algebraic
        ] + [
            x for x in self.markers
            if x not in self.states and x not in self.algebraic and x not in self.signals
        ]

    def ensure_defaults(self) -> None:
        for name in self.variables:
            self.variable_defaults.setdefault(name, AttributeDefaults())
        for name in self.parameters:
            self.parameter_defaults.setdefault(name, AttributeDefaults(value=1.0, minimum=1.0))
        signal_values = {"Idc": 0.0, "Iamp": 1.0, "Ifreq": 1.0, "Idel": 0.0, "Idf": 0.5}
        if self.signals:
            for name, value in signal_values.items():
                defaults = self.parameter_defaults.setdefault(
                    name, AttributeDefaults(value=value, minimum=value)
                )
                if name not in self.parameters:
                    self.parameters.append(name)
