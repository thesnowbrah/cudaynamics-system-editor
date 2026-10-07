from pathlib import Path

import pytest
import sympy as sp

from cudynamics_builder.generator import _waveform_expression, generate
from cudynamics_builder.editor import build_system_edit_plan, load_editable_system, validate_renames
from cudynamics_builder.installer import (
    build_plan, build_removal_plan, ensure_unique_system_identity, list_library_systems,
)
from cudynamics_builder.model import ConditionalEvent, EquationKind, METHODS, time_to_steps
from cudynamics_builder.parser import (
    ParseError, algebraic_order, normalize_reserved_identifiers, parse_system, validate_spec,
)


LORENZ_WITH_ALGEBRAIC = """dx/dt = sigma*(y-x)
dy/dt = x*(rho-z)-y
dz/dt = x*y-beta*z
energy = x^2+y^2+z^2
"""

PIX2TEX_ROSSLER = r"""\left\{\begin{array}{l} {{\frac{d x}{d t}=-y-z}} \\ {{\frac{d y}{d t}=x+a y}} \\ {{\frac{d z}{d t}=b+z(x-c)}}\end{array}\right."""
PIX2TEX_NESTED_ARRAY_ROSSLER = r"""\left\{\begin{array}{r} {{\begin{array}{r} {{\frac{d x}{d t}}=-y-z}} \\ {{\frac{d y}{d t}}=x+a y}} \\ {{\frac{d z}{d t}=b+z(x-c)}}\end{array}\right."""


def test_parser_classifies_states_algebraic_and_parameters():
    spec = parse_system(LORENZ_WITH_ALGEBRAIC, "generated_lorenz", "Generated Lorenz")
    assert spec.states == ["x", "y", "z"]
    assert spec.algebraic == ["energy"]
    assert set(spec.parameters) == {"sigma", "rho", "beta"}
    assert not validate_spec(spec)


def test_time_is_added_as_state():
    spec = parse_system("dx/dt = sin(t)-a*x", "forced", "Forced")
    assert "t" in spec.states
    assert "a" in spec.parameters


def test_pix2tex_array_output_is_normalized_automatically():
    spec = parse_system(PIX2TEX_ROSSLER, "rossler_test", "Rossler Test")
    assert spec.states == ["x", "y", "z"]
    assert set(spec.parameters) == {"a", "b", "c"}
    assert spec.equations[0].expression == "-y - z"
    assert "a*y" in spec.equations[1].expression
    assert sp.simplify(sp.sympify(spec.equations[2].expression) - sp.sympify("b + z*(x-c)")) == 0


def test_pix2tex_malformed_nested_array_output_is_normalized():
    spec = parse_system(PIX2TEX_NESTED_ARRAY_ROSSLER, "nested_rossler", "Nested Rossler")
    assert spec.states == ["x", "y", "z"]
    assert set(spec.parameters) == {"a", "b", "c"}
    assert not validate_spec(spec)


def test_main_math_functions_parse_and_generate_cuda_expressions():
    text = "dx/dt = sin(x)+cos(x)+sec(x)+csc(x)+cot(x)+abs(x)+sign(x)+max(a,x)-min(b,x)+sqrt(c)+exp(-x)+log(d)+log10(d)+log2(d)+floor(e)+ceil(f)+atan2(x,a)+hypot(x,b)"
    spec = parse_system(text, "functions", "Functions")
    assert set(spec.parameters) == {"a", "b", "c", "d", "e", "f"}
    source = generate(spec).files["systems/functions/functions.cu"]
    for token in ("sin(", "cos(", "tan(", "fabs(", "fmax(", "fmin(", "sqrt(", "exp(", "log(", "floor(", "ceil(", "atan2(", "> 0"):
        assert token in source


def test_latex_math_functions_from_pix2tex_are_supported():
    spec = parse_system(r"\frac{d x}{d t}=\sin(x)+\cos(x)+\max(a,x)+\min(b,x)", "latex_functions", "LaTeX functions")
    assert spec.states == ["x"]
    assert set(spec.parameters) == {"a", "b"}


def test_algebraic_cycle_is_rejected():
    spec = parse_system("dx/dt = q\nq = r+x\nr = q-x", "cycle", "Cycle")
    assert any("Циклическая" in error for error in validate_spec(spec))


def test_generation_contains_all_selected_methods_and_algebraic_eval():
    spec = parse_system(LORENZ_WITH_ALGEBRAIC, "generated_lorenz", "Generated Lorenz")
    spec.methods = list(("ExplicitEuler", "SemiExplicitEuler", "ImplicitEuler", "ExplicitMidpoint", "ImplicitMidpoint", "ExplicitRungeKutta4", "ExplicitDormandPrince8", "VariableSymmetryCD"))
    project = generate(spec)
    source = project.files["systems/generated_lorenz/generated_lorenz.cu"]
    header = project.files["systems/generated_lorenz/generated_lorenz.h"]
    for method in spec.methods:
        assert method in source
    assert "Vnext(energy)" in source
    assert "Vnext(x) = V(x) + H *" in source
    assert "numb xmp = V(x) + (numb)0.5 * H" in source
    assert "numb kx1 =" in source
    assert "evaluate_(" not in source and "derivative[" not in source
    assert "numb A[3][3]" in source
    assert "const numb eps" not in source and "shifted[" not in source
    assert "Exact affine solution of the reverse VSCD coordinate equation" in source
    assert "iteration < 6" not in source
    assert "oldY[v] - H * f" not in source
    assert "const int THREADS_PER_BLOCK_(name) = 64;" in header


def test_symbolic_jacobian_expands_algebraic_dependencies_and_abs():
    spec = parse_system("dx/dt = q + abs(x)\nq = x*x", "analytic", "Analytic")
    spec.methods = ["ImplicitEuler"]
    source = generate(spec).files["systems/analytic/analytic.cu"]
    jacobian_line = next(line for line in source.splitlines() if "numb A[1][1]" in line)
    assert "2*" in jacobian_line
    assert "> 0" in jacobian_line and "< 0" in jacobian_line
    assert "Vnext(q)" not in jacobian_line


def test_vscd_uses_analytic_scalar_newton_for_nonlinear_coordinate():
    spec = parse_system("dx/dt = sin(x) + a", "nonlinear_vscd", "Nonlinear VSCD")
    spec.methods = ["VariableSymmetryCD"]
    source = generate(spec).files["systems/nonlinear_vscd/nonlinear_vscd.cu"]
    assert "analytic scalar Newton" in source
    assert "vscdDerivative_x" in source
    assert "cos(vscdCandidate_x)" in source
    assert "evaluate_(name)(work" not in source


def test_metadata_writes_count_ranges_for_variables_and_parameters():
    spec = parse_system("dx/dt = a*x", "ranges", "Ranges")
    spec.variable_defaults["x"].range_kind = "Linear"
    spec.variable_defaults["x"].value = -2
    spec.variable_defaults["x"].maximum = 3
    spec.variable_defaults["x"].count = 51
    spec.parameter_defaults["a"].range_kind = "Linear"
    spec.parameter_defaults["a"].value = 0.1
    spec.parameter_defaults["a"].maximum = 1.1
    spec.parameter_defaults["a"].count = 11
    metadata = generate(spec).files["systems/ranges/ranges.txt"]
    assert "var x Linear -2 3 0.01 51" in metadata
    assert "param a Linear 0.1 1.1 0.01 11" in metadata


def test_creation_writes_selected_default_integration_method():
    spec = parse_system("dx/dt = a*x", "default_method", "Default method")
    spec.methods = ["ExplicitEuler", "ExplicitRungeKutta4"]
    spec.default_method = "ExplicitRungeKutta4"
    metadata = generate(spec).files["systems/default_method/default_method.txt"]
    assert "enum method 0ExplicitEuler 1ExplicitRungeKutta4" in metadata


def test_real_project_install_plan_is_read_only_and_complete():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not root.exists():
        pytest.skip("CUDAynamics project is not adjacent")
    spec = parse_system("dx/dt = a*x", "builder_test_only", "Builder test")
    plan = build_plan(root, spec, generate(spec))
    assert len(plan.writes) == 7
    assert "addKernel(builder_test_only)" in plan.diff
    assert not (root / "systems" / "builder_test_only").exists()


def test_install_plan_preserves_uploaded_source_file():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not root.exists():
        pytest.skip("CUDAynamics project is not adjacent")
    source = Path(__file__).resolve()
    spec = parse_system("dx/dt = a*x", "builder_source_test", "Builder source test")
    plan = build_plan(root, spec, generate(spec), source)
    destination = root / "systems" / "builder_source_test" / "builder_source_test.py"
    assert plan.writes[destination] == source.read_bytes()
    assert "Binary source file: systems/builder_source_test/builder_source_test.py" in plan.diff


def test_existing_id_and_display_name_receive_numeric_suffixes():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "lorenz").exists():
        pytest.skip("Reference Lorenz system is unavailable")
    spec = parse_system("dx/dt = a*x", "LORENZ", "Lorenz system")
    messages = ensure_unique_system_identity(root, spec)
    assert spec.system_id.casefold() != "lorenz"
    assert spec.system_id.casefold().startswith("lorenz")
    assert spec.system_id[-1].isdigit()
    assert spec.display_name != "Lorenz system"
    assert spec.display_name[-1].isdigit()
    assert len(messages) == 2


def test_removal_plan_removes_system_registration_without_touching_project():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "lorenz").exists():
        pytest.skip("Reference Lorenz system is unavailable")
    before = (root / "systemsHeaders.h").read_bytes()
    systems = list_library_systems(root)
    lorenz = next(system for system in systems if system.system_id == "lorenz")
    assert lorenz.display_name == "Lorenz system"
    plan = build_removal_plan(root, ["lorenz"])
    assert len(plan.updated_files) == 4
    assert plan.directories == [root / "systems" / "lorenz"]
    assert "systems/lorenz/lorenz.h" not in plan.updated_files[root / "systemsHeaders.h"]
    assert "addKernel(lorenz);" not in plan.updated_files[root / "main.cpp"]
    assert "systems\\lorenz\\lorenz.cu" not in plan.updated_files[root / "CUDAynamics.vcxproj"]
    assert (root / "systemsHeaders.h").read_bytes() == before
    assert (root / "systems" / "lorenz").is_dir()


def test_system_editor_renames_identifiers_and_preserves_equations_and_config_aliases():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "lorenz").exists():
        pytest.skip("Reference Lorenz system is unavailable")
    model = load_editable_system(root, "lorenz")
    assert [item.original for item in model.variables] == ["x", "y", "z"]
    assert [item.original for item in model.parameters] == ["sigma", "rho", "beta"]
    model.system_id = "lorenz_editor_test"
    model.display_name = "Lorenz editor test"
    # A swap verifies that replacements are simultaneous rather than chained.
    model.variables[0].edited = "y"
    model.variables[1].edited = "x"
    model.parameters[0].edited = "sigma_new"
    model.config_attributes[0].minimum = "12.5"
    plan = build_system_edit_plan(model)
    source = plan.directory_files[Path("lorenz_editor_test.cu")]
    metadata = plan.directory_files[Path("lorenz_editor_test.txt")]
    assert "enum variables { y, x, z }" in source
    assert "Vnext(y) = V(y)" in source
    assert "P(sigma_new)" in source and "P(sigma)" not in source
    assert "var y Fixed 12.5" in metadata
    assert "param sigma_new" in metadata
    # Unchanged rho retains the library's Unicode display alias.
    assert "param ρ" in metadata
    assert "addKernel(lorenz_editor_test);" in plan.registration_files[root / "main.cpp"]
    assert not (root / "systems" / "lorenz_editor_test").exists()


def test_system_editor_rejects_duplicate_attribute_names():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "lorenz").exists():
        pytest.skip("Reference Lorenz system is unavailable")
    model = load_editable_system(root, "lorenz")
    model.variables[1].edited = model.variables[0].edited
    assert any("Duplicate attribute names" in problem for problem in validate_renames(model))


def test_system_editor_changes_only_default_flag_for_existing_methods():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "lorenz").exists():
        pytest.skip("Reference Lorenz system is unavailable")
    model = load_editable_system(root, "lorenz")
    assert model.default_method == "ExplicitEuler"
    assert "ExplicitRungeKutta4" in model.methods
    model.default_method = "ExplicitRungeKutta4"
    plan = build_system_edit_plan(model)
    metadata = plan.directory_files[Path("lorenz.txt")]
    assert "enum method 0ExplicitEuler 0ExplicitMidpoint 1ExplicitRungeKutta4 0VariableSymmetryCD" in metadata


def test_system_editor_accepts_legacy_configuration_without_method_enum():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not (root / "systems" / "bolshakov").exists():
        pytest.skip("Reference Bolshakov system is unavailable")
    model = load_editable_system(root, "bolshakov")
    assert model.methods == []
    assert model.default_method == ""
    plan = build_system_edit_plan(model)
    metadata = plan.directory_files[Path("bolshakov.txt")]
    assert "enum method" not in metadata


def test_sinusoidal_lorenz_variation_generates_supported_functions_for_every_method():
    spec = parse_system(
        "dx/dt=sigma*(-x+y)+k*sin(y/gamma)*sin(z/gamma)\n"
        "dy/dt=-x*z+rho*x-y+k*sin(x/gamma)*sin(z/gamma)\n"
        "dz/dt=x*y-beta*z+k*cos(y/gamma)*cos(x/gamma)",
        "lorenz_sine", "Sinusoidal Lorenz",
    )
    spec.methods = list(METHODS)
    source = generate(spec).files["systems/lorenz_sine/lorenz_sine.cu"]
    assert not validate_spec(spec)
    assert source.count("sin(") > 3 and source.count("cos(") > 3
    assert "numb A[3][3]" in source


def test_function_of_time_is_detected_as_signal_and_generates_three_waveform_blocks():
    spec = parse_system(
        "dx/dt=(mu-x^2-y^2)*x-omega*y+k*(F(t)-x*alpha)\n"
        "dy/dt=(mu-x^2-y^2)*y+omega*x\n"
        "domega/dt=-k*(F(t)-x*alpha)*y\n"
        "dalpha/dt=n*(F(t)-x*alpha)*x",
        "adaptive_hopf", "Adaptive Hopf",
    )
    spec.methods = ["ExplicitEuler", "ExplicitRungeKutta4", "ImplicitEuler", "VariableSymmetryCD"]
    source = generate(spec).files["systems/adaptive_hopf/adaptive_hopf.cu"]
    metadata = generate(spec).files["systems/adaptive_hopf/adaptive_hopf.txt"]
    assert spec.signals == ["F"]
    assert "t" in spec.states and "F" not in spec.parameters
    assert source.count("ifSIGNAL(P(signal),") == 3
    assert "ifSIGNAL(P(signal), square)" in source
    assert "ifSIGNAL(P(signal), sine)" in source
    assert "ifSIGNAL(P(signal), triangle)" in source
    assert source.count("ifMETHOD(P(method),") == 3 * len(spec.methods)
    assert "Vnext(F)" in source and "Vnext(t)" in source
    assert "enum waveforms { square, sine, triangle };" in source
    assert "M_PI" not in source and "Derivative(" not in source and "Subs(" not in source
    assert "enum signal 1square 0sine 0triangle" in metadata
    assert "param Idc Fixed 0 " in metadata
    assert "param Idf Fixed 0.5 " in metadata


def test_nested_algebraic_dependencies_are_eliminated_from_implicit_and_vscd_math():
    spec = parse_system(
        "dx/dt = r + sin(q)\n"
        "q = x*x + a\n"
        "r = q*y\n"
        "dy/dt = q-x",
        "nested_algebraic", "Nested algebraic",
    )
    spec.methods = ["ImplicitEuler", "ImplicitMidpoint", "SemiExplicitEuler", "VariableSymmetryCD"]
    source = generate(spec).files["systems/nested_algebraic/nested_algebraic.cu"]
    assert not validate_spec(spec)
    math_before_outputs = source.split("Vnext(q) =", 1)[0]
    assert "V(q)" not in math_before_outputs and "V(r)" not in math_before_outputs
    assert "numb A[2][2]" in source
    assert "vscdDerivative_" in source


def test_post_step_condition_is_emitted_after_each_method_with_simultaneous_resets():
    spec = parse_system("dv/dt=a*u+b*v+i\ndu/dt=b*u-a*v", "resonate", "Resonate and fire")
    spec.events = [ConditionalEvent("v >= vPeak", {"v": "c", "u": "d"})]
    spec.parameters += ["vPeak", "c", "d"]
    spec.parameters = sorted(set(spec.parameters))
    spec.ensure_defaults()
    spec.methods = ["ExplicitEuler", "ExplicitRungeKutta4"]
    source = generate(spec).files["systems/resonate/resonate.cu"]
    assert not validate_spec(spec)
    assert source.count("if (Vnext(v) >= P(vPeak))") == len(spec.methods)
    assert source.count("const numb event0_v = P(c);") == len(spec.methods)
    assert source.count("const numb event0_u = P(d);") == len(spec.methods)
    assert source.index("const numb event0_v") < source.index("Vnext(v) = event0_v")


def test_waveform_equations_have_expected_values():
    values = {"t": 0, "Idc": 2, "Iamp": 3, "Ifreq": 1, "Idel": 0, "Idf": sp.Rational(1, 4)}
    assert _waveform_expression("square").subs(values | {"t": sp.Rational(1, 10)}) == 5
    assert _waveform_expression("square").subs(values | {"t": sp.Rational(1, 3)}) == 2
    assert abs(float(_waveform_expression("sine").subs(values | {"t": sp.Rational(1, 4)})) - 5) < 1e-12
    triangle = _waveform_expression("triangle")
    assert triangle.subs(values) == -1
    assert triangle.subs(values | {"t": sp.Rational(1, 4)}) == 2
    assert triangle.subs(values | {"t": sp.Rational(1, 2)}) == 5


def test_signal_and_reset_event_are_combined_in_every_waveform_branch():
    spec = parse_system("dv/dt=a*u+b*v+I(t)\ndu/dt=b*u-a*v", "resonate_signal", "Resonate signal")
    spec.events = [ConditionalEvent("v >= vPeak", {"v": "c", "u": "d"})]
    spec.parameters = sorted(set(spec.parameters + ["vPeak", "c", "d"]))
    spec.ensure_defaults()
    spec.methods = ["ExplicitEuler", "ExplicitRungeKutta4"]
    source = generate(spec).files["systems/resonate_signal/resonate_signal.cu"]
    assert source.count("ifSIGNAL(P(signal),") == 3
    assert source.count("if (Vnext(v) >= P(vPeak))") == 3 * len(spec.methods)
    sine_block = source.split("ifSIGNAL(P(signal), sine)", 1)[1].split("ifSIGNAL(P(signal), triangle)", 1)[0]
    assert "V(t)" in sine_block and "tmp" in sine_block


def test_reserved_state_and_parameter_names_are_renamed_symbolically():
    spec = parse_system(
        "dh/dt = h + method + symmetry + COUNT",
        "reserved_names", "Reserved names",
    )
    assert spec.states == ["h_custom"]
    assert set(spec.parameters) == {"method_custom", "symmetry_custom", "COUNT_custom"}
    assert "h_custom" in spec.equations[0].expression
    assert len(spec.rename_warnings) == 4
    source = generate(spec).files["systems/reserved_names/reserved_names.cu"]
    assert "enum variables { h_custom }" in source
    assert "method_custom" in source and "symmetry_custom" in source and "COUNT_custom" in source
    assert not validate_spec(spec)


def test_reserved_rename_uses_unique_suffix_and_updates_events():
    spec = parse_system(
        "dmethod/dt = h + method_custom\ndmethod_custom/dt = method",
        "unique_reserved", "Unique reserved",
    )
    spec.events = [ConditionalEvent("method >= limit", {"method": "h"})]
    # Re-normalization also covers event rows added after the initial parse.
    from cudynamics_builder.parser import normalize_reserved_identifiers
    normalize_reserved_identifiers(spec)
    assert "method_custom2" in spec.states
    assert spec.events[0].condition == "method_custom2 >= limit"
    assert spec.events[0].assignments == {"method_custom2": "h_custom"}
    assert not validate_spec(spec)


def test_user_t_is_renamed_only_when_signal_needs_internal_time():
    conflicting = parse_system("dt/dt=-t\ndx/dt=F(t)+t", "time_conflict", "Time conflict")
    assert conflicting.states == ["t_custom", "x", "t"]
    assert conflicting.signals == ["F"]
    assert conflicting.equations[1].expression == "F + t_custom"
    assert any("'t'" in warning and "'t_custom'" in warning for warning in conflicting.rename_warnings)
    conflicting.events = [ConditionalEvent("t >= stop", {"x": "t_custom"})]
    from cudynamics_builder.parser import normalize_reserved_identifiers
    normalize_reserved_identifiers(conflicting)
    assert conflicting.events[0].condition == "t >= stop"
    assert conflicting.events[0].assignments == {"x": "t_custom"}

    canonical = parse_system("dt/dt=1\ndx/dt=F(t)+t", "time_ok", "Time OK")
    assert canonical.states == ["t", "x"]
    assert canonical.rename_warnings == []


def test_cuda_object_macro_names_are_also_reserved():
    spec = parse_system("dH/dt = name + numb", "macro_names", "Macro names")
    assert spec.states == ["H_custom"]
    assert set(spec.parameters) == {"name_custom", "numb_custom"}
    source = generate(spec).files["systems/macro_names/macro_names.cu"]
    assert "enum variables { H_custom }" in source
    assert "P(name_custom)" in source and "P(numb_custom)" in source


def test_unknown_event_targets_become_markers_and_other_symbols_parameters():
    spec = parse_system("dx/dt = a*x", "generic_event", "Generic event")
    spec.events = [ConditionalEvent(
        "variable1 >= parameter1",
        {"variable1": "parameter2", "variable2": "parameter2"},
    )]
    normalize_reserved_identifiers(spec)
    assert spec.markers == ["variable1", "variable2"]
    assert {"parameter1", "parameter2"} <= set(spec.parameters)
    assert not validate_spec(spec)
    source = generate(spec).files["systems/generic_event/generic_event.cu"]
    assert "Vnext(variable1) = V(variable1);" in source
    assert "Vnext(variable2) = V(variable2);" in source
    assert "if (Vnext(variable1) >= P(parameter1))" in source


def test_event_can_assign_algebraic_and_is_emitted_after_selected_equation():
    spec = parse_system("dx/dt = y\ndy/dt = -x\nq = x+y", "positioned", "Positioned")
    spec.events = [ConditionalEvent("q > limit", {"q": "reset"}, after_target="x")]
    normalize_reserved_identifiers(spec)
    spec.methods = ["ExplicitEuler"]
    assert not validate_spec(spec)
    source = generate(spec).files["systems/positioned/positioned.cu"]
    x_update = source.index("Vnext(x) = V(x) + H")
    event = source.index("if (V(q) > P(limit))")
    y_update = source.index("Vnext(y) = V(y) + H")
    assert x_update < event < y_update
    assert "Vnext(q) = event0_q;" in source


def test_simulation_time_is_converted_to_integer_step_counts():
    assert time_to_steps(10.0, 0.01) == 1000
    assert time_to_steps(0.0, 0.01, allow_zero=True) == 0
    # A non-divisible duration rounds upward so generated simulation time is
    # never shorter than requested.
    assert time_to_steps(1.001, 0.01) == 101
    with pytest.raises(ValueError):
        time_to_steps(0.0, 0.01)
