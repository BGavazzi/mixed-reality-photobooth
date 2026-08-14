"""
Unit tests for doctor.py's decision logic.

The probing half (sockets, HTTP, importlib) is I/O and isn't exercised here.
What is exercised is every judgement the tool makes -- which is the part that
has to be right, because a diagnostic that misreports a working install is
worse than no diagnostic at all. It's also why the checks take their facts as
arguments instead of discovering them: they can be tested against a broken
environment without breaking this one.
"""

import sys

import pytest

import doctor


def installed(*present):
    """Builds an is_installed() that reports only the named modules."""
    return lambda module: module in set(present)


# --- python version ---------------------------------------------------------

@pytest.mark.parametrize("version", [(3, 10, 0), (3, 12, 10), (4, 0, 0)])
def test_supported_python_versions_pass(version):
    assert doctor.check_python_version(version).level == doctor.OK


@pytest.mark.parametrize("version", [(3, 9, 18), (2, 7, 18), (3, 8, 0)])
def test_old_python_is_a_hard_failure(version):
    finding = doctor.check_python_version(version)
    assert finding.level == doctor.FAIL
    assert "3.10" in finding.detail


def test_the_real_interpreter_running_the_tests_is_supported():
    assert doctor.check_python_version().level == doctor.OK, \
        f"the test suite is running on unsupported Python {sys.version_info[:2]}"


# --- package groups ---------------------------------------------------------

def test_nothing_missing_reads_as_ok():
    finding = doctor.packages_finding(
        "core", doctor.CORE_PACKAGES, installed(*doctor.CORE_PACKAGES), "requirements.txt")
    assert finding.level == doctor.OK
    assert str(len(doctor.CORE_PACKAGES)) in finding.title


def test_missing_packages_are_named_with_the_reason_they_matter():
    finding = doctor.packages_finding(
        "core", doctor.CORE_PACKAGES, installed("fastapi", "requests"), "requirements.txt")

    assert finding.level == doctor.FAIL
    assert finding.fix == "pip install -r requirements.txt"
    # the pip name, not the module name -- "cv2" is not something you can install
    assert "opencv-python-headless" in finding.detail
    assert "rembg" in finding.detail
    assert "the rotoscope stage" in finding.detail, "each entry should say what breaks"
    assert "fastapi" not in finding.detail, "installed packages must not be listed as missing"


def test_optional_groups_warn_rather_than_fail():
    """A missing Runway SDK must not report the install as broken -- the
    default ComfyUI backend doesn't use it."""
    finding = doctor.packages_finding(
        "backends", doctor.BACKEND_PACKAGES, installed(), "requirements-backends.txt",
        level_if_missing=doctor.WARN)
    assert finding.level == doctor.WARN


def test_missing_packages_returns_pip_names_not_module_names():
    absent = doctor.missing_packages({"cv2": ("opencv-python-headless", "why")}, installed())
    assert absent == [("cv2", "opencv-python-headless", "why")]


def test_core_group_covers_everything_web_server_imports():
    """Guard against the group drifting from reality: every third-party
    module the photo booth imports at module scope should be listed, or the
    doctor will cheerfully report a green install that then fails to boot."""
    for module in ("fastapi", "uvicorn", "websockets", "requests", "PIL", "numpy", "cv2"):
        assert module in doctor.CORE_PACKAGES, f"{module} missing from CORE_PACKAGES"


# --- opencv build -----------------------------------------------------------

# Captured verbatim from real installs: opencv-python-headless 5.0.0.93 and
# opencv-python 5.0.0 on Windows. Both wheels export `imshow`, so an earlier
# hasattr()-based check called the headless build a GUI build -- these strings
# are here so that assumption can't come back.
HEADLESS_BUILD_INFO = """
General configuration for OpenCV 5.0.0 =====================================
  Version control:               5.0.0

  GUI:                           NONE
    VTK support:                 NO
"""

GUI_BUILD_INFO = """
General configuration for OpenCV 5.0.0 =====================================
  Version control:               5.0.0

  GUI:                           WIN32UI
    Win32 UI:                    YES
"""

# Some OpenCV builds leave the GUI: line bare and list backends beneath it.
BARE_GUI_BUILD_INFO = """
  GUI:
    GTK+:                        YES (ver 3.24.33)
    VTK support:                 NO
"""


class FakeCv2:
    """Mimics the real wheels: `imshow` exists either way; only the build
    report distinguishes them."""

    def __init__(self, gui):
        self.imshow = lambda *a: None  # present in BOTH builds -- that's the trap
        self._info = GUI_BUILD_INFO if gui else HEADLESS_BUILD_INFO

    def getBuildInformation(self):
        return self._info


def test_gui_detection_reads_the_build_report_not_the_symbol_table():
    """The regression this was rewritten for. Both wheels export imshow; only
    the headless one has no window backend."""
    assert doctor.opencv_has_gui(GUI_BUILD_INFO) is True
    assert doctor.opencv_has_gui(HEADLESS_BUILD_INFO) is False
    assert hasattr(FakeCv2(gui=False), "imshow"), "the trap must still be present in the fake"


def test_gui_detection_handles_a_bare_gui_line_with_backends_below():
    assert doctor.opencv_has_gui(BARE_GUI_BUILD_INFO) is True


def test_gui_detection_defaults_to_no_window_when_it_cannot_tell():
    """Better to under-promise: claiming a window backend that isn't there
    sends someone to debug the viewer instead of their install."""
    assert doctor.opencv_has_gui("") is False
    assert doctor.opencv_has_gui("some unrelated text") is False


def test_unreadable_build_info_warns_instead_of_crashing():
    class Broken:
        def getBuildInformation(self):
            raise RuntimeError("segfault-adjacent nonsense")

    finding = doctor.check_opencv_build(Broken(), wants_viewer=True)
    assert finding.level == doctor.WARN


def test_headless_opencv_is_fine_for_the_photo_booth():
    """The whole point of dropping the force-reinstall step: headless is the
    correct build unless you specifically want the Spout viewer's window."""
    finding = doctor.check_opencv_build(FakeCv2(gui=False), wants_viewer=False)
    assert finding.level == doctor.OK
    assert "headless" in finding.title


def test_headless_opencv_only_warns_when_the_viewer_is_wanted():
    finding = doctor.check_opencv_build(FakeCv2(gui=False), wants_viewer=True)
    assert finding.level == doctor.WARN
    assert finding.fix == "pip install --force-reinstall opencv-python"


def test_gui_opencv_passes_either_way():
    for wants_viewer in (True, False):
        assert doctor.check_opencv_build(FakeCv2(gui=True), wants_viewer).level == doctor.OK


def test_absent_opencv_is_left_to_the_package_check():
    """Reporting it here too produced two failures for one cause."""
    assert doctor.check_opencv_build(None, wants_viewer=False) is None


# --- models -----------------------------------------------------------------

WANTED = {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors",
          "control_net_name": "diffusers_xl_depth_full.safetensors"}


def test_all_models_present():
    available = {"ckpt_name": [WANTED["ckpt_name"], "other.safetensors"],
                 "control_net_name": [WANTED["control_net_name"]]}
    assert doctor.check_models(available, WANTED).level == doctor.OK


def test_missing_model_is_a_hard_failure_naming_the_file():
    available = {"ckpt_name": ["somethingelse.safetensors"],
                 "control_net_name": [WANTED["control_net_name"]]}
    finding = doctor.check_models(available, WANTED)

    assert finding.level == doctor.FAIL
    assert WANTED["ckpt_name"] in finding.detail
    assert WANTED["control_net_name"] not in finding.detail, "the present one shouldn't be listed"


def test_comfyui_reporting_no_models_at_all_fails_cleanly():
    finding = doctor.check_models({"ckpt_name": [], "control_net_name": []}, WANTED)
    assert finding.level == doctor.FAIL


def test_a_workflow_with_no_controlnet_does_not_demand_one():
    """txt2img has no ControlNet node; a None entry must not be reported as a
    missing file called 'None'."""
    finding = doctor.check_models({"ckpt_name": ["sdxl.safetensors"]},
                                  {"ckpt_name": "sdxl.safetensors", "control_net_name": None})
    assert finding.level == doctor.OK


# --- workflow reading -------------------------------------------------------

def test_workflow_model_names_reads_the_real_shipped_workflow():
    wanted = doctor.workflow_model_names()
    assert wanted["ckpt_name"], "should have found a CheckpointLoaderSimple node"
    assert wanted["control_net_name"], "should have found a ControlNetLoader node"
    assert wanted["ckpt_name"].endswith(".safetensors")


def test_workflow_model_names_survives_a_missing_file(monkeypatch, tmp_path):
    """doctor.py has to run on a broken checkout too."""
    monkeypatch.setattr(doctor, "PHOTOSHOOT_WORKFLOW", tmp_path / "nope.json")
    assert doctor.workflow_model_names() == {"ckpt_name": None, "control_net_name": None}


# --- output -----------------------------------------------------------------

def test_report_exit_code_reflects_only_hard_failures(capsys):
    assert doctor.report([doctor.Finding(doctor.OK, "fine")]) == 0
    assert doctor.report([doctor.Finding(doctor.WARN, "optional thing")]) == 0
    assert doctor.report([doctor.Finding(doctor.FAIL, "broken")]) == 1
    capsys.readouterr()


def test_ascii_only_output_survives_a_localised_os_error():
    """OS error strings arrive in the system locale. One of those must not be
    able to break the file's ASCII guarantee -- or, on a cp1252 console, raise
    UnicodeEncodeError while reporting somebody's broken install."""
    localised = "Nenhuma conexao pode ser feita — máquina recusou"
    cleaned = doctor.ascii_only(localised)
    assert cleaned.isascii()
    cleaned.encode("cp1252")  # must not raise


def test_doctor_source_is_pure_ascii():
    """Enforces the promise made in the module docstring."""
    source = (doctor.REPO_ROOT / "doctor.py").read_text(encoding="utf-8")
    non_ascii = sorted({c for c in source if ord(c) > 127})
    assert not non_ascii, f"doctor.py must stay ASCII-only, found: {non_ascii}"


# --- workflow roles ---------------------------------------------------------

def test_a_resolvable_workflow_passes_and_names_the_seed_field():
    class Resolved:
        roles = {"a": "1", "b": "2"}
        seed_field = "noise_seed"

    finding = doctor.check_workflow_roles(lambda: Resolved(), "video.json")
    assert finding.level == doctor.OK
    assert "2 roles" in finding.title
    assert "noise_seed" in finding.detail


def test_an_unresolvable_workflow_is_a_hard_failure_with_the_fix_command():
    """This one genuinely stops generation, so it must not be a warning."""
    def boom():
        raise ValueError("no ControlNetApplyAdvanced node")

    finding = doctor.check_workflow_roles(boom, "photoshoot_bg_api.json")
    assert finding.level == doctor.FAIL
    assert "no ControlNetApplyAdvanced" in finding.detail
    assert "workflow_graph.py" in finding.fix


def test_workflow_role_failure_survives_a_localised_os_error():
    def boom():
        raise OSError("Nao foi possivel encontrar o arquivo — caminho invalido")

    finding = doctor.check_workflow_roles(boom, "x.json")
    assert finding.detail.isascii()


def test_the_real_shipped_workflow_resolves():
    """Runs the actual resolver against the actual file, which is the only
    version of this check that can catch a bad re-export."""
    import workflow_graph
    finding = doctor.check_workflow_roles(
        lambda: workflow_graph.load(doctor.PHOTOSHOOT_WORKFLOW),
        doctor.PHOTOSHOOT_WORKFLOW.name)
    assert finding.level == doctor.OK


# --- brand kits -------------------------------------------------------------

class FakeBrand:
    def __init__(self, brand_id, looks=2, logo=True, seed_policy="locked"):
        self.id = brand_id
        self.looks = tuple(range(looks))
        self.logo = object() if logo else None
        self.seed_policy = seed_policy


def test_no_brand_kits_installed_is_fine():
    """Running without kits is the app's original behaviour, not a fault."""
    finding = doctor.check_brand_kits({}, expected_dirs=0)
    assert finding.level == doctor.OK
    assert "Optional" in finding.detail


def test_loaded_kits_are_summarised():
    finding = doctor.check_brand_kits({"acme": FakeBrand("acme", looks=3)}, expected_dirs=1)
    assert finding.level == doctor.OK
    assert "acme: 3 looks" in finding.detail


def test_a_skipped_pack_is_a_hard_failure_naming_the_shortfall():
    """The silent case this check exists for: load_brands() drops a malformed
    pack rather than raising, so the only symptom is a client missing from the
    dropdown -- which nobody notices until they go looking for it."""
    finding = doctor.check_brand_kits({"acme": FakeBrand("acme")}, expected_dirs=3)
    assert finding.level == doctor.FAIL
    assert "1 of 3" in finding.title
    assert "acme" in finding.detail
    assert finding.fix


def test_doctor_imports_only_the_standard_library():
    """It has to run on a bare interpreter, before pip has done anything --
    that's the entire reason it's useful for install problems."""
    source = (doctor.REPO_ROOT / "doctor.py").read_text(encoding="utf-8")
    third_party = {"requests", "fastapi", "numpy", "PIL", "cv2", "websockets", "uvicorn"}
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not stripped.startswith("#"):
            root = stripped.split()[1].split(".")[0]
            assert root not in third_party, f"doctor.py must not import {root}: {stripped!r}"
