import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "CSUF CS Advising System.py"


def load_module():
    spec = importlib.util.spec_from_file_location("csuf_advising", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_completed_courses():
    mod = load_module()
    parsed = mod.parse_completed_courses("CPSC 120A, cpsc 120l, math 150a")
    assert parsed == ["CPSC 120A", "CPSC 120L", "MATH 150A"]


def test_progress_summary_has_units_and_next_courses():
    mod = load_module()
    response = mod.build_fast_advising_response(
        "what can i take next",
        ["CPSC 120A", "CPSC 120L", "CPSC 121A", "CPSC 121L", "MATH 150A", "STATISTICS"],
    )
    assert "Progress estimate" in response
    assert "Likely eligible next courses" in response


def test_extract_completed_from_query():
    mod = load_module()
    query, completed = mod.extract_completed_from_query(
        "what can I take next? completed: CPSC 120A, CPSC 120L"
    )
    assert "completed" not in query.lower()
    assert completed == ["CPSC 120A", "CPSC 120L"]
