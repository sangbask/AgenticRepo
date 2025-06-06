import subprocess

def test_streamlit_runs():
    """Smoke test: Checks that the Streamlit app launches without crashing."""
    result = subprocess.run(
        ["streamlit", "run", "FraudDetectorAIAgents.py", "--headless", "--server.port=8502"],
        capture_output=True,
        text=True,
        timeout=10
    )
    assert "RuntimeError" not in result.stderr
    assert "Traceback" not in result.stderr
