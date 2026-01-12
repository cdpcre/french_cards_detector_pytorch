import subprocess
import os
import sys

def test_scripts_help():
    """Smoke test to verify scripts can be invoked and imports are correct."""
    scripts = [
        "scripts/train_fast.py",
        "scripts/train_custom.py",
        "scripts/evaluate_model.py",
        # "scripts/predict.py" # Predict has a server that might not exit nicely with help or might try to bind port.
        # Although argparse usually captures --help. Let's check if it uses argparse. 
        # predict.py uses flask, it might not have argparse or --help might be flask's.
    ]
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    for script in scripts:
        print(f"Testing {script} --help")
        cmd = [sys.executable, script, "--help"]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        assert result.returncode == 0, f"{script} failed execution with --help: {result.stderr}"

def test_predict_import():
     """Test that predict.py can be imported without error."""
     # We can't easily run it as it starts a server. But checking import is good enough for structure.
     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
     env = os.environ.copy()
     env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
     
     cmd = [sys.executable, "-c", "import scripts.predict; print('Import success')"]
     result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, env=env)
     assert result.returncode == 0, f"scripts/predict.py import failed: {result.stderr}"

if __name__ == "__main__":
    test_scripts_help()
    test_predict_import()
    print("All structure tests passed!")
