import argparse
import subprocess

def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NX-414 - Predicting Neural Activity")
    parser.add_argument("mode", choices=["linear", "task-driven", "data-driven", "best"],
                        help="Choose the model")
    
    args = parser.parse_args()
    
    script_map = {
        "linear": "linear_models.py",
        "task-driven": "task_driven_models.py",
        "data_driven": "data_driven_models.py",
        "best": "best_model.py",
    }
    
    script_to_run = script_map.get(args.mode)
    if script_to_run:
        run_script(script_to_run)
    else:
        print("Invalid mode selected.")