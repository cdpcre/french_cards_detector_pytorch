import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_step(command, step_name):
    print(f"\n{'='*20} Starting {step_name} {'='*20}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"{'='*20} {step_name} Completed Successfully {'='*20}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during {step_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="End-to-End YOLOv11 Training Pipeline")
    parser.add_argument('--skip-ingestion', action='store_true', help="Skip data ingestion step")
    parser.add_argument('--skip-verification', action='store_true', help="Skip data verification step")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--model', type=str, default='yolo11n.pt', help="Base model")
    
    args = parser.parse_args()

    # 1. Data Ingestion
    if not args.skip_ingestion:
        # unify_datasets.py is in data_ingestion/
        ingestion_script = Path("data_ingestion/unify_datasets.py")
        if not ingestion_script.exists():
            print(f"Error: {ingestion_script} not found.")
            sys.exit(1)
        run_step(f"python {ingestion_script}", "Data Ingestion")
    else:
        print("Skipping Data Ingestion...")

    # 2. Data Verification
    if not args.skip_verification:
        verify_script = Path("verify_ingestion.py")
        if not verify_script.exists():
            print(f"Error: {verify_script} not found.")
            sys.exit(1)
        run_step(f"python {verify_script}", "Data Verification")
    else:
        print("Skipping Data Verification...")

    # 3. Training
    train_cmd = (
        f"python train.py "
        f"--epochs {args.epochs} "
        f"--batch {args.batch} "
        f"--model {args.model} "
        f"--project runs/train "
        f"--data datasets/unified/data.yaml "
        f"--lr 5e-4 "
        f"--mosaic 1.0 "
        f"--workers 0"
    )
    run_step(train_cmd, "Training")

    print("\nPipeline execution finished successfully!")
    print("You can now run the evaluation notebook to inspect the results.")

if __name__ == "__main__":
    main()
