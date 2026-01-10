# Cell 1: Imports
import torch
from pathlib import Path
from src.nn.sudoku_evaluator_uncertainty import SudokuEvaluator

# Set up paths
CHECKPOINT_PATH = "./train/runs/2026-01-06_16-21-25/checkpoints/last.ckpt"
DATA_DIR = "./data/sudoku_6x6_large"

def main():
    evaluator = SudokuEvaluator(
        checkpoint_path=CHECKPOINT_PATH,
        data_dir=DATA_DIR,
        batch_size=256,
        device="auto",
        num_workers=0,
        eval_split="val"
    )

    print(f"Model loaded successfully!")
    print(f"Grid size: {evaluator.grid_size}x{evaluator.grid_size}")
    print(f"Vocab size: {evaluator.vocab_size}")

    results_viz = evaluator.visualize_sample(
        split="val",
        sample_idx=3,
        show_confidence=True,
        save_gif=False,
        num_stochastic_runs=3,      
        dropout_enabled=True,       
    )

if __name__ == '__main__':
    main()