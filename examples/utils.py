def visualize_sudoku_text(batch, idx=0, grid_size=6):
    input_tensor = batch['input'][idx]
    label_tensor = batch['output'][idx]
    
    # 1. We need to determine the max_grid_size used by the dataset to reshape correctly.
    # We can infer it from the tensor length: sqrt(sequence_length)
    seq_len = input_tensor.numel()
    max_grid_size = int(seq_len**0.5)

    # Box dimensions
    if grid_size == 6: box_rows, box_cols = 2, 3
    elif grid_size == 9: box_rows, box_cols = 3, 3
    elif grid_size == 4: box_rows, box_cols = 2, 2
    else: box_rows, box_cols = 2, 3

    def decode_cell(val):
        val = val.item()
        if val == 2: return "."
        if val > 2: return str(val - 2)
        return "?" # 0=PAD or 1=EOS

    def render_grid(tensor):
        full_grid = tensor.reshape(max_grid_size, max_grid_size)
        grid = full_grid[:grid_size, :grid_size]
        
        lines = []
        dash_segment = "-" * (box_cols * 2 + 1)
        h_sep = "+" + "+".join([dash_segment] * (grid_size // box_cols)) + "+"
        
        for r in range(grid_size):
            if r % box_rows == 0:
                lines.append(h_sep)
            
            row_str = "|"
            for c in range(grid_size):
                cell = decode_cell(grid[r, c])
                row_str += f" {cell}"
                if (c + 1) % box_cols == 0:
                    row_str += " |"
            lines.append(row_str)
        lines.append(h_sep)
        return "\n".join(lines)

    # Stats calculation (only on valid crop)
    full_input_2d = input_tensor.reshape(max_grid_size, max_grid_size)
    valid_input_crop = full_input_2d[:grid_size, :grid_size]
    
    givens = (valid_input_crop > 2).sum().item()
    empty = (valid_input_crop == 2).sum().item()

    print("=" * 60)
    print(f"Sample {idx} (grid_size={grid_size}x{grid_size})")
    print("=" * 60)
    print(f"Givens: {givens}, Empty: {empty}\n")
    
    print("Puzzle:")
    print(render_grid(input_tensor))
    print("\nSolution:")
    print(render_grid(label_tensor))
    print("\n")