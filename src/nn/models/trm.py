"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.modules.utils import compute_lr
from src.nn.utils.constants import IGNORE_LABEL_ID
from src.nn.optimizers.muon import Muon

try:
    from adam_atan2 import AdamATan2
    print(f"*"*60)
    print("Imported AdamATan2 successfully")
except ImportError:
    print("Failed to import adam2")

from lightning import LightningModule

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.trm_block import (
    CastedEmbedding,
    CastedLinear,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    RotaryEmbedding,
    RotaryEmbedding2D,
)
from src.nn.modules.utils import stablemax_cross_entropy, trunc_normal_init_
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor  # High-level state (y = the solution representation)
    z_L: torch.Tensor  # Low-level state (z = the problem representation)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""

    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]  # Stores current batch data


class TRMModule(LightningModule):
    """
    HRM implementation following Figure 2 pseudocode exactly.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,  # min(2, hidden_size // 64)
        max_grid_size: int = 30,
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        N_supervision_val: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        halt_exploration_prob: float = 0.1,
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        rope_theta: int = 10000,
        pos_emb_type: str = "1d",
        use_mlp_t: bool = False,
        use_conv_swiglu: bool = False,
        use_board_swiglu: bool = False,
        lr_min_ratio: float = 1.0,
        use_muon: bool = False,
        vocab_size: int = 0,  # Should be set from datamodule
        num_puzzles: int = 0,  # Should be set from datamodule
        batch_size: int = 0,  # Should be set from datamodule
        pad_value: int = -1,  # Should be set from datamodule
        seq_len: int = 0,  # Should be set from datamodule
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization
        self.automatic_optimization = False

        self.forward_dtype = torch.bfloat16

        # Token embeddings
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        log.info(f"Creating TRM with vocab size={vocab_size}, seq_len={seq_len}, puzzle_emb_len={puzzle_emb_len} {pos_emb_type=} {puzzle_emb_dim=}")
        log.info(f"{use_mlp_t=}, {use_conv_swiglu=}, {use_board_swiglu=}")

         # Input embedding

        self.input_embedding = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        if pos_emb_type == "2d":
            log.info("Using 2D Rotary Embeddings")
            self.pos_embedding = RotaryEmbedding2D(
            dim=hidden_size // num_heads,
            prefix_len=puzzle_emb_len,
            max_grid_size=int(math.sqrt(seq_len)),  # e.g., 9 for seq_len=81
            base=rope_theta,
        )
        elif pos_emb_type == "1d":
            log.info("Using 1D Rotary Embeddings")
            self.pos_embedding = RotaryEmbedding(
                dim=hidden_size // num_heads,
                max_position_embeddings=seq_len + puzzle_emb_len,
                base=rope_theta,
            )
        else:
            log.info("Not using Rotary Embeddings")

        if not use_mlp_t:
            assert pos_emb_type is not None, "Rotary embeddings required if using attention"

        # a single network (not two separate networks)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=ffn_expansion,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
            mlp_t=use_mlp_t,
            puzzle_emb_ndim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
            use_conv_swiglu=use_conv_swiglu,
            use_board_swiglu=use_board_swiglu,
            rows = max_grid_size,
            cols = max_grid_size,
            dropout=0.25
        )

        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(num_layers)]
        )

        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q_head = CastedLinear(hidden_size, 1, bias=True) # learn to stop, not to continue

        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)  # Strong negative bias

        # State for carry (persisted across training steps)
        self.carry = None

        self.z_H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.z_L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Add puzzle embeddings
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,  # Reference uses 0 init
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = puzzle_emb_len
            log.info(f"Created puzzle_emb with num_puzzles={num_puzzles}, batch_size={batch_size}")
            log.info(f"puzzle_emb.local_weights.shape: {self.puzzle_emb.local_weights.shape}")
            log.info(f"puzzle_emb.weights.shape: {self.puzzle_emb.weights.shape}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

        self.manual_step = 0

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        if stage == "fit":
            # Calculate steps from dataset and epochs
            dm = self.trainer.datamodule
        
            # Use num_groups for steps calculation (not total puzzles)
            if hasattr(dm, 'num_train_groups'):
                samples_per_epoch = dm.num_train_groups
            else:
                samples_per_epoch = len(dm.train_dataset)
            
            steps_per_epoch = samples_per_epoch // dm.batch_size
            
            if self.trainer.max_epochs > 0:
                self.total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                self.total_steps = float("inf")

            log.info("Training configuration:")
            log.info(f"  Groups (unique puzzles): {getattr(dm, 'num_train_groups', 'N/A')}")
            log.info(f"  Total puzzles (with aug): {len(dm.train_dataset)}")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Total steps: {self.total_steps}")

                # Add torch.compile for faster training
            if "DISABLE_COMPILE" not in os.environ and hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    log.info("Compiling inner_forward with torch.compile...")
                    self.inner_forward = torch.compile(
                        self.inner_forward,
                        mode="reduce-overhead",  # Good for repeated calls (your H/L cycles)
                        fullgraph=False,         # Allow graph breaks for dynamic control flow
                    )
                    log.info("Compilation successful")
                except Exception as e:
                    log.warning(f"torch.compile failed, running uncompiled: {e}")
            else:
                log.info('*' * 60)
                log.info("torch.compile not available or disabled, running uncompiled")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.input_embedding(input.to(torch.int32))

        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Scale
        return self.embed_scale * embedding

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["input"].shape[0]
        device = batch["input"].device

        return TRMCarry(
            inner_carry=self.empty_carry(
                batch_size, device
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Default to halted
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )

    def empty_carry(self, batch_size: int, device: torch.device) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
        )

    def inner_forward(
        self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )
        #print(f"[inner forward] seq_info: {seq_info}")
        # Input encoding
        input_embeddings = self._input_embeddings(batch["input"], batch["puzzle_identifiers"])
        #print(f"[inner forward] batch[\"input\"].shape: {batch["input"].shape}")
        #print(f"[inner forward] input_embeddings shape: {input_embeddings.shape}")
        #print(f"[inner forward] input_embeddings[0,0]: {input_embeddings[0,0,:]}")
        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        # 1 with grad
        for _ in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)
    
        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :] # discard puzzle embeddings
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Q-head; uses the first puzzle_emb position
        #print(f"[inner forward] z_H shape: {z_H.shape}")
        #print(f"[inner forward] z_L shape: {z_L.shape}")
        #print(f"[inner forward] q_logits shape: {q_logits.shape}")
        #print(f"[inner forward] output shape: {output.shape}")
        return new_carry, output, q_logits[..., 0]

    def _visualize_inner_forward_debug(
        self,
        batch: Dict[str, torch.Tensor],
        output_H: torch.Tensor,
        output_L: torch.Tensor,
        q_logits: torch.Tensor,
        sample_idx: int,
    ):
        """
        Visualize z_H and z_L evolution for a specific sample as Sudoku grids.
        
        Args:
            batch: Current batch
            sample_idx: Which sample to track (0 by default)
            z_H: Current z_H state tensor
            z_L_history: List of z_L states at each cycle
            cycle_log: List of cycle names for labeling
            final_output: Final predictions from lm_head
            final_q_logit: Final halting logit
        """
        import numpy as np
        
        # Only log every N steps to avoid spam
        #log_interval = 100
        #if self.manual_step % log_interval != 0:
        #    return
        
        inputs = batch["input"]
        targets = batch["output"]
        grid_size = 6
        
        # Extract grid info (reshape using max_grid_size)
        inp = inputs[sample_idx].reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        tgt = targets[sample_idx].reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        
        inp_grid = inp[:grid_size, :grid_size]
        tgt_grid = tgt[:grid_size, :grid_size]

        # Compute confidence and prediction from out_H=lm_head(z_H) and out_L=lm_head(z_L)
        sample_logits_H = output_H[sample_idx]  # [seq_len, vocab]
        probs_H = torch.softmax(sample_logits_H, dim=-1)
        confidence_H, preds_H = probs_H.max(dim=-1)
        
        sample_logits_L = output_L[sample_idx]  # [seq_len, vocab]
        probs_L = torch.softmax(sample_logits_L, dim=-1)
        confidence_L, preds_L = probs_L.max(dim=-1)

        # Reshape to grids
        pred_H = preds_H.reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        conf_H = confidence_H.reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        pred_L = preds_L.reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        conf_L = confidence_L.reshape(self.hparams.max_grid_size, self.hparams.max_grid_size)
        pred_H_grid = pred_H[:grid_size, :grid_size].clone()
        conf_H_grid = conf_H[:grid_size, :grid_size].clone()
        pred_L_grid = pred_L[:grid_size, :grid_size].clone()
        conf_L_grid = conf_L[:grid_size, :grid_size].clone()
        
        def decode_cell(val, is_input=False):
            val = val.item() if hasattr(val, 'item') else val
            if val == 0:
                return "."  # PAD
            elif val == 1:
                return "X"  # EOS
            elif val == 2:
                return "_" if is_input else "?"  # Empty
            else:
                return str(val - 2)  # Actual value (1-9)
        
        def format_cell(val, conf=None, prev_val=None, target_val=None, is_empty=False):
            """Format a cell with optional markers."""
            cell_str = decode_cell(val)
            
            # Markers: * = changed, ! = wrong
            prefix = " "
            if prev_val is not None and val != prev_val:
                prefix = "*"  # Changed from previous step
            elif target_val is not None and val != target_val and is_empty:
                prefix = "!"  # Wrong vs target (only for cells that need prediction)
            
            return prefix + cell_str
        
        def get_box_dims(grid_size: int) -> tuple:
            """Get box dimensions for a grid size."""
            if grid_size == 4:
                return 2, 2
            elif grid_size == 6:
                return 2, 3
            elif grid_size == 9:
                return 3, 3
            else:
                raise ValueError(f"Unsupported grid_size: {grid_size}")
        
        def grid_to_lines(grid, conf_grid=None, prev_grid=None, target=None, input_mask=None):
            """Convert grid to list of formatted lines."""
            lines = []
            box_rows, box_cols = get_box_dims(grid_size)
            
            for r in range(grid_size):
                if r > 0 and r % box_rows == 0:
                    # Horizontal separator
                    sep_parts = []
                    for s in range(grid_size // box_cols):
                        sep_parts.append("-" * (box_cols * 2))
                    lines.append("+".join(sep_parts))
                
                row_str = ""
                for c in range(grid_size):
                    if c > 0 and c % box_cols == 0:
                        row_str += "|"
                    
                    is_empty = input_mask[r, c].item() if input_mask is not None else False
                    prev_val = prev_grid[r, c] if prev_grid is not None else None
                    target_val = target[r, c] if target is not None else None
                    
                    row_str += format_cell(grid[r, c], None, prev_val, target_val, is_empty)
                
                lines.append(row_str)
            
            return lines
        
        def compute_metrics(pred, target, mask):
            """Compute accuracy metrics."""
            correct = (pred == target) & mask
            total = mask.sum().item()
            if total == 0:
                return 1.0, 0
            return correct.sum().item() / total, (mask & ~correct).sum().item()

        def create_grid_frame(
            grid: torch.Tensor,
            target: torch.Tensor,
            prev_grid: torch.Tensor,
            input_mask: torch.Tensor,
            size: int,
            cell_size: int,
            offset: int,
            box_rows: int,
            box_cols: int,
            font,
            label_font,
            step: int,
            total_steps: int,
            is_input: bool = False,
        ):
            """Create a single frame for the GIF."""
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (size, size), 'white')
            draw = ImageDraw.Draw(img)
            
            def decode_cell(val):
                val = val.item() if hasattr(val, 'item') else val
                if val == 0 or val == 1 or val == 2:
                    return ""
                else:
                    return str(val - 2)
            
            # Draw cells
            for r in range(grid_size):
                for c in range(grid_size):
                    x = offset + c * cell_size
                    y = offset + r * cell_size
                    
                    cell_val = grid[r, c]
                    is_empty_cell = input_mask[r, c].item()
                    
                    # Fill cell background (always white)
                    draw.rectangle(
                        [x, y, x + cell_size, y + cell_size],
                        fill='white',
                        outline=None
                    )
                    
                    # Draw cell value
                    val_str = decode_cell(cell_val)
                    if val_str:
                        # Determine text color
                        if not is_empty_cell:
                            # Given cell - gray
                            text_color = '#666666'
                        elif not is_input: # and grid[r, c] != prev_grid[r, c]:
                            # New prediction this step - check if correct
                            if grid[r, c] == target[r, c]:
                                text_color = '#228B22'  # Forest green - correct
                            else:
                                text_color = '#FF8C00'  # Dark orange - incorrect
                        else:
                            # Existing prediction or input - black
                            text_color = 'black'
                        
                        # Center text in cell
                        bbox = draw.textbbox((0, 0), val_str, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        text_x = x + (cell_size - text_w) // 2
                        text_y = y + (cell_size - text_h) // 2 - bbox[1]
                        
                        draw.text((text_x, text_y), val_str, fill=text_color, font=font)
        
            # Draw grid lines
            grid_size_px = cell_size * grid_size
            
            # Thin lines for all cells
            for i in range(grid_size + 1):
                # Horizontal
                y = offset + i * cell_size
                draw.line([(offset, y), (offset + grid_size_px, y)], fill='black', width=1)
                # Vertical
                x = offset + i * cell_size
                draw.line([(x, offset), (x, offset + grid_size_px)], fill='black', width=1)
            
            # Thick lines for boxes
            for i in range(grid_size // box_rows + 1):
                y = offset + i * box_rows * cell_size
                draw.line([(offset, y), (offset + grid_size_px, y)], fill='black', width=3)
            for i in range(grid_size // box_cols + 1):
                x = offset + i * box_cols * cell_size
                draw.line([(x, offset), (x, offset + grid_size_px)], fill='black', width=3)
            
            # Draw step label
            if is_input:
                label = "Input"
            else:
                label = f"Step {step}/{total_steps}"
            
            bbox = draw.textbbox((0, 0), label, font=label_font)
            label_w = bbox[2] - bbox[0]
            label_x = size - label_w - 10
            label_y = 10
            draw.text((label_x, label_y), label, fill='black', font=label_font)
            
            return img

        # 1. Prepare the lines for all four grids
        input_mask = (inp_grid == 2)  # Empty cells to predict
        input_lines = grid_to_lines(inp_grid, input_mask=input_mask) 
        target_lines = grid_to_lines(tgt_grid, input_mask=input_mask)
        grid_H_lines = grid_to_lines(pred_H_grid, target=tgt_grid, input_mask=input_mask)
        grid_L_lines = grid_to_lines(pred_L_grid, target=tgt_grid, input_mask=input_mask)
        
        # 2. Compute metrics for the headers
        acc_H, err_H = compute_metrics(pred_H_grid, tgt_grid, input_mask)
        acc_L, err_L = compute_metrics(pred_L_grid, tgt_grid, input_mask)

        # 3. Define formatting width (grid_size * 2 + 2 for markers and separators)
        col_w = 16 

        # 4. Construct Headers
        header_str = (
            f"{'INPUT':<{col_w}} | "
            f"{'TARGET':<{col_w}} | "
            f"{'z_H (Acc: ' + f'{acc_H:.1%}' + ')':<{col_w}} | "
            f"{'z_L (Acc: ' + f'{acc_L:.1%}' + ')':<{col_w}}"
        )

        print(f"\n" + "="*len(header_str))
        print(f"STEP: {self.manual_step} | SAMPLE: {sample_idx}")
        print(f"MARKERS: '!' = Wrong Prediction, '*' = Change Detected")
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))

        # 5. Zip and Print Rows
        for l_in, l_tgt, l_h, l_l in zip(input_lines, target_lines, grid_H_lines, grid_L_lines):
            print(f"{l_in:<{col_w}} | {l_tgt:<{col_w}} | {l_h:<{col_w}} | {l_l:<{col_w}}")

        # 6. Optional: Halting Info
        q_val = torch.sigmoid(q_logits[sample_idx]).item()
        print("-" * len(header_str))
        print(f"Halting Probability (q): {q_val:.4f} " + ("-> STOPPING" if q_val > 0.5 else "-> CONTINUING"))
        print("="*len(header_str) + "\n")

        # 7. Save grid
        size = 400
        padding = size // 10
        grid_area = size - 2 * padding
        cell_size = grid_area // grid_size
        grid_size_px = cell_size * grid_size
        offset = (size - grid_size_px) // 2
        box_rows, box_cols = get_box_dims(grid_size)
        label_font_size = size // 20
        font_name = "Arial.ttf"
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL not installed. Run: pip install Pillow")
            return
        try:
            # Try common fonts
            font_size = size // 12
            label_font_size = size // 20
            for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    label_font = ImageFont.truetype(font_name, label_font_size)
                    break
                except (OSError, IOError):
                    continue
            else:
                font = ImageFont.load_default()
                label_font = font
        except Exception:
            font = ImageFont.load_default()
            label_font = font

        img = create_grid_frame(
            grid=pred_H_grid,                # Prediction grid from z_H
            target=tgt_grid,                 # Target grid
            prev_grid=inp_grid,              # Previous grid (input)
            input_mask=(inp_grid == 2),      # Mask for empty cells
            size=size,
            cell_size=cell_size,
            offset=offset,
            box_rows=box_rows,
            box_cols=box_cols,
            font=font,
            label_font=label_font,
            step=self.manual_step,
            total_steps=self.hparams.N_supervision,
            is_input=False,
        )

        # Save the image
        img.save(f"z_H_grid_step{self.manual_step}_sample{sample_idx}.png")
        print(f"Saved z_H grid image: z_H_grid_step{self.manual_step}_sample{sample_idx}.png")

    def inner_forward_debug(
        self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["input"], batch["puzzle_identifiers"])
        
        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        
        # 1 with grad
        for _ in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)
        
        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output_H = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        # Also get output_L for visualization
        with torch.no_grad():
            output_L = self.lm_head(z_L)[:, self.puzzle_emb_len:]

        # Visualize the tracked sample
        if self.training and hasattr(self, 'manual_step') and self.current_epoch >= 499:
            sample_idx = 0
            self._visualize_inner_forward_debug(
                batch,
                output_H,
                output_L,
                q_logits,
                sample_idx,
            )
        
        return new_carry, output_H, q_logits[..., 0]

    def forward(
        self, carry: TRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, q_halt_logits = self.inner_forward_debug(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )

            is_last_step = new_steps >= n_supervision_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.hparams.N_supervision > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)

                 # Print halting decision for sample 0
                print(f"[Halting Decision] Step {self.manual_step} | Sample 0: halted = {halted[0].item()}")

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics without circular reference."""
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        labels = new_carry.current_data["output"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)

            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"].squeeze() >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Compute losses: These are per-sequence losses that will be summed
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        return new_carry, total_loss, metrics, new_carry.halted.all()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that implements supervision through multiple forward passes.
        Each sequence can run up to N_supervision (halt_max_steps) times.
        """
        #print(f"[training_step] batch_idx: {batch_idx}")
        batch_size = batch["input"].shape[0]

        # Handle case when not attached to trainer (for testing)
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            # For testing without trainer
            if not hasattr(self, "_optimizers"):
                raise RuntimeError("No optimizer available. Set model._optimizers for testing.")
            opts = self._optimizers

        # Initialize carry if first batch
        if self.carry is None:
            self.carry = self.initial_carry(batch)

        # Forward with loss computation
        self.carry, loss, metrics, _ = self.compute_loss_and_metrics(self.carry, batch)

        scaled_loss = loss / batch_size
        scaled_loss.backward()

        if batch_idx % 50 == 0:
            self.grad_monitoring()

        lr_this_step = None
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Learning rate scheduling with warmup
        current_step = self.manual_step

        # Base learning rates for each optimizer
        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:  # If we have puzzle embedding optimizer
            base_lrs.append(self.hparams.learning_rate_emb)

        # Compute learning rate for this step
        for opt, base_lr in zip(opts, base_lrs):
            if current_step < self.hparams.warmup_steps:
                lr_this_step = compute_lr(
                    base_lr=base_lr,
                    lr_warmup_steps=self.hparams.warmup_steps,
                    lr_min_ratio=self.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=self.total_steps,
                )
            else:
                # Constant LR after warmup
                lr_this_step = base_lr

            # Update learning rate
            if hasattr(opt, "_optimizer"):
                for param_group in opt._optimizer.param_groups:
                    param_group["lr"] = lr_this_step
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr_this_step
                opt.step()
                opt.zero_grad()

        self.log_metrics(metrics, lr_this_step=lr_this_step, batch_size=batch_size)

        # Assert LM loss is not NaN
        assert not torch.isnan(metrics.get("lm_loss")), f"LM loss is NaN at step {self.manual_step}"

        self.manual_step += 1

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Simplified validation using loss head."""
        batch_size = batch["input"].shape[0]

        with torch.no_grad():
            # Create fresh carry for validation
            carry = self.initial_carry(batch)

            # Accumulate metrics across all supervision steps
            accumulated_metrics = {}
            total_loss = 0.0
            n_steps = 0

            # Run up to N_supervision iterations
            while True:
                # Forward with loss computation
                carry, loss, metrics, all_halted = self.compute_loss_and_metrics(carry, batch)

                # Accumulate metrics
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()

                total_loss += loss.item()
                n_steps += 1

                if all_halted:
                    break

            # Compute averages
            count = accumulated_metrics.get("count", batch_size)
            if count > 0:
                avg_metrics = {
                    "val/loss": total_loss / (n_steps * batch_size),
                    "val/accuracy": accumulated_metrics.get("accuracy", 0) / count,
                    "val/exact_accuracy": accumulated_metrics.get("exact_accuracy", 0) / count,
                    "val/q_halt_accuracy": accumulated_metrics.get("q_halt_accuracy", 0) / count,
                    "val/steps": accumulated_metrics.get("steps", 0) / count,
                    "val/lm_loss": accumulated_metrics.get("lm_loss", 0) / (n_steps * batch_size),
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0)
                    / (n_steps * batch_size),
                }
            else:
                avg_metrics = {
                    f"val/{k}": 0.0
                    for k in [
                        "loss",
                        "accuracy",
                        "exact_accuracy",
                        "q_halt_accuracy",
                        "steps",
                        "lm_loss",
                        "q_halt_loss",
                    ]
                }

            # Log metrics
            for name, value in avg_metrics.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name in ["val/loss", "val/exact_accuracy"]),
                    sync_dist=True,
                )
            return avg_metrics

    def grad_monitoring(self):
        with torch.no_grad():
            # 1. Total gradient norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=float('inf')  # Don't actually clip, just compute norm
            ).item()
            
            # 2. Key component gradient norms
            grad_metrics = {}
            
            # First attention layer
            if hasattr(self.lenet.layers[0], 'self_attn') and \
                self.lenet.layers[0].self_attn.qkv_proj.weight.grad is not None:
                    grad_metrics['first_attn'] = self.lenet.layers[0].self_attn.qkv_proj.weight.grad.norm().item()
            
            # Last MLP layer
            if self.lenet.layers[-1].mlp.down_proj.weight.grad is not None:
                grad_metrics['last_mlp'] = self.lenet.layers[-1].mlp.down_proj.weight.grad.norm().item()
            
            # Output heads
            if self.lm_head.weight.grad is not None:
                grad_metrics['lm_head'] = self.lm_head.weight.grad.norm().item()
            
            if self.q_head.weight.grad is not None:
                grad_metrics['q_head'] = self.q_head.weight.grad.norm().item()
            
            # Log main metric
            self.log('grad/total_norm', total_grad_norm, on_step=True, prog_bar=True)
            
            # Log gradient flow ratio (first vs last layer)
            if 'first_attn' in grad_metrics and 'last_mlp' in grad_metrics:
                ratio = grad_metrics['first_attn'] / (grad_metrics['last_mlp'] + 1e-8)
                self.log('grad/flow_ratio', ratio, on_step=True, prog_bar=True)
            
            # Optional: log individual components
            for name, value in grad_metrics.items():
                self.log(f'grad/{name}', value, on_step=True)
            
            # Warning for problematic gradients
            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")

    def log_metrics(self, metrics: dict, lr_this_step: float = None, batch_size: int = None):

        # Log learning rate (will log the last optimizer's LR)
        self.log("train/lr", lr_this_step, on_step=True)

        # Log metrics
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/accuracy", metrics.get("accuracy", 0) / count, on_step=True)
                self.log(
                    "train/exact_accuracy",
                    metrics.get("exact_accuracy", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )
                self.log(
                    "train/q_halt_accuracy",
                    metrics.get("q_halt_accuracy", 0) / count,
                    on_step=True,
                )
                self.log(
                    "train/steps",
                    metrics.get("steps", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )

                self.log("train/lm_loss", metrics.get("lm_loss", 0) / batch_size, on_step=True)
                self.log(
                    "train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True
                )

                avg_halt_steps = metrics.get("steps", 0) / metrics["count"]
                early_halt_rate = avg_halt_steps < self.hparams.N_supervision
                self.log("train/early_halt_rate", early_halt_rate, on_step=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        """Don't interfere with training carry during validation."""
        pass

    def on_validation_epoch_end(self):
        """Don't interfere with training carry during validation."""
        pass

    def on_train_epoch_start(self):
        # Update sampler epoch for proper shuffling
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, 'on_train_epoch_start'):
                dm.on_train_epoch_start(self.current_epoch)
    
    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different parameter groups."""

        base_lr = self.hparams.learning_rate
        embedding_lr = self.hparams.learning_rate_emb

        if self.hparams.use_muon:
            adam_params = [p for p in self.parameters() if p.ndim != 2]
            muon_params = [p for p in self.parameters() if p.ndim == 2]

            print('*' * 60)
            print("Using Muon optimizer")
            for name, p in self.named_parameters():
                if p.ndim == 2:
                    print(f"Muon param: {name} | shape: {p.shape}")

            if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
                optimizers = [
                    CastedSparseEmbeddingSignSGD_Distributed(
                        self.puzzle_emb.buffers(),  # type: ignore
                        lr=embedding_lr,  # Needs to be set by scheduler
                        weight_decay=self.hparams.weight_decay,
                        world_size=1,
                    )]
            else:
                optimizers = []
                
            optimizers.append(
                Muon([
                    {
                        "params": muon_params,
                        "use_muon": True,
                        "lr": base_lr,
                    },
                    {
                        "params": adam_params,
                        "use_muon": False,
                        "lr": base_lr,
                        "weight_decay": 0.1,
                        "adamw_betas": (0.9, 0.95),
                        "adamw_eps": 1e-8,
                    },
                ])
            )
            
        else:
            optimizers = []
            # Main optimizer
            # Use AdamATan2 if available
            try:
                main_opt = AdamATan2(
                    self.parameters(),
                    lr=base_lr,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9, 0.95),
                )
            except NameError:
                main_opt = torch.optim.AdamW(
                    self.parameters(),
                    lr=base_lr,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9, 0.95),
                )
            optimizers.append(main_opt)

            # Force sparse embedding to be leaf tensors
            if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
                # Force sparse embedding local weights to be leaf tensors
                self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(
                    True
                )

                # Add sparse embedding optimizer
                sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                    self.puzzle_emb.buffers(),
                    lr=embedding_lr,
                    weight_decay=self.hparams.weight_decay,
                    world_size=1,
                )
                optimizers.append(sparse_opt)

        return optimizers
