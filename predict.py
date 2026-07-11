from __future__ import annotations

import polars as pl
import torch
from tqdm import tqdm

import json2vec as j2v
from json2vec.structs.enums import Strata, TensorKey, Tokens

CHECKPOINT = ".gcp_dump/fbc/goq7neec/checkpoints/epoch=61-step=15686.ckpt"
BATCH_SIZE = 8

evaluation_records = pl.read_ndjson("data/arc_problem_sets_evaluation.jsonl")

@j2v.preprocess
def pair_examples(record: dict) -> j2v.Observation:
    problem_set = record["problem_set"]
    return j2v.Observation({
        "examples": [
            {"grids": [{"pixels": inp["pixels"]}, {"pixels": out["pixels"]}]}
            for inp, out in zip(problem_set["inputs"], problem_set["outputs"])
        ],
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = j2v.Model.load_from_checkpoint(CHECKPOINT, map_location=device)
model.eval()
model.to(device)

pixel_address = next(iter(model.schema.target))
examples_length, grids_length, pixels_length = model.schema.shapes[pixel_address]
print(
    f"pixel address: {pixel_address} "
    f"(examples={examples_length}, grids={grids_length}, pixels={pixels_length})"
)

records = evaluation_records.to_dicts()
print(f"records: {len(records)}")

total_pixels_correct = 0
total_pixels_predicted = 0
total_grids_correct = 0
total_grids = 0

for target_example_idx in range(examples_length):
    eligible = [
        r for r in records
        if len(r["problem_set"]["inputs"]) > target_example_idx
    ]
    if not eligible:
        continue

    for start in tqdm(
        range(0, len(eligible), BATCH_SIZE),
        desc=f"example slot {target_example_idx} ({len(eligible)} records)",
    ):
        batch = eligible[start:start + BATCH_SIZE]

        # Encode with mask=False so we can selectively hide only the pixels
        # of the chosen output grid — everything else stays observed.
        inputs = model.encode(
            batch=batch,
            preprocess=pair_examples,
            strata=Strata.test,
            mask=False,
        )
        inputs = inputs.to(device)
        cell = inputs[pixel_address]

        selected = torch.zeros_like(cell.state, dtype=torch.bool)
        selected[:, target_example_idx, 1, :] = True
        selected = selected & cell.state.eq(Tokens.valued.value)
        if not selected.any():
            continue

        cell.hide(selected, cache_targets=True, trainable=True)

        with torch.inference_mode():
            predictions = model.forward(inputs, strata=Strata.test)

        pixel_prediction = next(p for p in predictions if p.address == pixel_address)
        state_pred = pixel_prediction.payload[TensorKey.state].argmax(dim=-1)
        content_pred = pixel_prediction.payload[TensorKey.content].argmax(dim=-1)

        true_state = cell.targets[TensorKey.state]
        true_content = cell.targets[TensorKey.content]

        grid = (slice(None), target_example_idx, 1, slice(None))
        pixel_correct = (
            state_pred[grid].eq(true_state[grid])
            & content_pred[grid].eq(true_content[grid])
        )  # shape: (batch, 900)

        total_pixels_correct += int(pixel_correct.sum().item())
        total_pixels_predicted += pixel_correct.numel()

        grid_correct = pixel_correct.all(dim=-1)  # shape: (batch,)
        total_grids_correct += int(grid_correct.sum().item())
        total_grids += grid_correct.numel()

print()
print(f"grids evaluated:  {total_grids}")
print(
    f"grids fully correct (all {pixels_length} pixels): "
    f"{total_grids_correct} / {total_grids} "
    f"= {total_grids_correct / max(total_grids, 1):.4%}"
)
print(f"pixels evaluated: {total_pixels_predicted}")
print(
    f"pixels correct:   {total_pixels_correct} / {total_pixels_predicted} "
    f"= {total_pixels_correct / max(total_pixels_predicted, 1):.4%}"
)
