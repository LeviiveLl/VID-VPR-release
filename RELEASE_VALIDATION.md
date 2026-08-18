# Release validation

Validated on 2026-08-17 with Python 3.10 and PyTorch 2.10.

## Checkpoints

- Both public checkpoints contain a complete model state and public architecture configuration.
- Teacher strict load: 454 tensors.
- Student strict load: 495 tensors.
- No optimizer, scheduler, absolute training path, or intermediate-checkpoint dependency is stored.
- Checksums are recorded in `checkpoints/SHA256SUMS`.

## Forward checks

- Teacher random-input smoke test: output shape `[1, 4096]`, L2 norm `0.99999994`.
- Student random-input smoke test: output shape `[1, 8448]`, L2 norm `1.0000`.
- Original-final versus public-student parity on the same input:
  - maximum absolute error: `0.0`;
  - mean absolute error: `0.0`;
  - cosine similarity: `1.00000024`.

## Code checks

- Python compilation passed for the package, scripts, and tests.
- All four command-line entry modules expose a working `--help` command.
- Unit tests: `2 passed`.
- Source paths, source text, filenames, and checkpoint metadata were scanned for names of external VPR methods; no matches remain.

GPU execution was unavailable in the final isolated packaging environment. The numerical parity check was therefore run on CPU.
