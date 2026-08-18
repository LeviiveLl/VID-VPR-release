# Checkpoints

`teacher.pth` is the final VLM-conditioned teacher. It consumes an image, a cached VLM hidden-state sequence, and its valid-token mask.

`student.pth` is the final deployment model. It consumes images only and produces independently indexable, L2-normalized 8448-D descriptors.

Each checkpoint has this format:

```python
{
    "format_version": 1,
    "architecture": str,
    "model_config": dict,
    "model_state_dict": dict[str, Tensor],
}
```

Verify downloads with `sha256sum -c checkpoints/SHA256SUMS`.
