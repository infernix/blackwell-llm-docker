# Scale codec coding-agent handoff

Give the coding agent `IMPLEMENTATION_PROMPT.md` as its instruction. The other files are evidence and reference code, not a new implementation.

- `target-scales-g256.json`: original user-provided full offline audit report.
- `audit_facts.json`: compact derived totals and proposed-format arithmetic, with report SHA-256.
- `checkpoint_audit.py`: original audit source, unchanged; it is not the requested new codec.

The actual checkpoint and serving source trees are not included. The agent needs local read access to the checkpoint to encode real scale bytes, and a suitable GPU/software environment for GPU tests. Synthetic CPU tests can run without either. Keep source checkpoint files read-only and create sidecars outside that directory.

The requested first milestone is a standalone Python CLI plus an optional GPU module. Native vLLM/B12X integration is a separate measured gate. This bundle contains a build specification, not an implemented or benchmarked GPU codec.
