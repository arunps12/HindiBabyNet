# External Models

This directory holds externally cloned model repositories that run in their own virtual environments.

## VTC 2.0

Clone VTC here:

```bash
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git external_models/VTC
cd external_models/VTC && uv sync
```

See the main [README](../README.md#vtc-20-backend-optional) for full setup instructions.
