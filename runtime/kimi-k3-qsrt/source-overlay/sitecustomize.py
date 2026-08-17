"""Run a source-locked vLLM tree with extensions from a binary image.

Set ``VLLM_SOURCE_OVERLAY_ACTIVE=1``, ``VLLM_SOURCE_OVERLAY_ROOT`` to the
repository root, and ``VLLM_BINARY_PACKAGE_DIR`` to the image's compiled
``vllm`` package. Kimi serving launchers enable the overlay explicitly so
unrelated Python processes do not import vLLM during interpreter startup.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path


def _activate_vllm_source_overlay() -> None:
    source_root = os.environ.get("VLLM_SOURCE_OVERLAY_ROOT")
    binary_package_dir = os.environ.get("VLLM_BINARY_PACKAGE_DIR")
    if not source_root:
        raise RuntimeError("VLLM_SOURCE_OVERLAY_ROOT is not configured")

    source_root = str(Path(source_root).resolve())
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)

    if binary_package_dir:
        binary_package_dir = str(Path(binary_package_dir).resolve())

        class _VLLMBinaryFallback(importlib.abc.MetaPathFinder):
            """Resolve compiled vLLM submodules from the binary package."""

            def find_spec(self, fullname, path=None, target=None):
                if not fullname.startswith("vllm."):
                    return None
                relative_parent = fullname.split(".")[1:-1]
                search_dir = str(Path(binary_package_dir, *relative_parent))
                spec = importlib.machinery.PathFinder.find_spec(
                    fullname, [search_dir]
                )
                if spec is None or spec.origin is None:
                    return None
                if fullname == "vllm.vllm_flash_attn.cute" or fullname.startswith(
                    "vllm.vllm_flash_attn.cute."
                ):
                    return spec
                if not any(
                    spec.origin.endswith(suffix)
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES
                ):
                    return None
                return spec

        sys.meta_path.insert(0, _VLLMBinaryFallback())

    vllm = importlib.import_module("vllm")
    source_package_dir = str(Path(source_root, "vllm").resolve())
    loaded_package_dir = str(Path(vllm.__file__).resolve().parent)
    if loaded_package_dir != source_package_dir:
        raise RuntimeError(
            "vLLM source overlay resolved the wrong package: "
            f"expected={source_package_dir}, loaded={loaded_package_dir}"
        )

    if binary_package_dir:
        if binary_package_dir not in vllm.__path__:
            vllm.__path__.append(binary_package_dir)

        # The source repository contains the public FlashAttention wrapper;
        # the binary package supplies generated FA4 CuTeDSL modules. Extending
        # the subpackage preserves source ownership for Python code while
        # loading generated modules from the ABI-compatible image.
        flash_attention = importlib.import_module("vllm.vllm_flash_attn")
        binary_flash_attention_dir = str(
            Path(binary_package_dir, "vllm_flash_attn").resolve()
        )
        if binary_flash_attention_dir not in flash_attention.__path__:
            flash_attention.__path__.append(binary_flash_attention_dir)


if os.environ.get("VLLM_SOURCE_OVERLAY_ACTIVE") == "1":
    try:
        _activate_vllm_source_overlay()
    except Exception as exc:
        # site.execsitecustomize catches ordinary exceptions and lets Python
        # continue. SystemExit preserves the source-identity invariant by
        # terminating startup before an unintended vLLM package can execute.
        raise SystemExit(f"vLLM source overlay activation failed: {exc}") from exc
