# -*- mode: python ; coding: utf-8 -*-
# Builds the onedir backend bundled into the Electron installer's
# extraResources. onedir (not onefile) so the packaged app starts fast --
# onefile re-extracts its payload to a temp dir on every launch.
#
#   pyinstaller sam3_backend.spec
#
# torch/transformers do importlib.metadata lookups at import time that fail
# silently when frozen unless their dist-info metadata is copied in alongside
# collect_all's binaries/datas -- both are required, collect_all alone isn't
# enough.
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

for pkg in ("torch", "torchvision", "transformers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in (
    "torch", "torchvision", "tqdm", "regex", "requests", "packaging", "filelock",
    "numpy", "tokenizers", "importlib_metadata", "huggingface_hub",
    "safetensors", "pyyaml", "fastapi", "starlette", "uvicorn", "pydantic",
    "pillow", "pycocotools",
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# export_core.py loads this by file path at runtime (it isn't a valid module
# name), so PyInstaller's import-graph analysis can't discover it on its
# own -- bundle it explicitly, landing at sys._MEIPASS/3_export.py.
datas += [("../3_export.py", ".")]

a = Analysis(
    ["server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sam3-backend",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="sam3-backend",
)
