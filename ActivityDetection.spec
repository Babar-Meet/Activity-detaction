# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('models', 'models'), ('yolov8n.pt', '.')]
binaries = []
hiddenimports = ['ultralytics', 'mediapipe', 'mediapipe.tasks', 'mediapipe.tasks.c', 'mediapipe.tasks.python', 'mediapipe.python.solutions.face_mesh', 'mediapipe.python.solutions.face_mesh_connections']
datas += collect_data_files('mediapipe')
binaries += collect_dynamic_libs('mediapipe')
hiddenimports += collect_submodules('mediapipe.tasks')
hiddenimports += collect_submodules('mediapipe.python.solutions')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ActivityDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ActivityDetection',
)
