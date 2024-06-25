# -*- mode: python ; coding: utf-8 -*-
#block_cipher = None

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('ASLalphabet.jpg', '.'),
        ('model/keypoint_classifier/control_classifier.py', 'model/keypoint_classifier'),
        ('model/keypoint_classifier/hand_sign_classifier.py', 'model/keypoint_classifier'),
        ('model/customModel/ASLclassifier_label.csv', 'model/customModel'),
        ('model/customModel/control_label.csv', 'model/customModel'),
        ('model/customModel/controlModel/control_classifierctrl.tflite', 'model/customModel/controlModel'),
        ('model/customModel/modelB/ASLclassifier4a.tflite', 'model/customModel/modelB'),
        ('model/__init__.py', 'model'),
        ('utils/cvfpscalc.py', 'utils'),
        ('utils/__init__.py', 'utils'),
        ('C:\\Users\\ivono\\AppData\\Roaming\\Python\\Python310\\site-packages\\mediapipe\\modules', 'mediapipe\\modules'),
    ],
    hiddenimports=[],
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
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
