# Data Directory

Per-subject data goes here. Each subject directory contains:

```
data/{subject}/
├── transformsVHAP/              # VHAP face tracker output
│   ├── canonical_flame_param.npz
│   ├── flame_param/
│   │   ├── 000000.npz
│   │   └── ...
│   ├── transforms_train.json
│   ├── transforms_val.json
│   └── transforms_test.json
├── view_000/
│   ├── images/                  # Extracted and cropped frames
│   └── masks/                   # Background segmentation masks
├── audio_full.wav               # Full audio from training video (16kHz mono)
├── voice_reference.wav          # Best 30-60s segment for TTS voice cloning
└── persona.json                 # Executive persona definition
```
