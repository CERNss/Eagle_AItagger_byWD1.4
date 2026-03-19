# AGENTS.md — Eagle AI Tagger System Spec

## Project Overview

WD14 (WaifuDiffusion 1.4) model-based image auto-tagging tool for Eagle asset manager. Performs GPU-accelerated ONNX inference to generate tags (Chinese/English) for images, writing results into Eagle's `metadata.json` per-image structure.

**Target platform**: Linux + NVIDIA GPU + Docker (migrating from Windows desktop usage)
**Runtime**: Python 3.8+, ONNX Runtime GPU, CUDA 12.x

---

## Architecture

### Entry Points

| File | Role | Notes |
|---|---|---|
| `main.py` | CLI entry point | argparse: `--config`, `--image_list` |
| `run.bat` | Windows launcher | **TO BE REMOVED** — replace with Dockerfile CMD or `run.sh` |
| `uptags.py` | Standalone tool | Batch-updates existing Eagle tags (English -> Chinese) |

### Core Modules (`main/`)

```
main.py
  -> main/mainp.py          # Orchestrator: loads config, dispatches work, collects results
      -> unified_config.py   # Config system (INI -> dataclasses)
      -> task_dispatcher.py  # Batch creation + adaptive batch sizing
      -> process_pool_manager.py  # Multiprocessing pool lifecycle
          -> manager.py      # Worker process: loads model, processes batches
              -> tagger.py   # ONNX inference + tag processing
              -> image_utils.py  # PIL/OpenCV image preprocessing
      -> result_collector.py # Aggregates results, writes JSON, generates reports
      -> progress_monitor.py # Real-time progress display (background thread)
      -> check_update.py     # GitHub version checker
```

### Data Flow

```
image_list.txt (file paths)
  -> mainp.py: parse paths -> [{image_path, json_path}, ...]
  -> task_dispatcher.py: split into batches
  -> process_pool_manager.py: distribute via mp.Queue
  -> manager.py (N worker processes, each loads own ONNX model):
      -> tagger.py: ONNX inference -> raw tags
      -> tagger.py: filter by threshold, exclude/include, sort
  <- result_queue: collect batch results
  -> result_collector.py: merge results, write to Eagle metadata.json files
  -> progress_monitor.py: display progress bar (background thread)
```

### Multiprocessing Model

- `ProcessPoolManager` spawns N `worker_process` instances (N = `config.process.max_workers`)
- Each worker independently loads the ONNX model into GPU memory
- Communication: `mp.Queue` for task dispatch and result collection
- Worker health monitoring: timeout detection, automatic restart of dead workers
- **Important for Linux**: default `start_method` is `fork`; may need `spawn` if CUDA context issues arise

### ONNX Inference (`tagger.py`)

- Provider chain: `['CUDAExecutionProvider', 'CPUExecutionProvider']` — auto-fallback already implemented
- Model input: preprocessed image tensor `(1, H, W, 3)` float32
- Model output: confidence array aligned with tag CSV
- Tags split: first 4 = ratings, rest = content tags
- Supports Chinese (`right_tag_cn`) and English (`name`) tag columns

### Config System (`unified_config.py`)

INI file parsed into typed dataclasses:

- `VersionConfig`: version, update_notes
- `ModelConfig`: model_path (Path), tags_path (Path)
- `TagConfig`: threshold, replace_underscore, escape_tags, use_chinese_name, additional/exclude tags, sort order
- `ProcessConfig`: max_workers, batch_size, max_retries, checkpoint_interval, add_write_mode
- `ReportConfig`: create_csv_report

Config is serialized to dict via `to_dict()` for passing through multiprocessing queues.

---

## Dependencies

### Core (required)

| Package | Purpose |
|---|---|
| `onnxruntime-gpu` | ONNX model inference with CUDA |
| `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc. | CUDA runtime libraries |
| `opencv-python` | Image I/O and preprocessing |
| `pillow` | Image loading and format handling |
| `numpy` | Array operations |
| `pandas` | Tag CSV loading and mapping |
| `requests` | Version check (GitHub) |
| `packaging` | Version comparison |
| `psutil` | System monitoring |
| `pynvml` | NVIDIA GPU monitoring |

### Suspicious / Likely Unnecessary

| Package | Issue |
|---|---|
| `torch` | **No `import torch` anywhere in codebase**. ~2GB. Likely vestigial. Verify and remove. |
| `xformers` | **No `import xformers` anywhere in codebase**. Likely vestigial. Verify and remove. |
| `pyreadline3` | **Windows-only**. Must remove for Linux. |
| `tzdata` | Windows needs this for timezone; Linux has system tzdata. Can remove. |

### requirements.txt Encoding Issue

File is **UTF-16LE with BOM** (not UTF-8). Will cause `pip install -r` failures in many Linux/Docker environments. Must re-encode to UTF-8.

---

## Eagle Integration

Eagle is a desktop asset management app. Its library structure:

```
<library>.library/
  backup/
  images/
    <HASH>.info/
      <filename>.png       # The image
      metadata.json        # Tags written here
  actions.json
  metadata.json
  mtime.json
  saved-filters.json
  tags.json
```

This tool reads image paths from `image_list.txt`, runs inference, and writes tags back to each image's `metadata.json`.

The `uptags.py` tool separately scans an entire Eagle library to batch-update English tags to Chinese using the CSV mapping table.

---

## Migration Status: Windows -> Linux

### Already Cross-Platform (no changes needed)

- All `main/` module path handling uses `pathlib.Path`
- ONNX provider fallback chain
- Multiprocessing architecture (standard `mp`)
- Config system (`configparser` + dataclasses)
- Image processing pipeline (PIL + OpenCV + numpy)
- JSON/CSV I/O

### Must Fix (blocking for Linux/Docker)

| Item | File | Line(s) | Issue |
|---|---|---|---|
| `input()` blocks | `mainp.py` | 44, 105 | Hangs in headless container (no tty) |
| `input()` blocks | `check_update.py` | 60 | Hangs in headless container |
| requirements.txt encoding | root | — | UTF-16LE, must convert to UTF-8 |
| `pyreadline3` dependency | requirements.txt | — | Windows-only, crashes on Linux |

### Should Fix (functional issues on Linux)

| Item | File | Line(s) | Issue |
|---|---|---|---|
| Hardcoded `\` path | `uptags.py` | 151 | `r"csv\Tags-cn_2024_ver-1.0.csv"` — not a valid Linux path |
| `input()` in uptags | `uptags.py` | 39, 188 | Interactive prompts, problematic if containerized |
| Version check writes temp file | `check_update.py` | 31-40 | Writes `temp_remote_config.ini` to CWD, fragile in containers |

### Should Remove

| Item | Reason |
|---|---|
| `run.bat` | Windows-only launcher |
| `old-version/` directory | Unused legacy code with Windows-specific patterns |
| `torch`, `xformers` in requirements | Not imported, ~2GB bloat |

### Recommended Additions (for Docker/service mode)

- `run.sh` or Dockerfile `CMD ["python", "main.py"]`
- `logging` module instead of `print()` (for structured log collection)
- Explicit `mp.set_start_method('spawn')` if CUDA fork issues arise
- Configurable output paths for reports (currently hardcoded filenames)
- Health check endpoint if exposing as HTTP/gRPC service later

---

## File Tree

```
.
+-- main.py                  # CLI entry point
+-- config.ini               # Runtime configuration
+-- image_list.txt           # Input: image paths (one per line)
+-- requirements.txt         # Python dependencies (NEEDS UTF-8 re-encode)
+-- run.bat                  # Windows launcher (TO REMOVE)
+-- uptags.py                # Standalone tag migration tool
+-- AGENTS.md                # This file
+-- README.md                # User documentation (NEEDS Linux rewrite)
+-- csv/
|   +-- Tags-cn_2024_ver-1.0.csv   # Tag dictionary (English + Chinese)
|   +-- ...archived dicts
+-- model/
|   +-- (*.onnx files go here)
+-- main/
|   +-- __init__.py
|   +-- mainp.py             # Orchestrator
|   +-- unified_config.py    # Config dataclasses
|   +-- tagger.py            # ONNX inference engine
|   +-- image_utils.py       # Image preprocessing
|   +-- manager.py           # Worker process logic
|   +-- process_pool_manager.py  # Process pool lifecycle
|   +-- task_dispatcher.py   # Batch creation/sizing
|   +-- result_collector.py  # Result aggregation + JSON writing
|   +-- progress_monitor.py  # Progress display thread
|   +-- check_update.py      # Version checker
+-- old-version/             # Legacy code (TO REMOVE)
```
