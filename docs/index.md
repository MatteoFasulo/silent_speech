# Voicing Silent Speech

This repository contains code for synthesizing speech audio from silently mouthed words captured with electromyography (EMG). This is an updated fork supporting the **TinyMyo** EMG foundation model.

## Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Download & Prepare Data**:
   ```bash
   # Download data from Zenodo
   python download_data.py
   # Build HDF5 for fast loading
   python build_hdf5.py
   ```

3. **Train Transduction (EMG to Audio)**:
   ```bash
   python transduction_model.py
   ```

4. **Train Recognition (EMG to Text)**:
   ```bash
   # Setup KenLM and Lexicon
   python get_lexicon.py
   # Start training
   python recognition_model.py
   ```

## Sections

<div class="grid cards" markdown>

- __Silent Speech Synthesis__

	---
    Transduction and automated speech recognition models for silent speech synthesis. From EMG to audio features (MFCCs).
	[Open Silent Speech Synthesis docs](emg2audio/index.md)

- __Silent Speech Recognition__

    ---
    Transcription and decoding of silent speech signals directly to text using CTC.
    [Open Silent Speech Recognition docs](emg2text/index.md)

</div>
