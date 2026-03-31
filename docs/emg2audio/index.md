# Silent Speech Synthesis

The transduction model converts EMG signals into audio features (MFCCs), which are then reconstructed into waveforms using a HiFi-GAN vocoder.

## Core API

### Main Script
The `transduction_model.py` script handles training, validation, and evaluation using ASR models to compute WER on generated audio.

::: transduction_model
    handler: python
    options:
      group_by_category: true
      show_root_heading: false
      show_root_toc_entry: false

### Vocoder Utilities
Handles the conversion from MFCC features back to audio waveforms.

::: vocoder
    handler: python
    options:
      group_by_category: true
      show_root_heading: true
      show_root_toc_entry: true
