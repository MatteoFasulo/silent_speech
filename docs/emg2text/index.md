# Silent Speech Recognition

The recognition model performs direct EMG-to-text transcription using CTC loss and a KenLM-backed beam search decoder.

## Core API

### Main Script
The `recognition_model.py` script manages the training loop and Word Error Rate (WER) evaluation using character-based CTC.

::: recognition_model
    handler: python
    options:
      group_by_category: true
      show_root_heading: false
      show_root_toc_entry: false

### Lexicon & LM Setup
Utilities for preparing the decoding environment.

::: get_lexicon
    handler: python
    options:
      group_by_category: true
      show_root_heading: true
      show_root_toc_entry: true
