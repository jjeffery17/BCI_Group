# SSVEP experiment toolkit

This folder contains three working parts for running the SSVEP session and one support module:

- `idun_viewer.py` — connects to the IDUN Guardian device, shows live EEG/quality/impedance information, and writes a timestamped EEG CSV file.
- `ssvep_experiment.py` — runs the PsychoPy SSVEP task, logs experiment events, and uses the stimulus definitions from `blinking_stimuli.py`.
- `blinking_stimuli.py` — the stimulus engine used by the experiment. It is mostly a support module now; see the code comments and class docstrings for details.
- `README.md` — this file.

## Experiment order

Run the experiment in this order:

1. Start `idun_viewer.py` and wait for the EEG signal to stabilise.
2. Run `ssvep_experiment.py`.
3. Close `idun_viewer.py` after the experiment finishes.

The viewer and the experiment are designed to be used together in that sequence so you can confirm the signal quality before stimulus presentation begins.

## Files in this folder

### `idun_viewer.py`

This script opens a live EEG viewer for the IDUN Guardian device. It:

- connects using `IDUN_API_TOKEN` or the token set in the script,
- measures impedance during startup,
- shows live raw and filtered EEG traces,
- displays battery, impedance, quality, HEOG, and jaw-clench status,
- streams EEG samples to a CSV file named like `eeg_recording_<timestamp>.csv`.

The viewer waits for the device connection before opening the graph window, and it closes the CSV cleanly when recording ends.

### `ssvep_experiment.py`

This is the main experiment script. The current setup is:

- 4 rounds, one per target location,
- 5 repeats per round,
- 5 seconds of stimulation per repeat,
- a short fixed break between repeats,
- a brief rest before the experiment begins.

The current target layout is:

- top left: 6 Hz
- top right: 8 Hz
- bottom left: 11 Hz
- bottom right: 15 Hz

The script uses PsychoPy for the task window and event timing, and writes an event log to a CSV file named like `ssvep_events_<datetime>.csv`.

### `blinking_stimuli.py`

This module provides the stimulus classes and helper functions used by the SSVEP task. It supports the flicker scheduling, image preparation, and layout helpers used by `ssvep_experiment.py`. It is included as a dependency module rather than a standalone step in the experimental workflow.

## Setup

Install the required Python packages:

```bash
pip install psychopy pygame numpy matplotlib idun-guardian-sdk
```

Make sure these files are in the same directory:

- `ssvep_experiment.py`
- `idun_viewer.py`
- `blinking_stimuli.py`
- your stimulus images, if you are using custom ones

If you are using the current default experiment configuration, the image paths in `ssvep_experiment.py` should also exist exactly as written there.

## Running the experiment

1. Open a terminal in this folder.
2. Run:

```bash
python idun_viewer.py
```

3. Wait until the EEG signal looks stable and the viewer has finished its initial connection and impedance checks.
4. In a second terminal, run:

```bash
python ssvep_experiment.py
```

5. Complete the SSVEP task in PsychoPy.
6. Close `idun_viewer.py` when the experiment is done.

## Output files

You should expect two CSV files during a session:

- `eeg_recording_<timestamp>.csv` from `idun_viewer.py`
- `ssvep_events_<datetime>.csv` from `ssvep_experiment.py`

## Troubleshooting

If `idun_viewer.py` does not close normally, use Task Manager to close it if it is stuck in the connection stage.

Do not force-kill the viewer while it is writing the CSV unless that is absolutely necessary.

If the experiment does not start, check that PsychoPy is installed and that `blinking_stimuli.py` is in the same folder as `ssvep_experiment.py`.
