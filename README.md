# yt2guide

Turns a YouTube video into a step-by-step guide, entirely on this machine:

1. `yt-dlp` downloads the video (max 1080p) → `video.mp4`
2. `faster-whisper` (`large-v3-turbo`, GPU) transcribes it → `transcript.srt`, `transcript.txt`
3. `ffmpeg` samples the video twice a second; a frame is kept whenever the screen settles after a change of ≥ 0.5 % of pixels against the last kept frame (≥ 1 s apart, no cap; one per minute when only small changes happen) → `frames/NNNN_MM-SS_<sec>.jpg`
4. `gemma4:e4b` (Ollama) describes each frame, one at a time, with the transcript around it → `frames.md`
5. `qwen3.6:35b-a3b` (Ollama) writes the guide from the transcript and frame descriptions → `guide.md`

The guide adapts to the video: a tutorial gets key takeaways + numbered steps with menu paths;
a podcast or talk gets key ideas, topics with times, advice, quotes and claims to check.

Every step is skipped when its output already exists, so an interrupted run resumes
and a new goal only reruns step 5 (delete `guide.md` first).

## Usage

```bash
yt2guide "https://www.youtube.com/watch?v=..."      # the model picks what matters itself
yt2guide URL -g "how to set this up for our company: ..."   # optional: tailor the guide
yt2guide URL -o myvideo --goal-file goal.txt      # longer goal in a file
yt2guide URL --audio-only                         # podcast/interview: audio only, no screenshots (faster)
yt2guide URL --no-guide                           # only transcript + frames + descriptions
yt2guide URL --think                              # text model thinks first: better, slower
yt2guide URL --change 0.02 --max-frames 60   # fewer frames: bigger change needed, optional cap
```

`guide.md` is written in Czech by default (`--out-lang`). Run `yt2guide -h` for all options.

## Requirements

- Linux with an NVIDIA GPU
- [uv](https://docs.astral.sh/uv/), `ffmpeg`, Node.js (yt-dlp needs a JS runtime for YouTube)
- [Ollama](https://ollama.com); the two models (~3 GB and ~23 GB) are pulled on the first run

## Install

```bash
git clone git@github.com:rudajura/yt2guide.git ~/yt2guide
~/yt2guide/install.sh      # creates .venv (Python 3.12) and links ~/bin/yt2guide
```

The Whisper model (~1.6 GB) downloads on the first transcription.

## Verification

Unit tests cover the pure functions (timestamps, frame selection, transcript window, media
lookup) and run without a GPU, network, Ollama or any models.

```bash
uv run --no-project --python 3.12 --with-requirements requirements-dev.txt pytest -q
```

## How it fits together

- `.venv` holds `faster-whisper`, `yt-dlp`, `yt-dlp-ejs` and the CUDA 12 cuDNN/cuBLAS wheels
  (`requirements.txt`). The `yt2guide` wrapper puts those libraries on `LD_LIBRARY_PATH`.
- Ollama: the script starts it when it is not running (`systemctl start ollama`, else `ollama serve`)
  and pulls a missing model on the first run. Before transcribing it asks Ollama to unload every
  model from the GPU, and it unloads each of its own models when done, so Whisper, the vision model
  and the text model never compete for VRAM.
- Update yt-dlp when YouTube downloads break: `uv pip install --python .venv/bin/python -U yt-dlp yt-dlp-ejs`.
