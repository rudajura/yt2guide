"""YouTube video -> transcript + key screenshots -> step-by-step guide, all local.

Steps (each one is skipped when its output already exists, so a run can be resumed):
  1. yt-dlp downloads the video            -> video.mp4 (audio.* with --audio-only)
  2. faster-whisper transcribes it         -> transcript.srt, transcript.txt
  3. ffmpeg + numpy pick a frame whenever the screen settles after a change -> frames/*.jpg
  4. a vision model describes each frame   -> frames.md
  5. a text model writes the guide         -> guide.md
"""

import argparse
import base64
import gc
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"
TIMINGS: list[tuple[str, float]] = []


def log(msg: str) -> None:
    print(f"[yt2guide] {msg}", file=sys.stderr, flush=True)


def step(name: str):
    def wrap(fn):
        def run(*a, **kw):
            t = time.monotonic()
            log(f"{name} ...")
            result = fn(*a, **kw)
            TIMINGS.append((name, time.monotonic() - t))
            return result
        return run
    return wrap


def ts(seconds: float, sep: str = ":") -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}{sep}{m:02d}{sep}{s:02d}" if h else f"{m:02d}{sep}{s:02d}"


def media(out: Path) -> Path | None:
    video = out / "video.mp4"
    if video.exists():
        return video
    return next((p for p in sorted(out.glob("audio.*")) if p.suffix not in (".json", ".part")), None)


@step("download")
def download(url: str, out: Path, audio_only: bool) -> None:
    have = media(out)
    if have and (audio_only or have.name == "video.mp4"):
        return
    fmt = ["-f", "ba/b", "-o", str(out / "audio.%(ext)s")] if audio_only else \
          ["-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4", "-o", str(out / "video.%(ext)s")]
    subprocess.run(
        ["yt-dlp", "--js-runtimes", "node", *fmt, "--write-info-json", "--no-playlist", url],
        check=True,
    )


@step("transcribe")
def transcribe(out: Path, model_name: str, language: str | None) -> list[dict]:
    seg_file = out / "transcript.json"
    if seg_file.exists():
        return json.loads(seg_file.read_text())
    free_vram()
    import numpy as np
    from faster_whisper import WhisperModel

    pcm = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(media(out)),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
    segments, info = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                      condition_on_previous_text=False)
    log(f"language: {info.language}, duration {ts(info.duration)}")
    segs = []
    for s in segments:
        text = s.text.strip()
        if segs and text == segs[-1]["text"]:
            segs[-1]["end"] = s.end
        else:
            segs.append({"start": s.start, "end": s.end, "text": text})
        print(f"\r  {ts(s.end)} / {ts(info.duration)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    del model
    gc.collect()

    def srt_time(x: float) -> str:
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    (out / "transcript.srt").write_text("".join(
        f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n"
        for i, s in enumerate(segs, 1)))
    (out / "transcript.txt").write_text(
        "".join(f"[{ts(s['start'])}] {s['text']}\n" for s in segs))
    seg_file.write_text(json.dumps(segs, ensure_ascii=False))
    return segs


SAMPLE_FPS, SAMPLE_W, SAMPLE_H = 2, 320, 180


def pick_times(samples, change: float, min_gap: float, max_gap: float, max_frames: int) -> list[float]:
    """Pick the moments worth a screenshot from small grayscale samples taken SAMPLE_FPS times a second.

    A frame is kept when it differs from the last KEPT frame in at least `change` of its pixels
    (so slow typing adds up) and the picture has settled (it barely differs from the previous
    sample, so a scroll gives its end state, not every step of it). Without a settled change,
    a frame is still kept after `max_gap` seconds if anything changed at all.
    """
    import numpy as np

    times: list[float] = []
    ref = prev = None
    ref_t = 0.0
    for i, frame in enumerate(samples):
        t = i / SAMPLE_FPS
        f = frame.astype(np.int16)
        if ref is None:
            times.append(t)
            ref, ref_t, prev = f, t, f
            continue
        vs_ref = float((np.abs(f - ref) > 24).mean())
        settled = float((np.abs(f - prev) > 24).mean()) < change / 5
        prev = f
        if t - ref_t < min_gap:
            continue
        if (vs_ref >= change and settled) or (t - ref_t >= max_gap and vs_ref >= change / 5):
            times.append(t)
            ref, ref_t = f, t
    if prev is not None and times[-1] != t and float((np.abs(prev - ref) > 24).mean()) >= change / 5:
        times.append(t)
    if max_frames and len(times) > max_frames:
        idx = [round(i * (len(times) - 1) / (max_frames - 1)) for i in range(max_frames)]
        times = [times[i] for i in sorted(set(idx))]
    return times


def video_samples(video: Path):
    import numpy as np

    size = SAMPLE_W * SAMPLE_H
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-an",
         "-vf", f"fps={SAMPLE_FPS},scale={SAMPLE_W}:{SAMPLE_H},format=gray",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    while (buf := proc.stdout.read(size)) and len(buf) == size:
        yield np.frombuffer(buf, np.uint8).reshape(SAMPLE_H, SAMPLE_W)
    proc.wait()


@step("frames")
def extract_frames(out: Path, change: float, max_frames: int, min_gap: float,
                   max_gap: float) -> list[tuple[float, Path]]:
    frames_dir = out / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        return sorted(
            (int(p.stem.split("_")[2]), p) for p in frames_dir.glob("*.jpg"))
    video = out / "video.mp4"
    times = pick_times(video_samples(video), change, min_gap, max_gap, max_frames)
    log(f"{len(times)} frames")
    frames_dir.mkdir(exist_ok=True)
    result = []
    for i, t in enumerate(times, 1):
        p = frames_dir / f"{i:04d}_{ts(t, '-')}_{int(t)}.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "2", str(p)],
            check=True)
        result.append((int(t), p))
    return result


def ollama_chat(model: str, messages: list[dict], num_ctx: int, think: bool, num_predict: int,
                stream_to=None) -> str:
    body = {"model": model, "messages": messages, "stream": True, "think": think,
            "options": {"num_ctx": num_ctx, "temperature": 0.2, "num_predict": num_predict}}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, thinking = [], 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            thinking += len(chunk.get("message", {}).get("thinking") or "")
            piece = chunk.get("message", {}).get("content", "")
            parts.append(piece)
            if stream_to:
                print(piece, end="", file=stream_to, flush=True)
    if thinking:
        log(f"  ({model} also thought {thinking} chars)")
    return "".join(parts)


def unload(model: str) -> None:
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


def free_vram() -> None:
    """Whisper needs the GPU: ask Ollama to drop whatever models it still keeps loaded."""
    try:
        loaded = json.load(urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=5))["models"]
    except OSError:
        return
    for m in loaded:
        log(f"unloading {m['name']} from the GPU")
        unload(m["name"])


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=3).read()
        return True
    except OSError:
        return False


def ensure_ollama(models: list[str]) -> None:
    if not ollama_up():
        log("Ollama is not running, starting it ...")
        if subprocess.run(["systemctl", "start", "--no-ask-password", "ollama"],
                          capture_output=True).returncode != 0:
            logfile = open(Path.home() / ".ollama-serve.log", "ab")
            subprocess.Popen(["ollama", "serve"], stdout=logfile, stderr=logfile,
                             start_new_session=True)
        for _ in range(30):
            if ollama_up():
                break
            time.sleep(1)
        else:
            sys.exit("Ollama did not start; run `ollama serve` and try again.")
    have = {m["name"] for m in json.load(urllib.request.urlopen(f"{OLLAMA}/api/tags"))["models"]}
    for model in models:
        if model in have or f"{model}:latest" in have:
            continue
        log(f"pulling {model} (first run only) ...")
        req = urllib.request.Request(f"{OLLAMA}/api/pull", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for line in resp:
                chunk = json.loads(line)
                if "error" in chunk:
                    sys.exit(f"pull of {model} failed: {chunk['error']}")
                if chunk.get("total"):
                    print(f"\r  {chunk.get('completed', 0) / 1e9:.1f} / {chunk['total'] / 1e9:.1f} GB",
                          end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)


def transcript_around(segs: list[dict], t: float, before: float = 40, after: float = 20) -> str:
    return " ".join(s["text"] for s in segs if s["end"] >= t - before and s["start"] <= t + after)


@step("describe frames")
def describe_frames(out: Path, frames: list[tuple[float, Path]], segs: list[dict], model: str) -> list[dict]:
    cache = out / "frames.json"
    done = {d["file"]: d for d in json.loads(cache.read_text())} if cache.exists() else {}
    for i, (t, p) in enumerate(frames, 1):
        if p.name in done:
            continue
        log(f"  frame {i}/{len(frames)} @ {ts(t)}")
        prompt = (
            "This is a screenshot from a video. What is said around this moment:\n"
            f"\"{transcript_around(segs, t)}\"\n\n"
            "Describe precisely what is on the screen: application/page, menu path, visible "
            "buttons, field names and their values, settings, code or commands. Copy all "
            "readable UI text verbatim. If it is just a person talking or a title card, say so "
            "in one sentence. No introduction, no speculation.")
        text = ollama_chat(model, [{"role": "user", "content": prompt,
                                    "images": [base64.b64encode(p.read_bytes()).decode()]}],
                           num_ctx=8192, think=False, num_predict=1500)
        done[p.name] = {"file": p.name, "t": t, "text": text.strip()}
        cache.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1))
    unload(model)
    descs = sorted(done.values(), key=lambda d: d["t"])
    (out / "frames.md").write_text("".join(
        f"## [{ts(d['t'])}] {d['file']}\n\n![](frames/{d['file']})\n\n{d['text']}\n\n" for d in descs))
    return descs


@step("write guide")
def write_guide(out: Path, segs: list[dict], descs: list[dict], model: str, goal: str,
                out_lang: str, think: bool) -> None:
    title = ""
    info = next(iter(sorted(out.glob("*.info.json"))), None)
    if info:
        title = json.loads(info.read_text()).get("title", "")
    events = [(s["start"], f"[{ts(s['start'])}] SAID: {s['text']}") for s in segs]
    events += [(d["t"] + 0.01, f"[{ts(d['t'])}] SCREEN ({d['file']}): {d['text']}") for d in descs]
    timeline = "\n".join(e for _, e in sorted(events))
    task = goal or ("Extract what matters most in this video: decide yourself which points are "
                    "important and which are filler, and turn the important ones into a guide "
                    "that someone can follow without watching the video.")
    prompt = (
        f"Below is the timeline of a video titled \"{title}\": the spoken transcript (SAID)"
        + (" interleaved with descriptions of screenshots (SCREEN)" if descs else
           "; there are no screenshots, only the audio was processed") + ".\n\n"
        f"TASK: {task}\n\n"
        f"Write the answer in Markdown, in language: {out_lang}. First decide what kind of video "
        "this is and use the matching structure.\n"
        "A) It shows how to do something on screen (tutorial, demo, screen share): 1) key "
        "takeaways: the 3-7 most important points, most important first, one or two sentences "
        "each, 2) prerequisites, 3) numbered steps with exact menu paths, field names, values and "
        "commands (keep UI labels in their original language), each step with the video time "
        "[mm:ss] and the screenshot file it relies on, 4) gotchas and things the narrator warns "
        "about, 5) open points: what the video does not show and has to be decided or checked.\n"
        "B) It is mainly people talking (podcast, interview, talk, lecture) and the picture adds "
        "little: 1) key takeaways: the 3-10 most important ideas, most important first, 2) the "
        "main topics in the order discussed, each with [mm:ss] and what was said about it, "
        "3) concrete advice, numbers, tools, books, people and companies mentioned, 4) notable "
        "quotes, verbatim with [mm:ss], 5) claims worth checking. Ignore screenshots that only "
        "show people talking; use the ones showing slides, charts or screens.\n"
        "Do not repeat these instructions and do not mention the letters A or B; start "
        "directly with the first section. Use only facts from the timeline; mark anything you "
        "infer as (assumption).\n\n"
        f"TIMELINE:\n{timeline}")
    num_ctx = min(131072, max(8192, int(len(prompt) / 3) + 6144))
    log(f"prompt ~{len(prompt) // 4} tokens, num_ctx {num_ctx}")
    guide = ollama_chat(model, [{"role": "user", "content": prompt}], num_ctx=num_ctx,
                        think=think, num_predict=8000, stream_to=sys.stderr)
    print(file=sys.stderr)
    (out / "guide.md").write_text(guide.strip() + "\n")
    unload(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="YouTube URL (or anything yt-dlp supports)")
    ap.add_argument("-o", "--out", help="output folder (default ./yt2guide-<video id>)")
    ap.add_argument("-g", "--goal", default="", help="optional: tailor the guide, e.g. how to set it up for our company "
                         "(default: the model picks the most important points itself)")
    ap.add_argument("--goal-file", help="read the goal from a file")
    ap.add_argument("--lang", help="spoken language for Whisper (default: autodetect)")
    ap.add_argument("--out-lang", default="Czech", help="language of guide.md (default Czech)")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--vl-model", default="gemma4:e4b")
    ap.add_argument("--text-model", default="qwen3.6:35b-a3b")
    ap.add_argument("--think", action="store_true", help="let the text model think first (better, much slower)")
    ap.add_argument("--max-frames", type=int, default=0, help="cap on frames (default 0 = no cap)")
    ap.add_argument("--max-gap", type=float, default=60,
                    help="take a frame at least this often (seconds) if anything on screen changed")
    ap.add_argument("--change", type=float, default=0.005,
                    help="share of pixels that must change for a new frame (default 0.005 = 0.5%%; higher = fewer frames)")
    ap.add_argument("--min-gap", type=float, default=1, help="minimum seconds between frames")
    ap.add_argument("--audio-only", action="store_true",
                    help="download only the audio and skip screenshots (podcasts, interviews, talks)")
    ap.add_argument("--no-guide", action="store_true", help="stop after frames.md (or transcript with --audio-only)")
    a = ap.parse_args()

    goal = Path(a.goal_file).read_text() if a.goal_file else a.goal
    vid = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", a.url)
    out = Path(a.out or f"yt2guide-{vid.group(1) if vid else int(time.time())}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    log(f"output: {out}")

    ensure_ollama(([] if a.audio_only else [a.vl_model]) + ([] if a.no_guide else [a.text_model]))
    download(a.url, out, a.audio_only)
    segs = transcribe(out, a.whisper_model, a.lang)
    descs = []
    if not a.audio_only:
        frames = extract_frames(out, a.change, a.max_frames, a.min_gap, a.max_gap)
        descs = describe_frames(out, frames, segs, a.vl_model)
    if not a.no_guide:
        write_guide(out, segs, descs, a.text_model, goal, a.out_lang, a.think)

    total = sum(t for _, t in TIMINGS)
    log("timings: " + ", ".join(f"{n} {t / 60:.1f} min" for n, t in TIMINGS) + f"; total {total / 60:.1f} min")
    result = "guide.md" if not a.no_guide else "transcript.txt" if a.audio_only else "frames.md"
    log(f"done: {out / result}")


if __name__ == "__main__":
    main()
