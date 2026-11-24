import numpy as np
import pytest

import yt2guide


def frame(value: int) -> np.ndarray:
    """A uniform 18x32 grayscale sample frame."""
    return np.full((18, 32), value, np.uint8)


def partial_frame(base: int, changed_frac: float, hi: int) -> np.ndarray:
    """An 18x32 frame where only `changed_frac` of the pixels differ from `base`."""
    f = frame(base)
    flat = f.reshape(-1)
    flat[: int(round(changed_frac * flat.size))] = hi
    return f


# --- ts -----------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds, sep, expected",
    [
        (0, ":", "00:00"),
        (65, ":", "01:05"),
        (3599, ":", "59:59"),
        (3600, ":", "1:00:00"),
        (3725.9, ":", "1:02:05"),
        (65, "-", "01-05"),
    ],
)
def test_ts(seconds, sep, expected):
    assert yt2guide.ts(seconds, sep=sep) == expected


# --- transcript_around ----------------------------------------------------

def test_transcript_around_empty():
    assert yt2guide.transcript_around([], 100) == ""


def test_transcript_around_joins_overlapping_texts_in_order():
    t, before, after = 70, 40, 20  # window is [30, 90]
    segs = [
        {"start": 0, "end": 30, "text": "exact-end"},       # ends exactly at t-before -> included
        {"start": -10, "end": 29.9, "text": "just-before"},  # ends just before t-before -> excluded
        {"start": 40, "end": 50, "text": "middle"},
        {"start": 90, "end": 100, "text": "exact-start"},    # starts exactly at t+after -> included
        {"start": 90.1, "end": 120, "text": "just-after"},   # starts just after t+after -> excluded
    ]
    assert yt2guide.transcript_around(segs, t, before=before, after=after) == \
        "exact-end middle exact-start"


def test_transcript_around_custom_window():
    segs = [{"start": 20, "end": 30, "text": "x"}]
    assert yt2guide.transcript_around(segs, 25, before=3, after=3) == "x"
    assert yt2guide.transcript_around(segs, 100, before=3, after=3) == ""


# --- pick_times -----------------------------------------------------------

def test_pick_times_empty():
    assert yt2guide.pick_times([], 0.5, 1, 5, 0) == []


def test_pick_times_single_frame():
    assert yt2guide.pick_times([frame(10)], 0.5, 1, 5, 0) == [0.0]


def test_pick_times_static_image():
    samples = [frame(10)] * 5
    assert yt2guide.pick_times(samples, 0.5, 1, 5, 0) == [0.0]


def test_pick_times_jump_settles():
    # frame 0-1: unchanged (10); frame 2: jumps to 200; frames 3-4: stay at 200 (settled).
    samples = [frame(10), frame(10), frame(200), frame(200), frame(200)]
    assert yt2guide.pick_times(samples, 0.5, 1, 5, 0) == [0.0, 1.5]


def test_pick_times_change_before_min_gap_is_delayed_not_dropped():
    # the jump at t=0.5 is within min_gap=1 of the last kept time (t=0) so it is
    # not kept at that moment, but it is not lost either: it is kept once min_gap
    # has elapsed (t=1.0), not at the moment it first appeared (t=0.5).
    samples = [frame(10), frame(200), frame(200)]
    result = yt2guide.pick_times(samples, 0.5, min_gap=1, max_gap=5, max_frames=0)
    assert 0.5 not in result
    assert result == [0.0, 1.0]


def test_pick_times_small_change_added_after_max_gap():
    # a change of 0.2 (>= change/5 == 0.1, < change == 0.5) never triggers the
    # "settled change" rule on its own, so it is only picked up once max_gap elapses.
    ref = frame(10)
    small = partial_frame(10, 0.2, 200)
    samples = [ref] + [small] * 20
    result = yt2guide.pick_times(samples, change=0.5, min_gap=1, max_gap=3, max_frames=0)
    assert result == [0.0, 3.0]


def test_pick_times_continuous_change_not_added_until_settled():
    # a continuously changing image (e.g. a scroll) differs from the previous
    # sample every step, so it is never "settled" until it stops moving.
    samples = [frame(10), frame(60), frame(110), frame(160), frame(160)]
    result = yt2guide.pick_times(samples, change=0.5, min_gap=0.4, max_gap=100, max_frames=0)
    assert result == [0.0, 2.0]


def test_pick_times_end_frame_added_when_different_from_last_kept():
    # the clip ends mid-scroll: the last sample never settles, but it is still
    # kept because it differs from the last kept (reference) frame.
    samples = [frame(10), frame(60), frame(110), frame(160), frame(210)]
    result = yt2guide.pick_times(samples, change=0.5, min_gap=0.4, max_gap=100, max_frames=0)
    assert result == [0.0, 2.0]


def _five_jumps():
    vals = [10, 10, 10, 60, 60, 60, 110, 110, 110, 160, 160, 160, 210, 210, 210]
    samples = [frame(v) for v in vals]
    return samples


def test_pick_times_max_frames_zero_means_unlimited():
    samples = _five_jumps()
    full = yt2guide.pick_times(samples, 0.5, 0.4, 100, max_frames=0)
    assert len(full) == 5


def test_pick_times_max_frames_trims_evenly_keeping_first_and_last():
    samples = _five_jumps()
    full = yt2guide.pick_times(samples, 0.5, 0.4, 100, max_frames=0)
    trimmed = yt2guide.pick_times(samples, 0.5, 0.4, 100, max_frames=3)
    assert trimmed[0] == full[0]
    assert trimmed[-1] == full[-1]
    assert trimmed == [full[0], full[2], full[4]]


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError,
                    reason="pick_times(..., max_frames=1) divides by (max_frames - 1) == 0")
def test_pick_times_max_frames_one_divides_by_zero():
    samples = _five_jumps()
    yt2guide.pick_times(samples, 0.5, 0.4, 100, max_frames=1)


# --- media ------------------------------------------------------------

def test_media_none_when_empty(tmp_path):
    assert yt2guide.media(tmp_path) is None


def test_media_video_takes_priority_over_audio(tmp_path):
    (tmp_path / "audio.webm").write_text("")
    (tmp_path / "video.mp4").write_text("")
    assert yt2guide.media(tmp_path) == tmp_path / "video.mp4"


def test_media_first_audio_file_sorted_skipping_json_and_part(tmp_path):
    (tmp_path / "audio.part").write_text("")
    (tmp_path / "audio.info.json").write_text("{}")
    (tmp_path / "audio.webm").write_text("")
    assert yt2guide.media(tmp_path) == tmp_path / "audio.webm"


def test_media_none_when_only_part_and_json(tmp_path):
    (tmp_path / "audio.part").write_text("")
    (tmp_path / "audio.info.json").write_text("{}")
    assert yt2guide.media(tmp_path) is None
