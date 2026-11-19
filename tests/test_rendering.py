import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evac.config import load_config
from evac.render import render_video
from evac.sim import Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(HERE, "..", "scenarios", "etap2.yaml")
MAP = os.path.join(HERE, "..", "scenarios", "mapa_ms.json")


def test_record_interval_can_be_smaller_than_coarse_step(tmp_path, capsys):
    cfg, _ = load_config(
        SCENARIO,
        {
            "agents": {"count": 30, "party_phase_s": 0.0},
            "sim": {"t_max_s": 0.5, "record_dt_s": 0.05},
            "events": [],
            "alarm": {"mode": "legacy"},
            "hazard": {"enabled": False},
            "output": {"heatmap": False},
        },
    )
    res = Simulation(cfg, MAP, backend="numpy", record=True).run()
    times = np.asarray([frame["t"] for frame in res.frames])

    assert len(times) == 11
    assert np.allclose(np.diff(times), 0.05)
    assert np.isclose(times[-1], 0.5)

    out = render_video(res, str(tmp_path / "short.mp4"), fps=20, dpi=40, progress_every=4)
    assert os.path.isfile(out)
    progress = capsys.readouterr().out
    assert "Render: 4/11 klatek" in progress
    assert "Render: 8/11 klatek" in progress
    assert "Render: 11/11 klatek" in progress
    assert "Renderowanie zakończone w" in progress
    if out.endswith(".gif"):
        from PIL import Image
        with Image.open(out) as image:
            assert image.n_frames == len(times)
    else:
        try:
            import imageio_ffmpeg
        except ImportError:
            return
        reader = imageio_ffmpeg.read_frames(out, pix_fmt="rgb24")
        next(reader)
        assert sum(1 for _ in reader) == len(times)
        reader.close()
        frame_count, duration = imageio_ffmpeg.count_frames_and_secs(out)
        assert frame_count == len(times)
        assert abs(duration - (times[-1] - times[0])) <= 1 / 20 + 1e-3
