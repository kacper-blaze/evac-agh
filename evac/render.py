"""Wideo MP4 z przebiegu symulacji (matplotlib + ffmpeg; zapas: GIF)."""
from __future__ import annotations

import numpy as np


def _get_ffmpeg():
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def render_video(res, path: str, fps: int = 10, max_frames: int | None = None, dpi: int = 100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon

    sc, A = res.scene, res.agents
    frames = res.frames
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    x0, y0, x1, y1 = sc.bounds
    w, h = x1 - x0, y1 - y0
    fig_w = 8.0
    fig, ax = plt.subplots(figsize=(fig_w + 1.2, fig_w * (h + 10) / (w + 10) + 0.9), dpi=dpi, constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 5, x1 + 5)
    ax.set_ylim(y0 - 5, y1 + 5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.add_patch(Polygon(sc.polygon, closed=True, fc="#f4f1ea", ec="#444", lw=1.2, zorder=0))
    for t in sc.tunnels:
        ax.add_patch(Polygon(t, closed=True, fc="#f4f1ea", ec="#777", lw=0.8, zorder=0))
    for o in sc.fixed_obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="#8a8f98", ec="#444", lw=0.6, zorder=1))
    for o in sc.bench_obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="#9b6b43", ec="#5f3d25", lw=0.4, zorder=1))
    for b in sc.barriers:
        ax.plot([b["from"][0], b["to"][0]], [b["from"][1], b["to"][1]], color="k", lw=2, zorder=2)

    # dym
    smoke_im = None
    if "smoke" in frames[0]:
        cm = ListedColormap([(0.25, 0.25, 0.25, a) for a in np.linspace(0, 0.5, 256)])
        ext = [sc.x0, sc.x0 + res.frames[0]["smoke"].shape[1] * float(res.cfg.grid.smoke_cell_m),
               sc.y0, sc.y0 + res.frames[0]["smoke"].shape[0] * float(res.cfg.grid.smoke_cell_m)]
        smoke_im = ax.imshow(frames[0]["smoke"], origin="lower", extent=ext, cmap=cm, vmin=0, vmax=255,
                             zorder=3, interpolation="bilinear")
    # bramy
    gate_lines, gate_txt = [], []
    for k, e in enumerate(sc.exits):
        ln, = ax.plot([e.a[0], e.b[0]], [e.a[1], e.b[1]], lw=4, color="#2e9e4f", zorder=4, solid_capstyle="butt")
        gate_lines.append(ln)
        t = ax.text(e.center[0] + 2.5 * e.n[0], e.center[1] + 2.5 * e.n[1], e.id, fontsize=9, weight="bold",
                    ha="center", va="center", zorder=6)
        gate_txt.append(t)
    ox, oy = res.cfg.hazard.origin_xy
    if res.cfg.hazard.enabled:
        ax.plot([ox], [oy], marker="*", ms=14, color="#d9480f", zorder=6)
        ax.text(ox, oy + 3, str(res.cfg.hazard.source_label), color="#d9480f", fontsize=8, ha="center", zorder=6)

    # agenci: kolor wg paniki; przewróceni = czarne X; stojący = jaśniejsze
    cmap = plt.get_cmap("RdYlGn_r")
    sizes = (2 * A.r * 72 * fig_w / (w + 10) * 1.0) ** 2 * 0.9
    sc_art = ax.scatter(frames[0]["x"], frames[0]["y"], s=sizes, c=np.zeros(A.n), cmap=cmap, vmin=0, vmax=1,
                        zorder=5, linewidths=0)
    fall_art = ax.scatter([], [], s=sizes.mean() * 1.2, marker="x", c="k", zorder=7, linewidths=1.2)
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=10,
                   bbox=dict(fc="white", ec="#999", alpha=0.85), zorder=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("poziom paniki")
    ax.set_title(f"Ewakuacja Miasteczka Studenckiego AGH - N={A.n}, SFM (Helbing 2000)")

    def update(i):
        f = frames[i]
        act = f["state"] != 2
        xy = np.column_stack((f["x"], f["y"]))
        xy[~act] = np.nan
        sc_art.set_offsets(xy)
        sc_art.set_array(f["p"].astype(float) / 255.0)
        fl = f["state"] == 3
        fall_art.set_offsets(np.column_stack((f["x"][fl], f["y"][fl])) if fl.any() else np.empty((0, 2)))
        if smoke_im is not None and "smoke" in f:
            smoke_im.set_data(f["smoke"])
        for k, ln in enumerate(gate_lines):
            ln.set_color("#2e9e4f" if f["open"][k] else "#c92a2a")
        n_ev = int((f["state"] == 2).sum())
        info.set_text(f"t = {f['t']:+.0f} s od alarmu\newakuowani: {n_ev}/{A.n}\n"
                      f"w ruchu: {int((f['state'] == 1).sum())}   przewróceni: {int(fl.sum())}")
        return []

    ffmpeg = _get_ffmpeg()
    if ffmpeg:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2500,
                                        extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
        out = path
    else:
        writer = animation.PillowWriter(fps=fps)
        out = path.rsplit(".", 1)[0] + ".gif"
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out, writer=writer)
    plt.close(fig)
    return out
