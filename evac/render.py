"""Wideo MP4 z przebiegu symulacji (matplotlib + ffmpeg; zapas: GIF)."""
from __future__ import annotations

import time

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


def render_video(res, path: str, fps: int = 10, max_frames: int | None = None, dpi: int = 100,
                 progress_every: int = 100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon

    if progress_every < 1:
        raise ValueError("progress_every musi być większe od zera")
    render_started = time.perf_counter()
    sc, A = res.scene, res.agents
    frames = res.frames
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    x0, y0, x1, y1 = sc.bounds
    w, h = x1 - x0, y1 - y0
    fig_w = 8.0
    fig, ax = plt.subplots(figsize=(fig_w + 1.2, fig_w * (h + 10) / (w + 10) + 0.9), dpi=dpi, constrained_layout=True)
    from .stats import draw_map
    draw_map(ax, sc, res.cfg, draw_exits=False, legend=False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

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
    from matplotlib.patches import Circle
    ev = res.events_obj
    zone_art = []
    for e in (ev.all if ev is not None else []):
        mk = {"fire": "*", "bang": "X", "package": "P"}[e["type"]]
        ax.plot([e["xy"][0]], [e["xy"][1]], marker=mk, ms=14, color="#d9480f", mec="k", mew=0.6, zorder=6)
        ax.text(e["xy"][0], e["xy"][1] + 3, e["label"], color="#d9480f", fontsize=8, ha="center", zorder=6)
        if e["type"] == "package":
            c = Circle(e["xy"], 0.0, fc=(0.85, 0.1, 0.1, 0.12), ec="#c92a2a", lw=1.5, ls="--", zorder=4)
            ax.add_patch(c)
            zone_art.append(c)

    # agenci: kolor wg paniki; przewróceni = czarne X; stojący = jaśniejsze
    cmap = plt.get_cmap("RdYlGn_r")
    sizes = (2 * A.r * 72 * fig_w / (w + 10) * 1.0) ** 2 * 0.9
    sc_art = ax.scatter(frames[0]["x"], frames[0]["y"], s=sizes, c=np.zeros(A.n), cmap=cmap, vmin=0, vmax=1,
                        zorder=5, linewidths=0)
    fall_art = ax.scatter([], [], s=sizes.mean() * 1.2, marker="x", c="k", zorder=7, linewidths=1.2)
    # warstwy etapu 2: nieświadomi (szare), gapie (niebieska obwódka), lekko ranni (pomarańczowa obwódka)
    unaw_art = ax.scatter([], [], s=sizes.mean(), c="#9aa0a6", zorder=5.5, linewidths=0)
    gawk_art = ax.scatter([], [], s=sizes.mean() * 1.6, facecolors="none", edgecolors="#1c7ed6", zorder=6.5,
                          linewidths=1.0)
    inj_art = ax.scatter([], [], s=sizes.mean() * 1.6, facecolors="none", edgecolors="#e8590c", zorder=6.6,
                         linewidths=1.0)
    staged = str(res.cfg.alarm.mode) == "staged"
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="x", color="k", ls="", label="leży")]
    if staged:
        handles.append(Line2D([], [], marker="o", color="#9aa0a6", ls="", label="nieświadomy"))
    if (A.typ == 3).any() or A.returner.any():
        handles.append(Line2D([], [], marker="o", mfc="none", mec="#1c7ed6", ls="", label="gapi się / wraca po znajomego"))
    if str(res.cfg.injury.mode) == "dose":
        handles.append(Line2D([], [], marker="o", mfc="none", mec="#e8590c", ls="", label="lekko ranny"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    def _pts(f, m):
        return np.column_stack((f["x"][m], f["y"][m])) if m.any() else np.empty((0, 2))
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=10,
                   bbox=dict(fc="white", ec="#999", alpha=0.85), zorder=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("poziom paniki")
    var = f", wariant: {res.cfg.variant}" if res.cfg.variant else ""
    ax.set_title(f"Ewakuacja Miasteczka Studenckiego AGH - N={A.n}{var}, SFM (Helbing 2000)")

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
        txt = (f"t = {f['t']:+.0f} s od {'zdarzenia' if staged else 'alarmu'}\newakuowani: {n_ev}/{A.n}\n"
               f"w ruchu: {int((f['state'] == 1).sum())}   leżący: {int(fl.sum())}")
        if "aw" in f:
            un = (f["aw"] == 0) & (f["state"] == 0) & staged & (f["t"] >= 0)
            gk = f["gawk"] & (f["state"] == 0)
            lj = (f["inj"] == 1) & act & ~fl
            unaw_art.set_offsets(_pts(f, un))
            gawk_art.set_offsets(_pts(f, gk))
            inj_art.set_offsets(_pts(f, lj))
            if staged:
                txt += f"\nnieświadomi: {int(un.sum())}   gapie: {int(gk.sum())}"
            if (f["inj"] > 0).any():
                txt += f"\nranni: lekko {int((f['inj'] == 1).sum())}, ciężko {int((f['inj'] == 2).sum())}"
            for c, R in zip(zone_art, f.get("zones", [])):
                c.set_radius(R)
        info.set_text(txt)
        return []

    ffmpeg = _get_ffmpeg()
    try:
        import imageio_ffmpeg
    except ImportError:
        imageio_ffmpeg = None

    total_frames = len(frames)
    last_reported = 0

    def report_progress(i):
        nonlocal last_reported
        completed = i + 1
        if completed > last_reported and (
            completed % progress_every == 0 or completed == total_frames
        ):
            elapsed = time.perf_counter() - render_started
            print(f"Render: {completed}/{total_frames} klatek, {elapsed:.1f} s", flush=True)
            last_reported = completed

    print(f"Renderowanie: {total_frames} klatek (raport co {progress_every})", flush=True)
    if ffmpeg and imageio_ffmpeg is not None:
        out = path
        dynamic = ([smoke_im] if smoke_im is not None else []) + gate_lines + zone_art \
            + [sc_art, fall_art, unaw_art, gawk_art, inj_art, info]
        dynamic_ids = {id(artist) for artist in dynamic}
        foreground = [artist for artist in ax.get_children()
                      if id(artist) not in dynamic_ids and artist.get_visible() and artist.get_zorder() > 5]
        for artist in dynamic + foreground:
            artist.set_animated(True)
        draw_order = sorted(dynamic + foreground, key=lambda artist: artist.get_zorder())
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        width, height = fig.canvas.get_width_height()
        stream = imageio_ffmpeg.write_frames(
            out, (width, height), pix_fmt_in="rgba", pix_fmt_out="yuv420p", fps=fps,
            bitrate="2500k", codec="libx264", macro_block_size=1,
            output_params=["-preset", "ultrafast", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
        )
        stream.send(None)
        try:
            for i in range(len(frames)):
                update(i)
                report_progress(i)
                fig.canvas.restore_region(background)
                for artist in draw_order:
                    ax.draw_artist(artist)
                fig.canvas.blit(fig.bbox)
                stream.send(np.asarray(fig.canvas.buffer_rgba()))
        finally:
            stream.close()
    else:
        if ffmpeg:
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
            writer = animation.FFMpegWriter(
                fps=fps, codec="libx264", bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
            )
            out = path
        else:
            writer = animation.PillowWriter(fps=fps)
            out = path.rsplit(".", 1)[0] + ".gif"
        def update_with_progress(i):
            result = update(i)
            report_progress(i)
            return result

        anim = animation.FuncAnimation(fig, update_with_progress, frames=len(frames), blit=False)
        anim.save(out, writer=writer)
    plt.close(fig)
    print(f"Renderowanie zakończone w {time.perf_counter() - render_started:.1f} s: {out}", flush=True)
    return out
