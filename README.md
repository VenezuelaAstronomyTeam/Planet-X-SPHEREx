# Planet X & SPHEREx — time-lapse viewer

**NASA Space Apps Challenge 2026 · Challenge “Planet X y SPHEREx” · Team Outer Space**

A small web app that finds real calibrated **SPHEREx** images at **NASA/IPAC IRSA**, shows a cutout at any sky position and plays the available dates as a time-lapse.

- **Time-lapse** of the same patch of sky across epochs, with play/pause and speed control
- **Blink** between the oldest and newest frame
- **Difference** between two dates, with optional circles over the strongest changes
- **Search** by object name (SIMBAD/NED) or by RA/Dec, plus presets and zoom
- **Share** the full state in the URL and **export** the sequence as a GIF
- **Casual and Science modes**: a clean visual viewer, or parallax distances and a magnitude curve
- **Spanish / English** interface
- Plain **FastAPI + Astropy + NumPy + Pillow**: no React, no Node, no database, no FITS files written to disk

> This is an exploration prototype. A bright spot in the difference image is **not** a detection of Planet X or of any new object. Always cross-check with Aladin/VizieR and the official SPHEREx documentation.

---

## Requirements

- **Python 3.11+**
- Internet access to `irsa.ipac.caltech.edu`
- A modern browser (Chrome, Edge, Firefox or Safari)

---

## Run on Windows

1. Install Python from <https://www.python.org/downloads/> and check **“Add python.exe to PATH”**.
2. Double-click **`run.bat`** in this folder.
3. The first run installs the dependencies; after that the browser opens at **http://127.0.0.1:8000**.

`run.bat` creates a local `.venv`, installs `requirements.txt`, starts the server and opens the browser. Press **Ctrl+C** in the terminal to stop it.

## Run on any system

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Interactive API docs live at **http://127.0.0.1:8000/docs**.

---

## How to use

1. **Pick a target.** Choose a preset, type a name such as `M42` and press **Resolve**, or type RA/Dec in decimal degrees.
2. **Choose the view.** In **Science** mode, pick field of view (`0.02°`, `0.05°`, `0.1°`), detector band and colormap (`gray`, `inferno`, `viridis`). Casual mode uses the defaults and keeps the interface visual.
3. **Search.** The app asks IRSA for the SPHEREx images that cover that position and preloads the frames.
4. **Play.** Use play/pause, previous/next, the timeline and the `1× / 2× / 5×` speed.
5. **Zoom.** Mouse wheel or `−` / `+` (1×–8×), drag to pan, double-click to reset. The crosshair marks the requested coordinate.
6. **Blink.** Alternates every 400 ms between the oldest and the newest frame.
7. **Difference.** Subtracts the first available frame in the browser; fixed stars cancel to grey and variable or moving sources stand out.
8. **Circles.** Draws amber circles around the strongest difference regions.
9. **Share or export.** *Share state* copies a URL with the full view; *Export GIF* downloads the loaded sequence.
10. **Cross-check.** Buttons open Aladin Lite, IRSA Finder Chart and VizieR at the current position.
11. **Language.** Switch between **ES** and **EN** in the header; the choice is remembered.

### Casual and Science modes

Use the **Casual / Science** switch in the header. The mode is stored in the URL (`mode=science`) and in the browser.

- **Casual** keeps only the visual side: search, presets, playback, blink, difference, circles, zoom, sharing and GIF export. The band/FOV selectors, technical HUD blocks and the science panel are hidden.
- **Science** adds the detector band and field-of-view selectors, wavelength/coordinate HUD, and a panel below the image with:
  - **Parallax distance** from the nearest **Gaia DR3** star within 29″ of the centre, queried through the Heidelberg TAP service. It is a real geometric distance for that star, **not** for the nebula or galaxy in the image.
  - **Parallax quality**: value, error, signal-to-noise and the RUWE flag.
  - **Aperture magnitude per frame**, measured on each SPHEREx cutout in AB magnitudes (1.8 px aperture, about 11″ at 0.05°). This is not a catalogue magnitude: extended emission and the background estimate affect it, and points with errors above 1.5 mag are hidden from the curve.

### Good to know

- The viewer shows **at most 24 frames**, sampled evenly across all available dates. The status bar says how many exist in the archive (for example “24 of 342”).
- Use **one band** for difference and circles. With all six bands the frames mix wavelengths and the difference covers the whole image; the app warns you in that case.
- The magnitude curve is **aperture photometry on the cutout**, not catalogue photometry. Use one band for a meaningful light curve; mixing bands compares different wavelengths.
- SPHEREx cutouts keep the orientation of the parent image, so they are **not north-up**.
- IRSA sometimes answers `503` under load. The app retries and skips frames that keep failing; search again after a few seconds.
- Reopening a position that already loaded reuses the last good result for up to 6 hours.

---

## Project structure

```text
planet-x-spherex/
├── backend/
│   └── main.py          # FastAPI server + the whole UI embedded as INDEX_HTML
├── requirements.txt     # pinned Python dependencies
├── run.bat              # one-click Windows launcher
├── .gitignore           # excludes .venv, caches and OS/editor files
└── README.md            # this file
```

The interface (HTML, CSS and JavaScript) lives in the `INDEX_HTML` string at the end of `backend/main.py` and is served at `/`. Edit that block to change the UI; there is no build step. The local `.venv/` folder is created by `run.bat` and should not be committed.

---

## API

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service status, collections, bands and resolver |
| `GET` | `/api/resolve?name=M42` | Object name to RA/Dec with CDS Sesame (SIMBAD/NED) |
| `GET` | `/api/observations` | SIA v2 search for SPHEREx images |
| `GET` | `/api/cutout` | Render one FITS cutout as a 400 × 400 PNG |
| `GET` | `/api/diff` | Difference of two cutouts as a PNG |
| `POST` | `/api/gif` | Animated GIF from 2–24 cutouts |
| `GET` | `/api/science` | Nearest Gaia DR3 source, parallax and distance |
| `POST` | `/api/photometry` | Aperture magnitudes for a list of SPHEREx frames |
| `GET` | `/docs` | Swagger UI |

```bash
curl "http://127.0.0.1:8000/api/resolve?name=M42"
curl "http://127.0.0.1:8000/api/observations?ra=269.452&dec=4.693&radius=0.05&band=3"
curl "http://127.0.0.1:8000/api/science?ra=283.3963&dec=33.0292"
```

Only the calibrated collections **`spherex_dr1`** and **`spherex_qr3`** are used; `spherex_qr2` is excluded.

---

## SPHEREx bands

| Band | Wavelength range (µm) |
| --- | --- |
| D1 | 0.7455 – 1.1167 |
| D2 | 1.1033 – 1.6499 |
| D3 | 1.6301 – 2.4346 |
| D4 | 2.4027 – 3.8473 |
| D5 | 3.8113 – 4.4300 |
| D6 | 4.4115 – 5.0096 |

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `python` not found on Windows | Reinstall Python and check *“Add python.exe to PATH”*. |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. |
| `Address already in use` on port 8000 | Stop the old server or use `--port 8001` and open that port. |
| IRSA `502` / `503` | The local server is fine; IRSA is temporarily down. Wait a few seconds and press **Retry**. |
| “No calibrated SPHEREx images…” | Increase the search radius or try another position or preset. |
| GIF export is slow | The server downloads and renders up to 24 frames; the first export is the slowest. |

---

## Data and credits

Data: **SPHEREx** calibrated images from the **NASA/IPAC Infrared Science Archive (IRSA)**, collections `spherex_dr1` and `spherex_qr3`, queried through SIA v2.

Distances: **Gaia DR3** (Gaia Collaboration) through the Heidelberg TAP service. Magnitudes in the science panel are **aperture measurements on the SPHEREx cutouts**, not catalogue photometry.

Built for the **NASA Space Apps Challenge 2026**, challenge *“Planet X y SPHEREx”*, by **Team Outer Space**. Results are for exploration and communication only; no candidate is validated.
