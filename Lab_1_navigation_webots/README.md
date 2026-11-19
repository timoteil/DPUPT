# Lab 1 — Progressive navigation (Webots + Crazyflie)

This repository bundles:

- **Webots** simulation based on [Bitcraze crazyflie-simulation](https://github.com/bitcraze/crazyflie-simulation) (minimal world, Crazyflie model, shared PID controller).
- **`Lab_1_navigation/`** — the same progressive square lab code used on a real Crazyflie with OptiTrack (see [Lab_1_navigation/README.md](Lab_1_navigation/README.md) for hardware).

In the simulator, the drone flies a 1 m square at 0.6 m height. You implement the four stages in `Lab_1_navigation/step1_position.py` … `step4_acceleration.py`, set **`STAGE`** in `simulation/controllers/crazyflie_controller_lab_1_navigation/crazyflie_controller_lab_1_navigation.py`, save, and reload the world.

## Repository layout

```
Lab_1_navigation_webots/
├── README.md                 ← this file (simulator install)
├── Lab_1_navigation/         ← lab code (sim + real drone)
├── scripts/run-lab-world.sh   ← recommended way to start the sim (macOS/Linux)
├── simulation/                ← Webots project (not named `webots`; see macOS notes)
│   ├── worlds/Lab_1_navigation.wbt
│   ├── controllers/crazyflie_controller_lab_1_navigation/
│   └── protos/Crazyflie.proto
├── meshes/                   ← Crazyflie 2.1 mesh (from Bitcraze)
└── controllers_shared/       ← Webots velocity PID (from Bitcraze)
```

## Prerequisites

- **[Webots](https://cyberbotics.com/)** **R2025a** (the world file targets R2025a; newer Webots may ask to migrate the world).
- **Python 3.10+** and **NumPy** on the interpreter Webots uses for controllers (steps below).
- Disk space ~500 MB including Webots itself.

---

## macOS

### 1. Install Webots

Official guide: [Installing Webots](https://cyberbotics.com/doc/guide/installing-webots).

**If Webots is not installed yet:**

1. Download **R2025a** for macOS from [Cyberbotics downloads](https://cyberbotics.com/#download) or [GitHub releases (R2025a)](https://github.com/cyberbotics/webots/releases/tag/R2025a) (`webots-*-macos.dmg`).
2. Open the `.dmg`, drag **Webots** into **Applications**.
3. First launch: if macOS blocks the app, open **System Settings → Privacy & Security** and choose **Open Anyway** for Webots.

**Optional (Homebrew):** `brew install --cask webots` installs the same application into `/Applications/Webots.app`.

**Check version** (after adding Webots to your `PATH`, see below):

```bash
webots --version
```

Close any running Webots instance first; the command needs a free TCP port in the range 1234–1244.

**macOS 26 and later:** Webots **R2025a** quits in a fraction of a second and prints `Port 1234 is already in use` for every port in the range, even when nothing is listening. That is a Webots bug (a refused local connection is treated as a successful connection). It is not a missing permission, and quitting other Webots processes does not fix it. Install a build that includes the fix, from the [Webots nightly releases](https://github.com/cyberbotics/webots/releases) (a `nightly_*` tag from November 2026 or later; the macOS `.dmg`). Drag that **Webots** app into **Applications**, replacing R2025a, then run `./scripts/run-lab-world.sh` again.

### 2. Launch the lab (recommended)

From the repository root:

```bash
./scripts/run-lab-world.sh
```

That script sets macOS Qt paths, points `WEBOTS_HOME` at `/Applications/Webots.app`, and starts Webots from the **`simulation/`** project directory.

**Do not run** `webots webots/worlds/...` from the repo root. A project folder literally named `webots` makes the Webots CLI resolve the install under `/Users/you/work/Contents/...` and fail with `qt_warning_filters.conf` missing.

GUI-only alternative:

```bash
open -a Webots simulation/worlds/Lab_1_navigation.wbt
```

### 3. Optional: `webots` in Terminal

Add to `~/.zshrc`:

```bash
export WEBOTS_APP="/Applications/Webots.app"
export WEBOTS_HOME="$WEBOTS_APP"
export PATH="$WEBOTS_APP/Contents/MacOS:$PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="$WEBOTS_APP/Contents/lib/webots/qt/plugins/platforms"
export QT_PLUGIN_PATH="$WEBOTS_APP/Contents/lib/webots/qt/plugins"
```

Then always **`cd simulation`** before calling `webots worlds/Lab_1_navigation.wbt`, or use `./scripts/run-lab-world.sh`.

**Ports 1234–1244 already in use:** quit other Webots windows (**Webots → Quit**) or run `pkill -x webots` before starting again.

### 4. NumPy for Python controllers

`/usr/bin/python3` on a Mac without Command Line Tools is a stub, so Webots reports that Python was not found. This project uses a virtual environment at `.venv` (not committed). Create it once:

```bash
# install uv if needed: https://docs.astral.sh/uv/
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy
```

`simulation/controllers/crazyflie_controller_lab_1_navigation/runtime.ini` must set `COMMAND` to that interpreter. On this Mac it is:

`/Users/tlala/work/CF_webots/Lab_1_navigation_webots/.venv/bin/python`

The same path belongs in **Webots → Preferences → General → Python command**. Quit Webots and open it again after changing either one, then reload the world.

### 5. Run a stage

1. Edit the stage file under `Lab_1_navigation/` (for example `step1_position.py`).
2. Set `STAGE = 1` (or 2, 3, 4) in `simulation/controllers/crazyflie_controller_lab_1_navigation/crazyflie_controller_lab_1_navigation.py`.
3. Reload the world in Webots and press **Play**.

Real-drone setup on macOS (Crazyradio, OptiTrack) is the same as Linux in [Lab_1_navigation/README.md](Lab_1_navigation/README.md); USB permissions follow [Bitcraze’s install notes](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/).

---

## Linux

### 1. Install Webots

Follow [Cyberbotics — Installing Webots](https://cyberbotics.com/doc/guide/installing-webots). On Ubuntu, the `.deb` installer is typical.

Confirm Webots starts:

```bash
webots --version
```

### 2. Get this repository

```bash
git clone <your-github-url> Lab_1_navigation_webots
cd Lab_1_navigation_webots
```

Or download and extract the ZIP from GitHub, then `cd` into the folder.

### 3. Install NumPy for Webots’ Python

Webots runs controllers with its own Python. Install NumPy there (adjust `WEBOTS_HOME` if you installed elsewhere):

```bash
export WEBOTS_HOME=/usr/local/webots
"$WEBOTS_HOME/projects/controller/python" -m pip install --user numpy
```

If that Python is not found, open Webots → **Tools → Preferences → General** and note the **Python command** shown there; run `"<that command>" -m pip install --user numpy`.

For a **real Crazyflie** later, also create a venv in `Lab_1_navigation/` and `pip install -r requirements.txt` (see [Lab_1_navigation/README.md](Lab_1_navigation/README.md)).

### 4. Open the world

From this repository root:

```bash
./scripts/run-lab-world.sh
```

Or in the Webots GUI: **File → Open World…** → `simulation/worlds/Lab_1_navigation.wbt`.

The **project directory** must be the `simulation/` folder (the launch script and GUI both set this).

### 5. Run a stage

1. Edit the stage file under `Lab_1_navigation/` (for example `step1_position.py`).
2. Set `STAGE = 1` (or 2, 3, 4) in `simulation/controllers/crazyflie_controller_lab_1_navigation/crazyflie_controller_lab_1_navigation.py`.
3. In Webots, reload the world (**File → Reload World** or the reload button).
4. Press **Play**. Console output appears in the Webots terminal.

---

## Windows

### 1. Install Webots

Download the Windows installer from [cyberbotics.com](https://cyberbotics.com/) and install **R2025a** (or migrate the world if you use a newer release). Start **Webots** once from the Start menu to confirm it runs.

### 2. Get this repository

Clone with Git, or download **Code → Download ZIP** from GitHub and extract it, for example to:

`C:\Users\<you>\Lab_1_navigation_webots`

### 3. Install NumPy for Webots’ Python

Default install path is often `C:\Program Files\Webots`. In **PowerShell**:

```powershell
cd "C:\Program Files\Webots"
.\projects\controller\python.exe -m pip install --user numpy
```

If that path does not exist, open Webots → **Tools → Preferences → General** and use the **Python command** shown there:

```powershell
& "<path-from-preferences>" -m pip install --user numpy
```

For **real-drone** setup (Crazyradio, OptiTrack, `cflib`), see [Lab_1_navigation/README-Windows.md](Lab_1_navigation/README-Windows.md).

### 4. Open the world

In Webots: **File → Open World…** → select:

`Lab_1_navigation_webots\simulation\worlds\Lab_1_navigation.wbt`

Or from PowerShell (if `webots` is on your PATH):

```powershell
cd C:\Users\<you>\Lab_1_navigation_webots\simulation
webots worlds\Lab_1_navigation.wbt
```

### 5. Run a stage

Same as on Linux: edit `Lab_1_navigation\step*.py`, set **`STAGE`** in `crazyflie_controller_lab_1_navigation.py`, reload the world, press **Play**.

### Optional: controller `runtime.ini`

`simulation/controllers/crazyflie_controller_lab_1_navigation/runtime.ini` defaults to `python3`. On Windows, if the controller fails to start, set `COMMAND` to Webots’ controller Python, for example:

```ini
[python]
COMMAND = C:\Program Files\Webots\projects\controller\python.exe
```

Use the path from **Tools → Preferences → General** if yours differs.

---

## Simulation vs real hardware

| | Webots | Real Crazyflie |
| --- | --- | --- |
| Entry point | World + `crazyflie_controller_lab_1_navigation.py` | `python stepN_….py …` (see lab README) |
| Flight loop in sim | `Lab_1_navigation/sim_runner.py` | `Lab_1_navigation/runner.py` + `fly_square.py` |
| Dependencies | NumPy (+ Webots `controller` module at runtime) | NumPy, `cflib`, OptiTrack network |

The stage files in `Lab_1_navigation/` are shared. Template signatures for the handout live in `step1_position_st.py` and similar `*_st*.py` files.

---

## Uploading to GitHub

This folder is intended to be the **root of a new Git repository** (not the whole `CF_webots` tree):

```bash
cd Lab_1_navigation_webots
git init -b main
git add -A
git commit -m "Lab 1 navigation with Webots simulator"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Do not commit virtual environments, `__pycache__`, or local calibration JSON files (see `.gitignore`).

---

## License and attribution

Webots assets, meshes, and `controllers_shared/` come from the Bitcraze **crazyflie-simulation** project; see [LICENSE](LICENSE). Lab navigation code in `Lab_1_navigation/` is part of your course materials unless you add a separate license file there.
