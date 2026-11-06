# Lab 2 — Follow Me

A Crazyflie with an AI-deck follows a person. The deck camera is shown in an OpenCV window, YOLO11 nano marks the person, and OptiTrack supplies the drone pose. With `--hover` the drone climbs and flies so the person stays near the center of the image.

This folder is meant to be copied or cloned on its own and run on Windows. The Python scripts, `yolo11n.pt`, and the axis calibration files are all here. Create the virtual environment on the Windows PC. It only works on the machine that created it.

## What you need

- Windows 10 or 11, 64-bit
- Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/). In the installer, check **Add python.exe to PATH**. Python 3.13 needs extra Visual Studio build tools for the Crazyflie library, so use 3.11 or 3.12.
- A Crazyflie with an AI-deck running the [wifi-img-streamer](https://github.com/bitcraze/aideck-gap8-examples/tree/master/examples/other/wifi-img-streamer) firmware
- A Crazyradio USB dongle
- OptiTrack Motive on another PC, streaming the drone's rigid body
- Two network connections on this PC at the same time:
  - Wi-Fi joined to the AI-deck
  - Ethernet, or a second adapter, on the Motive network

The AI-deck access point is:

| | |
| --- | --- |
| Network name | `WiFi streaming example` |
| Password | none |
| Address | `192.168.4.1` |
| Port | `5000` |

That Wi-Fi has no internet. Install Python packages **before** you join it.

## Install on Windows

Plug into a network that can reach the internet. The PyTorch download is a few hundred megabytes and can take several minutes.

### Command Prompt

```bat
cd /d C:\path\to\Lab_2_Follow_Me
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`setup.bat` runs those commands for you:

```bat
cd /d C:\path\to\Lab_2_Follow_Me
setup.bat
.venv\Scripts\activate.bat
```

### PowerShell

```powershell
cd C:\path\to\Lab_2_Follow_Me
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, allow it for this window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` runs the install for you:

```powershell
cd C:\path\to\Lab_2_Follow_Me
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
```

The CPU build of PyTorch is enough. The camera image is 324×244. Install that CPU build **before** `requirements.txt` so pip does not pull the much larger CUDA wheel. A machine with an NVIDIA GPU can skip the CPU line and let `ultralytics` install its default PyTorch.

`requirements.txt` installs:

| Package | What it is for |
| --- | --- |
| `opencv-python` | Camera window |
| `ultralytics` | YOLO11 nano person detector |
| `cflib` | Crazyradio link to the Crazyflie |
| `numpy` | Images and poses |
| `matplotlib` | Pulled in with the Crazyflie tools |

## Crazyradio driver

Windows needs a USB driver before `cflib` can open the dongle. There is no separate radio package. `pip install cflib` brings the Python side, including the libusb library.

1. Plug in the Crazyradio.
2. Download [Zadig](https://zadig.akeo.ie/) and run it.
3. If the dongle is missing from the list, open **Options** and choose **List All Devices**.
4. Select the Crazyradio.
5. Choose the **libusb** driver and click **Install** or **Replace Driver**.
6. In Device Manager the dongle should show up as a libusb device.

Bitcraze documents the same steps here: [Installing the USB driver on Windows](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

Close the Crazyflie client before starting a script. The dongle allows one link.

## Motive

In Motive:

- Transmission Type: **Multicast**
- Broadcast Frame Data: **on**
- Data port: `1511`
- Multicast address: `239.255.42.99`

Quit any other NatNet client first. It holds UDP port 1511.

When Windows asks, allow Python on the private network so it can receive that UDP stream.

To add the firewall rule yourself, open PowerShell as administrator and run this after the virtual environment exists. Change the path if the folder is somewhere else:

```powershell
New-NetFirewallRule -DisplayName "Lab 2 Follow Me Python" -Direction Inbound -Program "C:\path\to\Lab_2_Follow_Me\.venv\Scripts\python.exe" -Action Allow -Profile Private
```

## Axis calibration

Each rigid body has a file in this folder named `axis_calibration_<id>.json`. Files are included for body ids 2, 3, 4, 5, 6, and 7. `axis_calibration.json` is the fallback when a numbered file is missing.

At the start of a flight, face the nose at the same wall used when that file was measured, and hold still when the script asks. Then press Enter.

## Run

Run every command from this folder, with `.venv` activated.

### 1. Camera only

Join `WiFi streaming example`, then:

```bat
python opencv-viewer.py
```

A window named `AI-deck` should show the live picture. Press `q` or Esc to stop. If the window never opens, check the connection:

```bat
ping 192.168.4.1
```

### 2. Person boxes, drone stays on the ground

Use the Motive PC address for the first argument. `--client` is **this** PC's address on the Motive network, which is the Ethernet adapter, not `192.168.4.x`. `--address` is the Crazyradio address printed on that Crazyflie. `--body-id` must match a calibration file.

```bat
python follow_me.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

The window shows the person box. The drone stays disarmed. Close the window, or press `q`, to stop.

### 3. Follow

Use a clear area. The drone climbs, then moves to keep the person in view. Closing the window sends it back toward the takeoff point and lands.

```bat
python follow_me.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
```

`--climb` is the height above takeoff before following (default 0.9 m). The flight stays inside `--max-radius` (default 1 m) and between `--min-up` and `--max-up`. Speed is limited to 0.3 m/s and yaw to 30 deg/s.

`follow_me_pid.py` is the same flight with PID gains on the image error. Its speed limit is 0.5 m/s. The command line is the same:

```bat
python follow_me_pid.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
```

### Hover in place

`stream_hover_yolo.py` climbs, holds, and lands. It follows nobody. Add `--detection` to draw YOLO boxes during that hover.

```bat
python stream_hover_yolo.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7 --hover
```

`stream_hover_v2.py` is a shorter hover: 0.4 m over 4 seconds, then a 10 second hold. It has no `--flip` flag.

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--client` | This PC's IP on the Motive network |
| `--body-id` | Motive rigid body id |
| `--address` | Crazyradio address, for example `0xE7E7E7E7E7` |
| `--hover` | Arm, climb, and fly. Without it the drone stays on the ground |
| `--flip` | Rotate the camera image 180 degrees |
| `-n` / `-p` | AI-deck IP and port. Defaults are `192.168.4.1` and `5000` |
| `--scale 1` | Show the image at native size. Default is 2 |
| `--save` | Write the frames on screen under `stream_out\` |

## If it fails

**`py` is not recognized.** Reinstall Python and check **Add python.exe to PATH**, then open a new Command Prompt.

**`Activate.ps1` cannot be loaded.** Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window, then activate again.

**The camera window does not open.** Confirm this PC is on `WiFi streaming example` and `ping 192.168.4.1` answers. One viewer at a time. Run the script from a normal desktop session, with the virtual environment activated.

**`No UDP packets arrived on port 1511`.** Motive must be multicasting on port 1511 to `239.255.42.99`. Allow Python through the firewall. Pass `--client` with the address of the adapter that can see Motive. Close any other script that is already listening on 1511.

**The script sees other rigid bodies, not yours.** The number after `--body-id` has to be the id Motive assigned to this drone.

**`No axis calibration for rigid body`.** This folder has files for ids 2 through 7. Use one of those ids, or add `axis_calibration_<id>.json` next to the scripts.

**The radio does not connect.** Install the libusb driver with Zadig. Close the Crazyflie client. Only one program can hold the dongle.

**The window opens and then the process quits with `OMP: Error #15`.** OpenCV and PyTorch both ship OpenMP. The scripts set `KMP_DUPLICATE_LIB_OK` before those libraries load. If you still see it, set it in the same Command Prompt and start again:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
python follow_me.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

**YOLO sits still after the axis map.** The lab network often has no DNS. The scripts set `YOLO_OFFLINE` and load `yolo11n.pt` from this folder, so they do not download weights at startup. Keep that file next to the scripts.

## What is in this folder

| File | Role |
| --- | --- |
| `opencv-viewer.py` | Live AI-deck window |
| `follow_me.py` | Follow the person |
| `follow_me_pid.py` | Same follow, with PID gains |
| `stream_hover_yolo.py` | Camera, OptiTrack, radio, and YOLO shared by the follow scripts |
| `stream_hover_v2.py` | Hover in place, same arguments |
| `yolo11n.pt` | YOLO11 nano weights |
| `axis_calibration_*.json` | OptiTrack to Crazyflie axis map for each rigid body |
| `requirements.txt` | Python packages |
| `setup.bat`, `setup.ps1` | Create `.venv` and install those packages |

`opencv-viewer.py` is Copyright (C) 2021 Bitcraze AB and is licensed under the GNU General Public License, version 2 or later. The notice is at the top of that file. `yolo11n.pt` is the Ultralytics YOLO11 nano checkpoint.

Captured frames, the virtual environment, and `__pycache__` are not part of the project. `.gitignore` leaves them out of git.
