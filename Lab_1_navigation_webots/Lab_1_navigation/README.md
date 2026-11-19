# Crazyflie progressive square

Four stages of a 1 m square on a real Crazyflie, using OptiTrack for position.
This folder holds the lab code. For Webots simulation, use the parent repository
[`Lab_1_navigation_webots`](../README.md) and run `./scripts/run-lab-world.sh` or open `simulation/worlds/Lab_1_navigation.wbt` in Webots.
On a real Crazyflie, this folder is enough once dependencies are installed.

Each stage is one empty function, `step()`. Its arguments and its return values are the names in the laboratory notes. A later stage leaves the earlier functions as they are and adds the next block.

| Script | What you write in `step()` |
| --- | --- |
| `step1_position.py` | Equations (3) and (4). Return `(v_x, v_y)`, `yaw_rate`, `height`, `index`, and `hold`. |
| `step2_small_yaw.py` | Equations (5) through (9). Return `yaw_rate` and `psi_set`. The Stage 1 velocity stays. |
| `step3_cruise.py` | Equations (10) and (11), and equation (4) on the last corner. Return `v`, `k`, and `hold`. |
| `step4_acceleration.py` | Equations (12) through (14). Return `velocity_sent`, `acceleration`, and `tilt`. |

`Ctrl+C` lands, then disarms. Close the Crazyflie client before you run a script. The radio allows one link, and that link carries both OptiTrack and the setpoints.

## Install

Python 3.10 or newer, a Crazyradio, and a Crazyflie with a firmware that supports arming and hover setpoints. Windows setup, including the Crazyradio driver, is in [README-Windows.md](README-Windows.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS the Crazyradio also needs libusb. The [Bitcraze USB permission instructions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/) cover Windows, macOS, and Linux.

## Motive

In Motive, stream rigid bodies over multicast:

- Transmission type: Multicast
- Data port: `1511`
- Multicast address: `239.255.42.99`

The laptop running these scripts must be on that network. Pass Motive's IP and this computer's IP on that network. `--body-id` is the rigid-body id in Motive. `--address` is the Crazyflie radio address, for example `0xE7E7E7E7E7`.

## Calibrate once per drone

Motors stay off. Slide the drone by hand: up, toward the nose, then to its left. The nose slide is the wall the drone will fly toward.

```bash
python3 calibrate_axes.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

That writes `axis_calibration_4.json` in this folder. It is local to the lab and is not meant to be committed. Repeat the calibration if the marker rigid body is rebuilt.

## Fly

Put the drone on the floor, nose aimed at the calibration wall, with a clear square in front of it. `+X` is toward that wall. `+Y` is to the left of the nose. The square starts at the takeoff point, 1 m on a side, and climbs 0.5 m. `--side` and `--climb` change those two numbers.

```bash
python3 step1_position.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step2_small_yaw.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step3_cruise.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step4_acceleration.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

Each script asks you to hold the drone still and press Enter so the current pose becomes forward. Then it waits until the onboard estimate matches OptiTrack, and asks for another Enter before arming.

Fly stage 1 before the later stages. Gains live in `gains.py`. Change one number at a time, after the stage that uses it is already flying.

## What each file does

- `step1_position.py` through `step4_acceleration.py` are the programs you run. Each one contains an empty `step()`. The docstring is the list of inputs and outputs for that stage.
- `runner.py` stores the waypoint index, the arrival timer, the yaw setpoint, and the last velocity. It passes those into `step()` under the names in the notes, and flies what `step()` returns.
- `geometry.py` turns world velocity into the body frame.
- `fly_square.py` is the shared flight loop: takeoff, the square, and landing.
- `mocap.py` sends the OptiTrack pose to the Crazyflie and stores the axis map.
- `motive.py` reads Motive's multicast packets.
- `calibrate_axes.py` measures that axis map.
