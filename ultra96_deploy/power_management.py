#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess


CPUFREQ_BASE = "/sys/devices/system/cpu"
AVAILABLE_GOVS = ["performance", "powersave", "ondemand", "conservative", "userspace"]


def get_cpu_count() -> int:
    return 4


def read_sysfs(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError) as e:
        return f"(error: {e})"


def write_sysfs(path: str, value: str):
    try:
        with open(path, 'w') as f:
            f.write(value)
    except PermissionError:
        print(f"  ERROR: Permission denied writing to {path}. Run with sudo.")
    except Exception as e:
        print(f"  ERROR: {e}")


def get_cpu_info():
    # prints current frequency, governor, and range for each core
    print("\n--- CPU Frequency Info ---")
    avail = get_available_governors(0)
    if avail:
        print(f"  Available governors: {', '.join(avail)}")
    for cpu_id in range(get_cpu_count()):
        base = f"{CPUFREQ_BASE}/cpu{cpu_id}"
        online = read_sysfs(f"{base}/online") if cpu_id > 0 else "1"
        if online != "1":
            print(f"  CPU{cpu_id}: OFFLINE")
            continue
        freq = read_sysfs(f"{base}/cpufreq/scaling_cur_freq")
        gov = read_sysfs(f"{base}/cpufreq/scaling_governor")
        min_f = read_sysfs(f"{base}/cpufreq/scaling_min_freq")
        max_f = read_sysfs(f"{base}/cpufreq/scaling_max_freq")
        try:
            freq_mhz = int(freq) / 1000
            print(f"  CPU{cpu_id}: {freq_mhz:.0f} MHz | gov={gov} | range={int(min_f)//1000}-{int(max_f)//1000} MHz")
        except ValueError:
            print(f"  CPU{cpu_id}: freq={freq} | gov={gov}")


def get_available_governors(cpu_id: int = 0) -> list:
    path = f"{CPUFREQ_BASE}/cpu{cpu_id}/cpufreq/scaling_available_governors"
    raw = read_sysfs(path)
    if raw.startswith("(error"):
        return []
    return raw.split()


def set_cpu_governor(governor: str):
    # sets governor on all online cores with auto-fallback if unavailable
    available = get_available_governors(0)
    if available and governor not in available:
        print(f"\n  WARNING: '{governor}' not available. Available: {available}")
        # Fall back: prefer ondemand > conservative > performance > first available
        for fallback in ['ondemand', 'conservative', 'performance']:
            if fallback in available:
                print(f"  Falling back to '{fallback}'")
                governor = fallback
                break
        else:
            if available:
                governor = available[0]
                print(f"  Falling back to '{governor}'")
            else:
                print(f"  ERROR: No governors available.")
                return

    print(f"\nSetting CPU governor to '{governor}'...")
    for cpu_id in range(get_cpu_count()):
        online_path = f"{CPUFREQ_BASE}/cpu{cpu_id}/online"
        if cpu_id > 0 and os.path.exists(online_path) and read_sysfs(online_path) != "1":
            continue
        path = f"{CPUFREQ_BASE}/cpu{cpu_id}/cpufreq/scaling_governor"
        if os.path.exists(path):
            write_sysfs(path, governor)
    print(f"  Done. Governor set to: {governor}")


def set_cpu_frequency(freq_khz: int):
    # sets exact frequency (needs userspace governor) or clamps min/max
    available = get_available_governors(0)
    if 'userspace' not in available:
        print(f"\n  WARNING: 'userspace' governor not available. Cannot set exact frequency.")
        print(f"  Available governors: {available}")
        print(f"  Using min/max frequency limits instead...")
        for cpu_id in range(get_cpu_count()):
            online_path = f"{CPUFREQ_BASE}/cpu{cpu_id}/online"
            if cpu_id > 0 and os.path.exists(online_path) and read_sysfs(online_path) != "1":
                continue
            write_sysfs(f"{CPUFREQ_BASE}/cpu{cpu_id}/cpufreq/scaling_max_freq", str(freq_khz))
            write_sysfs(f"{CPUFREQ_BASE}/cpu{cpu_id}/cpufreq/scaling_min_freq", str(freq_khz))
        print(f"  Done. Frequency clamped to {freq_khz // 1000} MHz.")
        return
    print(f"\nSetting CPU frequency to {freq_khz // 1000} MHz...")
    set_cpu_governor("userspace")
    for cpu_id in range(get_cpu_count()):
        online_path = f"{CPUFREQ_BASE}/cpu{cpu_id}/online"
        if cpu_id > 0 and os.path.exists(online_path) and read_sysfs(online_path) != "1":
            continue
        path = f"{CPUFREQ_BASE}/cpu{cpu_id}/cpufreq/scaling_setspeed"
        if os.path.exists(path):
            write_sysfs(path, str(freq_khz))
    print(f"  Done.")


def set_online_cores(n_cores: int):
    # enables first n cores, offlines the rest (CPU0 always on)
    total = get_cpu_count()
    n_cores = max(1, min(n_cores, total))
    print(f"\nSetting {n_cores}/{total} CPU cores online...")
    for cpu_id in range(1, total):  # CPU0 always online
        path = f"{CPUFREQ_BASE}/cpu{cpu_id}/online"
        if os.path.exists(path):
            value = "1" if cpu_id < n_cores else "0"
            write_sysfs(path, value)
            state = "ONLINE" if value == "1" else "OFFLINE"
            print(f"  CPU{cpu_id}: {state}")

    print(f"  CPU0: ONLINE (always)")


def set_pl_clock(freq_mhz: int, clock_idx: int = 0):
    # sets PL fabric clock via PYNQ Clocks API or sysfs fallback
    print(f"\nSetting PL clock {clock_idx} to {freq_mhz} MHz...")
    try:
        from pynq import Clocks
        Clocks.fclk0_mhz = freq_mhz
        actual = Clocks.fclk0_mhz
        print(f"  PL clock 0 set to: {actual:.1f} MHz")
    except ImportError:
        print("  PYNQ not available. Attempting sysfs fallback...")
        fclk_path = f"/sys/class/fclk/fclk{clock_idx}/set_rate"
        if os.path.exists(fclk_path):
            write_sysfs(fclk_path, str(freq_mhz * 1_000_000))
        else:
            print(f"  ERROR: {fclk_path} not found. Cannot set PL clock without PYNQ.")


def disable_pl_clocks(keep_clock0=True):
    # sets PL clocks 1-3 to minimum (~1 MHz) via PYNQ Clocks API
    # (full gate-off via sysfs enable not available on this kernel)
    print("\n--- Minimising Unused PL Clocks ---")
    try:
        from pynq import Clocks
        for i in range(4):
            if i == 0 and keep_clock0:
                continue
            try:
                setattr(Clocks, f'fclk{i}_mhz', 1)
                actual = getattr(Clocks, f'fclk{i}_mhz')
                print(f"  PL Clock {i}: set to {actual:.1f} MHz (minimum)")
            except Exception as e:
                print(f"  PL Clock {i}: cannot minimise ({e})")
    except ImportError:
        print("  PYNQ not available. Cannot control PL clocks.")


def enable_pl_clocks(freq_mhz: int = 100):
    # restores all 4 PL clocks to a given frequency
    print(f"\n--- Restoring PL Clocks to {freq_mhz} MHz ---")
    try:
        from pynq import Clocks
        for i in range(4):
            try:
                setattr(Clocks, f'fclk{i}_mhz', freq_mhz)
                actual = getattr(Clocks, f'fclk{i}_mhz')
                print(f"  PL Clock {i}: {actual:.1f} MHz")
            except Exception as e:
                print(f"  PL Clock {i}: error ({e})")
    except ImportError:
        print("  PYNQ not available. Cannot control PL clocks.")


def get_pl_clock_info():
    print("\n--- PL Clock Info ---")
    try:
        from pynq import Clocks
        for i in range(4):
            try:
                freq = getattr(Clocks, f'fclk{i}_mhz')
                print(f"  PL Clock {i}: {freq:.1f} MHz")
            except Exception:
                pass
    except ImportError:
        print("  PYNQ not available. Cannot read PL clocks.")


def disable_unused_peripherals():
    # disables BT and sets DP to auto-power (skips WiFi for SSH)
    print("\n--- Disabling Unused Peripherals ---")

    bt_path = "/sys/class/rfkill/rfkill0/state"
    if os.path.exists(bt_path):
        write_sysfs(bt_path, "0")
        print("  Bluetooth: DISABLED (rfkill)")
    else:
        print("  Bluetooth: not found")


    dp_path = "/sys/class/drm/card0/device/power/control"
    if os.path.exists(dp_path):
        write_sysfs(dp_path, "auto")
        print("  DisplayPort: set to 'auto' power")


def enable_all_peripherals():
    print("\n--- Re-enabling Peripherals ---")
    bt_path = "/sys/class/rfkill/rfkill0/state"
    if os.path.exists(bt_path):
        write_sysfs(bt_path, "1")
        print("  Bluetooth: ENABLED")


def read_power_watts() -> float:
    # reads TOTAL board power by summing ALL power rails from IRPS5401 PMICs
    # Ultra96-V2 has 2x IRPS5401 (hwmon0, hwmon1), each with 6 channels
    # hwmon sysfs power*_input is in microwatts
    hwmon_base = "/sys/class/hwmon"
    if not os.path.isdir(hwmon_base):
        return -1.0
    total_uw = 0
    found = False
    for hwmon in sorted(os.listdir(hwmon_base)):
        hwmon_path = os.path.join(hwmon_base, hwmon)
        if not os.path.isdir(hwmon_path):
            continue
        # only read from IRPS5401 PMICs (skip iio_hwmon etc)
        name_path = os.path.join(hwmon_path, "name")
        if os.path.exists(name_path):
            name = read_sysfs(name_path)
            if "irps5401" not in name:
                continue
        # sum V * I for each channel (more reliable than power*_input on some FW)
        idx = 1
        while True:
            v_path = os.path.join(hwmon_path, f"in{idx}_input")
            i_path = os.path.join(hwmon_path, f"curr{idx}_input")
            if not os.path.exists(v_path) or not os.path.exists(i_path):
                break
            try:
                v_mv = int(read_sysfs(v_path))
                i_ma = int(read_sysfs(i_path))
                total_uw += v_mv * i_ma  # mV * mA = uW
                found = True
            except ValueError:
                pass
            idx += 1
    if found:
        return total_uw / 1_000_000.0
    return -1.0


def get_power_info():
    # prints per-rail breakdown and total board power
    print("\n--- Power Consumption (per rail) ---")
    hwmon_base = "/sys/class/hwmon"
    total_mw = 0
    if not os.path.isdir(hwmon_base):
        print("  No hwmon found.")
        return -1.0
    for hwmon in sorted(os.listdir(hwmon_base)):
        hwmon_path = os.path.join(hwmon_base, hwmon)
        if not os.path.isdir(hwmon_path):
            continue
        name_path = os.path.join(hwmon_path, "name")
        if os.path.exists(name_path):
            name = read_sysfs(name_path)
            if "irps5401" not in name:
                continue
        else:
            continue
        print(f"  {hwmon} ({read_sysfs(name_path)}):")
        idx = 1
        while True:
            v_path = os.path.join(hwmon_path, f"in{idx}_input")
            i_path = os.path.join(hwmon_path, f"curr{idx}_input")
            if not os.path.exists(v_path) or not os.path.exists(i_path):
                break
            try:
                v_mv = int(read_sysfs(v_path))
                i_ma = int(read_sysfs(i_path))
                p_mw = v_mv * i_ma / 1000.0
                label_path = os.path.join(hwmon_path, f"in{idx}_label")
                label = read_sysfs(label_path) if os.path.exists(label_path) else f"rail{idx}"
                print(f"    {label:<12} {v_mv:>5} mV x {i_ma:>5} mA = {p_mw:>7.0f} mW")
                total_mw += p_mw
            except ValueError:
                pass
            idx += 1
    total_w = total_mw / 1000.0
    print(f"  {'─'*48}")
    print(f"  Total board power: {total_w:.2f} W  ({total_mw:.0f} mW)")
    return total_w


PROFILES = {
    'normal': {
        'description': 'Default Ultra96 state - all cores, 100MHz PL, everything enabled',
        'governor': 'ondemand',
        'cores': 4,
        'pl_freq_mhz': 100,
        'cpu_freq_mhz': None,       # let governor manage
        'restore_clocks': True,
    },
    'performance': {
        'description': 'Maximum performance - all cores, max frequency',
        'governor': 'performance',
        'cores': 4,
        'pl_freq_mhz': 100,
        'restore_clocks': True,
    },
    'balanced': {
        'description': 'Balanced power/performance',
        'governor': 'ondemand',
        'cores': 4,
        'pl_freq_mhz': 100,
    },
    'low_power': {
        'description': 'Minimum power - reduced cores and frequency',
        'governor': 'powersave',
        'cores': 2,
        'pl_freq_mhz': 25,
        'cpu_freq_mhz': 300,
        'disable_clocks': True,
    },
    'inference_only': {
        'description': 'Optimised for FPGA inference - minimal CPU and PL clocks, unused peripherals off',
        'governor': 'powersave',
        'cores': 1,
        'pl_freq_mhz': 25,
        'cpu_freq_mhz': 300,
        'disable_clocks': True,
    },
}


def apply_profile(profile_name: str):
    # applies a named power profile (governor + cores + PL clock + peripherals)
    if profile_name not in PROFILES:
        print(f"Unknown profile: {profile_name}")
        print(f"Available: {list(PROFILES.keys())}")
        return

    profile = PROFILES[profile_name]
    print(f"\n{'='*60}")
    print(f"  Applying Power Profile: {profile_name}")
    print(f"  {profile['description']}")
    print(f"{'='*60}")

    set_cpu_governor(profile['governor'])
    set_online_cores(profile['cores'])

    # set CPU frequency if specified, otherwise let governor manage
    cpu_freq = profile.get('cpu_freq_mhz')
    if cpu_freq:
        set_cpu_frequency(cpu_freq * 1000)

    set_pl_clock(profile['pl_freq_mhz'])

    if profile.get('restore_clocks'):
        enable_pl_clocks(100)
    elif profile.get('disable_clocks'):
        disable_pl_clocks(keep_clock0=True)

    if profile_name in ('low_power', 'inference_only'):
        disable_unused_peripherals()
    else:
        enable_all_peripherals()

    print(f"\n  Profile '{profile_name}' applied.")
    get_system_status()


def get_system_status():
    print(f"\n{'='*60}")
    print("  SYSTEM POWER STATUS")
    print(f"{'='*60}")
    get_cpu_info()
    get_pl_clock_info()
    get_power_info()



def main():
    parser = argparse.ArgumentParser(description="Ultra96 Power Management")
    parser.add_argument("--mode", choices=list(PROFILES.keys()), help="Apply a power profile")
    parser.add_argument("--governor", choices=AVAILABLE_GOVS, help="Set CPU governor directly")
    parser.add_argument("--cpu_freq", type=int, help="Set CPU frequency in MHz")
    parser.add_argument("--cores", type=int, help="Number of CPU cores to keep online (1-4)")
    parser.add_argument("--pl_freq", type=int, help="Set PL clock 0 frequency in MHz")
    parser.add_argument("--pl_clocks", choices=["disable", "enable"],
                        help="Disable (set to ~1 MHz) or enable (restore to 100 MHz) PL clocks 1-3")
    parser.add_argument("--status", action="store_true", help="Print current power status")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        apply_profile('inference_only')
        return

    if args.status:
        get_system_status()
        return

    if args.mode:
        apply_profile(args.mode)
        return

    if args.governor:
        set_cpu_governor(args.governor)
    if args.cpu_freq:
        set_cpu_frequency(args.cpu_freq * 1000)
    if args.cores:
        set_online_cores(args.cores)
    if args.pl_freq:
        set_pl_clock(args.pl_freq)
    if args.pl_clocks == "disable":
        disable_pl_clocks(keep_clock0=True)
    elif args.pl_clocks == "enable":
        enable_pl_clocks(100)

    get_system_status()


if __name__ == "__main__":
    main()
