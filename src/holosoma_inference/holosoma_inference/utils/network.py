"""Network interface auto-detection for robot communication."""

from __future__ import annotations

import os

import netifaces as ni

_SKIP_PREFIXES = ("lo", "wl", "docker", "br-", "veth", "virbr", "vnet", "tun", "tap")
_EXPECTED_PREFIXES_BY_ROBOT = {
    "g1": ("192.168.123.",),
    "h1": ("192.168.123.",),
    "h1_2": ("192.168.123.",),
    "go2": ("192.168.123.",),
    "t1": ("192.168.10.",),
}


def _interface_is_up(ifname: str) -> bool:
    try:
        with open(f"/sys/class/net/{ifname}/operstate") as f:
            return f.read().strip().lower() == "up"
    except OSError:
        return False


def _ipv4_addrs(ifname: str) -> list[str]:
    try:
        entries = ni.ifaddresses(ifname).get(ni.AF_INET, [])
    except ValueError:
        return []
    addrs: list[str] = []
    for entry in entries:
        addr = str(entry.get("addr", "")).strip()
        if addr:
            addrs.append(addr)
    return addrs


def validate_robot_interface(interface: str, robot_family: str | None = None) -> tuple[bool, str]:
    """Best-effort validation for a user-selected robot interface."""
    addrs = _ipv4_addrs(interface)
    if not addrs:
        return False, f"interface '{interface}' has no IPv4 address"

    family = str(robot_family or "").lower()
    expected_prefixes = _EXPECTED_PREFIXES_BY_ROBOT.get(family, ())
    if not expected_prefixes:
        return True, f"interface '{interface}' IPv4={addrs}"

    for addr in addrs:
        if any(addr.startswith(prefix) for prefix in expected_prefixes):
            return True, f"interface '{interface}' IPv4={addr} matches expected subnet for {family}"

    return (
        False,
        f"interface '{interface}' IPv4={addrs} does not match expected subnet(s) "
        f"{expected_prefixes} for robot family '{family}'",
    )


def detect_robot_interface(robot_family: str | None = None) -> str:
    """Return the wired NIC that is UP and best matches the robot's expected subnet."""
    candidates: list[tuple[str, list[str]]] = []
    for ifname in sorted(os.listdir("/sys/class/net/")):
        if any(ifname.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if not _interface_is_up(ifname):
            continue
        addrs = _ipv4_addrs(ifname)
        if not addrs:
            continue
        candidates.append((ifname, addrs))

    family = str(robot_family or "").lower()
    expected_prefixes = _EXPECTED_PREFIXES_BY_ROBOT.get(family, ())
    if expected_prefixes:
        matches = [
            (ifname, addr)
            for ifname, addrs in candidates
            for addr in addrs
            if any(addr.startswith(prefix) for prefix in expected_prefixes)
        ]
        if len(matches) == 1:
            ifname, addr = matches[0]
            print(
                f"[network] auto-detected interface for {family}: {ifname} "
                f"(IPv4 {addr}, expected subnet {expected_prefixes})"
            )
            return ifname
        if len(matches) > 1:
            names = sorted({ifname for ifname, _ in matches})
            raise RuntimeError(
                f"Multiple UP interfaces match robot family '{family}' and subnet {expected_prefixes}: {names}. "
                "Please set --task.interface explicitly."
            )
        raise RuntimeError(
            f"No UP wired interface matches robot family '{family}' and subnet {expected_prefixes}. "
            f"Found candidates: {candidates or 'none'}. Please set --task.interface explicitly."
        )

    if len(candidates) == 1:
        ifname, addrs = candidates[0]
        print(f"[network] auto-detected interface: {ifname} (IPv4={addrs})")
        return ifname
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple UP wired interfaces found {candidates}. Please set --task.interface explicitly."
        )
    print("[network] no wired NIC found, falling back to loopback (lo)")
    return "lo"
