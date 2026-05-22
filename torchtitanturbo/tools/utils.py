# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import subprocess

from torchtitan.tools.logging import logger


def get_peak_flops(device_name: str) -> float:
    """NPU-patched get_peak_flops with Ascend device support."""
    try:
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")

    if "A100" in device_name:
        return 312e12
    elif "A6000" in device_name:
        return 154.85e12
    elif "H100" in device_name:
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:
            return 989e12
    elif "H200" in device_name:
        return 989e12
    elif "H20" in device_name:
        return 148e12
    elif "GB200" in device_name or "GB300" in device_name:
        return 2.5e15
    elif "B300" in device_name or "B200" in device_name:
        return 2.25e15
    elif "MI355X" in device_name:
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        return 1300e12
    elif "MI250X" in device_name:
        return 191.5e12
    elif "Ascend910B1" in device_name:
        return 373.88e12
    elif "Ascend910B2" in device_name:
        return 353.8944e12
    elif "Ascend910B3" in device_name:
        return 294.912e12
    elif "Ascend910B4" in device_name:
        return 245.76e12
    else:
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


def apply_patch():
    import torchtitan.tools.utils

    torchtitan.tools.utils.get_peak_flops = get_peak_flops
    logger.info("Patched torchtitan.tools.utils.get_peak_flops for NPU")
