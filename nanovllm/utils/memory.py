import os
import torch
from pynvml import *
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def get_gpu_memory():
    torch.cuda.synchronize()
    nvmlInit()
    visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
    cuda_device_idx = torch.cuda.current_device()
    cuda_device_idx = visible_device[cuda_device_idx]
    handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total
    used_memory = mem_info.used
    free_memory = mem_info.free
    nvmlShutdown()
    return total_memory, used_memory, free_memory

def get_gpu_info_panel() -> Panel:
    """获取一个包含 GPU 硬件信息的 rich Panel。"""
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("ID", justify="left", style="cyan")
    table.add_column("GPU Name", justify="left", style="magenta", no_wrap=True)
    table.add_column("Memory", justify="center", style="green")
    table.add_column("Pwr", justify="center", style="yellow")
    table.add_column("Temp", justify="center", style="red")

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            return Panel(Text("No NVIDIA GPU found.", justify="center"), title="Hardware Monitor", border_style="dim")

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            mem_used_mib = mem_info.used / 1024**2
            mem_total_mib = mem_info.total / 1024**2
            
            try:
                power_draw = f"{nvmlDeviceGetPowerUsage(handle) / 1000:.1f}W"
            except NVMLError:
                power_draw = "N/A"

            try:
                temperature = f"{nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)}°C"
            except NVMLError:
                temperature = "N/A"

            table.add_row(
                str(i),
                name,
                f"{mem_used_mib:.0f}/{mem_total_mib:.0f} MiB",
                power_draw,
                temperature
            )
        
        content = table
    except NVMLError as error:
        content = Text(f"Could not get GPU info: {error}", justify="center")
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass

    return Panel(content, title="[b]Hardware Monitor[/b]", border_style="blue")


def print_gpu_info(logger):
    """使用 prettytable 打印 GPU 硬件信息 (deprecated, use get_gpu_info_panel)"""
    from prettytable import PrettyTable
    try:
        nvmlInit()
        table = PrettyTable()
        table.field_names = ["ID", "GPU Name", "Driver", "Memory (Used/Total)", "Power (W)", "Temp (°C)"]
        table.align = "l"

        device_count = nvmlDeviceGetCount()
        if device_count == 0:
            logger.info("No NVIDIA GPU found.")
            return

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            driver_version = nvmlSystemGetDriverVersion()
            
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            mem_used = f"{mem_info.used / 1024**2:.2f}"
            mem_total = f"{mem_info.total / 1024**2:.2f}"
            
            try:
                power_draw = f"{nvmlDeviceGetPowerUsage(handle) / 1000:.2f}"
            except NVMLError:
                power_draw = "N/A" # Some GPUs don't support this.

            try:
                temperature = str(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU))
            except NVMLError:
                temperature = "N/A"

            table.add_row([
                i,
                name,
                driver_version,
                f"{mem_used} / {mem_total} MiB",
                power_draw,
                temperature
            ])
        
        logger.info("GPU Hardware Information:\n" + str(table))

    except NVMLError as error:
        logger.warning(f"Could not print GPU info: {error}. Is nvidia-smi working?")
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass # Already shutdown or failed to init