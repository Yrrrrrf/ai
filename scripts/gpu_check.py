# scripts/gpu_check.py
# A simple script to test GPU, driver version, and CUDA compatibility using PyTorch.

import subprocess
import torch

# Define some colors for clean output, similar to your other scripts
HEADER = "\033[1;32m"
INFO = "\033[1;34m"
SUCCESS = "\033[92m"
ERROR = "\033[91m"
RESET = "\033[0m"


def print_header(title: str):
    """Prints a styled header."""
    print(f"\n{HEADER}--- {title} ---{RESET}")


def check_system_gpu():
    """
    Uses the nvidia-smi command to check for the system-level GPU and driver.
    This tells us what the operating system can see.
    """
    print_header("System-Level GPU Check")
    try:
        # Execute nvidia-smi to get driver version, GPU name, and the CUDA version it supports
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        driver, name, cuda = result.stdout.strip().split(", ")
        print(f"  {INFO}Detected GPU:{RESET} {name}")
        print(f"  {INFO}NVIDIA Driver Version:{RESET} {driver}")
        print(f"  {INFO}System CUDA Version:{RESET} {cuda}")
        print(f"{SUCCESS} nvidia-smi found and executed successfully.{RESET}")

    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"{ERROR}nvidia-smi command not found or failed to execute.{RESET}")
        print(
            "  Please ensure NVIDIA drivers are installed and 'nvidia-smi' is in your system's PATH."
        )
    except Exception as e:
        print(f"{ERROR}An unexpected error occurred: {e}{RESET}")


def check_pytorch_cuda():
    """
    Checks if PyTorch can access the CUDA-enabled GPU.
    This tells us what your Python environment can see.
    """
    print_header("PyTorch CUDA Compatibility Check")
    print(f"  {INFO}PyTorch Version:{RESET} {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  {SUCCESS}SUCCESS:{RESET} PyTorch can access the CUDA-enabled GPU.")
        gpu_count = torch.cuda.device_count()
        print(f"  {INFO}CUDA Devices Found:{RESET} {gpu_count}")
        for i in range(gpu_count):
            print(f"    - {INFO}Device {i}:{RESET} {torch.cuda.get_device_name(i)}")
        print(f"  {INFO}PyTorch CUDA Version:{RESET} {torch.version.cuda}")
    else:
        print(f"{ERROR}FAILURE:{RESET} PyTorch cannot find a compatible CUDA-enabled GPU.")
        print("  This might be because:")
        print("    - Your NVIDIA drivers are not installed correctly.")
        print("    - You installed the CPU-only version of PyTorch.")
        print(
            "    - The CUDA version PyTorch was built with is incompatible with your driver."
        )


def main():
    """Main function to run the diagnostics."""
    print("\033[H\033[J", end="")  # Clear the screen
    print(f"{HEADER}GPU and CUDA Diagnostics Tool{RESET}")

    check_system_gpu()
    check_pytorch_cuda()

    print(f"\n{HEADER}Check complete.{RESET}")


if __name__ == "__main__":
    main()
