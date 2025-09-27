# /path/to/your/test_scripts/check_gpu.py

import sys
import subprocess
import os

def run_command(command):
    """Helper to run shell commands and return output or error."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False, # Don't raise an exception for non-zero exit codes
            shell=True # Needed for commands like 'which' or if command itself has spaces/pipes
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, f"Exception running '{command}': {e}"

def check_pytorch_cuda():
    """Checks PyTorch's CUDA availability and version."""
    print("--- PyTorch CUDA Diagnostics ---")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("CUDA available to PyTorch: YES ✅")
            print(f"PyTorch's detected CUDA version: {torch.version.cuda}")
            try:
                # Attempt to create a CUDA tensor to confirm functionality
                tensor = torch.ones(1, device="cuda")
                print("Successfully created a CUDA tensor. PyTorch GPU is functional. 🎉")
            except Exception as e:
                print(f"Failed to create a CUDA tensor: {e}. PyTorch might detect CUDA but can't use it.")
        else:
            print("CUDA available to PyTorch: NO ❌")
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"PyTorch built with CUDA version: {torch.version.cuda}, but cannot access it at runtime.")
            else:
                print("PyTorch is likely a CPU-only build.")

    except ImportError:
        print("PyTorch is not installed or not found in current environment. 🚫")
    except Exception as e:
        print(f"An unexpected error occurred during PyTorch check: {e}")
    print("-" * 30)

def check_nvidia_smi():
    """Checks system-wide NVIDIA GPU and driver status via nvidia-smi."""
    print("\n--- System NVIDIA Driver Diagnostics ---")
    success, output = run_command("nvidia-smi")
    if success:
        print("nvidia-smi found and executed successfully. System detects NVIDIA GPU. ✅")
        # Print first few lines to keep output concise but informative
        print("nvidia-smi output (first 10 lines):")
        print("\n".join(output.split('\n')[:10]))
    else:
        print(f"nvidia-smi not found or failed to execute. ❌\nError: {output}")
        print("This often means NVIDIA drivers are not installed or not in PATH.")
    print("-" * 30)

def check_env_vars():
    """Checks relevant environment variables."""
    print("\n--- Environment Variables Check ---")
    relevant_vars = [
        "LD_LIBRARY_PATH",
        "NIX_LD_LIBRARY_PATH", # Specific to NixOS nix-ld
        "PATH",
        "CUDA_PATH",
        "CUDA_HOME",
        "XDG_DATA_DIRS", # Relevant for finding .desktop files/icons
        "QT_QPA_PLATFORM",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "__EGL_VENDOR_LIBRARY_JSON_FILE", # Crucial for NVIDIA with Wayland
        "__GLX_VENDOR_LIBRARY_NAME",      # Crucial for NVIDIA with X11/XWayland
    ]
    found_any = False
    for var in relevant_vars:
        if var in os.environ:
            print(f"  {var}: {os.environ[var]}")
            found_any = True
    if not found_any:
        print("No critical GPU-related environment variables found.")
    print("-" * 30)


if __name__ == "__main__":
    check_pytorch_cuda()
    check_nvidia_smi()
    check_env_vars()
