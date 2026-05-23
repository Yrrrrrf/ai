# /path/to/your/test_scripts/check_gpu.py

import subprocess
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Custom theme for consistent styling
custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "danger": "bold red",
        "success": "bold green",
        "header": "bold magenta",
    }
)

console = Console(theme=custom_theme)


def run_command(command):
    """Helper to run shell commands and return output or error."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=True,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, f"Exception running '{command}': {e}"


def check_pytorch_cuda():
    """Checks PyTorch's CUDA availability and version."""
    content = []
    try:
        import torch

        content.append(
            Text.assemble(("PyTorch version: ", "info"), (torch.__version__, "bold"))
        )

        if torch.cuda.is_available():
            content.append(Text("CUDA available to PyTorch: YES ✅", style="success"))
            content.append(
                Text.assemble(
                    ("PyTorch's detected CUDA version: ", "info"),
                    (torch.version.cuda, "bold"),
                )
            )
            try:
                # Attempt to create a CUDA tensor to confirm functionality
                torch.ones(1, device="cuda")
                content.append(
                    Text(
                        "Successfully created a CUDA tensor. PyTorch GPU is functional. 🎉",
                        style="success",
                    )
                )
            except Exception as e:
                content.append(
                    Text(
                        f"Failed to create a CUDA tensor: {e}. PyTorch might detect CUDA but can't use it.",
                        style="danger",
                    )
                )
        else:
            content.append(Text("CUDA available to PyTorch: NO ❌", style="danger"))
            if hasattr(torch.version, "cuda") and torch.version.cuda:
                content.append(
                    Text(
                        f"PyTorch built with CUDA version: {torch.version.cuda}, but cannot access it at runtime.",
                        style="warning",
                    )
                )
            else:
                content.append(
                    Text("PyTorch is likely a CPU-only build.", style="info")
                )

    except ImportError:
        content.append(
            Text(
                "PyTorch is not installed or not found in current environment. 🚫",
                style="danger",
            )
        )
    except Exception as e:
        content.append(
            Text(
                f"An unexpected error occurred during PyTorch check: {e}",
                style="danger",
            )
        )

    console.print(
        Panel(
            "\n".join([str(c) for c in content]),
            title="[header]PyTorch CUDA Diagnostics[/header]",
            expand=False,
        )
    )


def check_nvidia_smi():
    """Checks system-wide NVIDIA GPU and driver status via nvidia-smi."""
    success, output = run_command("nvidia-smi")
    if success:
        # Extract first 10 lines
        lines = output.split("\n")[:10]
        summary = (
            "nvidia-smi found and executed successfully. System detects NVIDIA GPU. ✅"
        )
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[header]System NVIDIA Driver Diagnostics[/header]",
                subtitle=f"[success]{summary}[/success]",
                expand=False,
            )
        )
    else:
        error_msg = f"nvidia-smi not found or failed to execute. ❌\nError: {output}\nThis often means NVIDIA drivers are not installed or not in PATH."
        console.print(
            Panel(
                error_msg,
                title="[header]System NVIDIA Driver Diagnostics[/header]",
                style="danger",
                expand=False,
            )
        )


def check_env_vars():
    """Checks relevant environment variables."""
    relevant_vars = [
        "LD_LIBRARY_PATH",
        "NIX_LD_LIBRARY_PATH",
        "PATH",
        "CUDA_PATH",
        "CUDA_HOME",
        "XDG_DATA_DIRS",
        "QT_QPA_PLATFORM",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "__EGL_VENDOR_LIBRARY_JSON_FILE",
        "__GLX_VENDOR_LIBRARY_NAME",
    ]

    table = Table(
        title="Environment Variables Check",
        title_style="header",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Variable", style="bold")
    table.add_column("Value", overflow="fold")

    found_any = False
    for var in relevant_vars:
        if var in os.environ:
            table.add_row(var, os.environ[var])
            found_any = True

    if found_any:
        console.print(table)
    else:
        console.print(
            Panel(
                "No critical GPU-related environment variables found.",
                style="info",
                expand=False,
            )
        )


def check_prime_offload():
    """Checks that PRIME offload is working: iGPU for plain glxinfo, dGPU for nvidia-offload glxinfo."""
    glxinfo_available, _ = run_command("which glxinfo")
    if not glxinfo_available:
        console.print(
            Panel(
                "glxinfo not found. Install temporarily with: [bold]nix shell nixpkgs#mesa-demos[/bold] ⚠️\nSkipping PRIME offload check.",
                title="[header]PRIME Offload Check[/header]",
                style="warning",
                expand=False,
            )
        )
        return

    table = Table(
        title="PRIME Offload Check",
        title_style="header",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Mode")
    table.add_column("Renderer")
    table.add_column("Status")

    # Plain glxinfo
    success, output = run_command(
        "glxinfo -B 2>/dev/null | grep -E 'OpenGL renderer|OpenGL vendor'"
    )
    if success:
        renderer = output.replace("\n", " ").strip()
        status = (
            Text("✅ iGPU active", style="success")
            if any(x in output.lower() for x in ["amd", "mesa"])
            else Text("❌ dGPU active (DEFEATED)", style="danger")
        )
        table.add_row("Plain glxinfo", renderer, status)
    else:
        table.add_row("Plain glxinfo", "Failed", Text(output, style="danger"))

    # nvidia-offload glxinfo
    success, output = run_command(
        "nvidia-offload glxinfo -B 2>/dev/null | grep -E 'OpenGL renderer|OpenGL vendor'"
    )
    if success:
        renderer = output.replace("\n", " ").strip()
        status = (
            Text("✅ dGPU available", style="success")
            if "NVIDIA" in output or "nvidia" in output
            else Text("⚠️ Unexpected renderer", style="warning")
        )
        table.add_row("nvidia-offload", renderer, status)
    else:
        table.add_row(
            "nvidia-offload",
            "Not available/Failed",
            Text("Try manual PRIME env vars", style="info"),
        )

    console.print(table)


def check_idle_power():
    """Reports GPU idle power draw, power cap, and utilization from nvidia-smi."""
    success, output = run_command(
        "nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu --format=csv,noheader,nounits"
    )
    if success:
        parts = [p.strip() for p in output.split(",")]
        if len(parts) == 3:
            draw, limit, util = parts
            table = Table(title="GPU Power & Utilization", title_style="header")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_column("Status")

            draw_f = float(draw)
            if draw_f <= 6.0:
                status = Text("Excellent", style="success")
            elif draw_f <= 10.0:
                status = Text("Acceptable", style="success")
            elif draw_f <= 15.0:
                status = Text("Elevated", style="warning")
            else:
                status = Text("High (Active)", style="danger")

            table.add_row("Current Power Draw", f"{draw} W", status)
            table.add_row("Power Limit", f"{limit} W", "")
            table.add_row(
                "GPU Utilization",
                f"{util} %",
                "" if float(util) < 5 else Text("Active", style="warning"),
            )

            console.print(table)
        else:
            console.print(
                Panel(
                    f"Raw output: {output}",
                    title="[header]GPU Idle Power Draw[/header]",
                    style="warning",
                )
            )
    else:
        console.print(
            Panel(
                f"nvidia-smi query failed: {output}",
                title="[header]GPU Idle Power Draw[/header]",
                style="danger",
            )
        )


def check_session_env_clean():
    """Asserts the absence of environment variables that defeat PRIME offload and leak CUDA libs."""
    in_venv = "VIRTUAL_ENV" in os.environ or _is_in_venv()

    table = Table(title="Session Environment Cleanliness", title_style="header")
    table.add_column("Check")
    table.add_column("Value / Status")
    table.add_column("Result")

    # Check __GLX_VENDOR_LIBRARY_NAME
    val = os.environ.get("__GLX_VENDOR_LIBRARY_NAME")
    table.add_row(
        "__GLX_VENDOR_LIBRARY_NAME",
        str(val),
        Text("✅ OK", style="success")
        if val is None
        else Text("❌ Set", style="danger"),
    )

    # Check __EGL_VENDOR_LIBRARY_JSON_FILE
    val = os.environ.get("__EGL_VENDOR_LIBRARY_JSON_FILE")
    table.add_row(
        "__EGL_VENDOR_LIBRARY_JSON_FILE",
        str(val),
        Text("✅ OK", style="success")
        if val is None
        else Text("❌ Set", style="danger"),
    )

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_substrings = ["cuda-merged", "cudnn", "nvidia-x11"]
    found_cuda = [s for s in cuda_substrings if s in ld_path]

    if not found_cuda:
        table.add_row(
            "LD_LIBRARY_PATH (CUDA)", "None found", Text("✅ Clean", style="success")
        )
    else:
        status_text = "⚠️ Venv leaked" if in_venv else "❌ Leaked"
        table.add_row(
            "LD_LIBRARY_PATH (CUDA)",
            ", ".join(found_cuda),
            Text(status_text, style="warning" if in_venv else "danger"),
        )

    console.print(table)


def _is_in_venv():
    """Detects if the script is running inside a Python virtual environment."""
    if hasattr(sys, "real_prefix"):
        return True
    return sys.prefix != sys.base_prefix


if __name__ == "__main__":
    with console.status("[bold green]Running GPU Diagnostics...") as status:
        check_pytorch_cuda()
        check_nvidia_smi()
        check_env_vars()
        check_prime_offload()
        check_idle_power()
        check_session_env_clean()
    console.print("\n[bold success]Diagnostics complete![/bold success]")
