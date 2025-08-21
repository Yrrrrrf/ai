# ~/Documents/Lab/ai/flake.nix
{
  description = "A development environment for PyTorch with CUDA support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "pytorch-cuda-devshell";

          buildInputs = with pkgs; [
            # The CUDA toolkit
            cudatoolkit

            # The NVIDIA driver libraries
            linuxPackages.nvidia_x11

            # The C++ standard library, required by PyTorch's C++ backend
            stdenv.cc.cc.lib

            # --- THE FIX: The GL Vendor-Neutral Dispatch library ---
            # This is crucial for pre-built binaries to find and load the GPU driver.
            libglvnd

            # A system Python interpreter
            python3

            # uv for convenience
            uv
          ];

          shellHook = ''
            echo ""
            echo "==============================================="
            echo "  Welcome to the PyTorch + CUDA Dev Shell!   "
            echo "==============================================="
            echo ""
            echo "Environment is ready. To use your project's dependencies, run:"
            echo "  > source .venv/bin/activate"
            echo ""
          '';
        };
      }
    );
}
