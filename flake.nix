# GPU PDE Lab development environment.
#
# This flake deliberately exposes a devShell and nothing else -- no
# `packages.default`, no `checks`, so `nix build` and `nix flake check` do
# nothing here. That is a constraint, not an oversight, and it is worth
# writing down because the obvious improvement is a trap.
#
# A hermetic package would have to take torch from nixpkgs rather than pip.
# At the pinned nixpkgs (nixos-25.05, ac62194c3917) that is not a cached
# path: `nix build --dry-run nixpkgs#python311Packages.torch` reports
# python3.11-torch-2.7.0 and python3.11-triton-3.1.0 as "will be built",
# meaning compiling PyTorch and Triton from source -- hours of CPU per
# evaluation, on every machine, for a dependency pip fetches as a wheel in
# under a minute. Everything else the project needs (pyside6, pyqtgraph,
# matplotlib, numpy, pytest) is already cached; torch alone is the blocker.
#
# So the division is: nix supplies the things that are painful to get any
# other way on NixOS (a working PySide6/Qt, the CUDA libraries on
# LD_LIBRARY_PATH, the Triton libcuda override below), and pip supplies the
# Python packages into a venv. CI and the Dockerfile are where the install
# gets verified reproducibly.
{
  description = "GPU PDE Lab development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };

    pythonForVenv = pkgs.python311.withPackages (ps: [
      ps.pyside6
    ]);

  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonForVenv
        pkgs.python311Packages.pip
        pkgs.python311Packages.virtualenv

        pkgs.cudaPackages.cudatoolkit
        pkgs.cudaPackages.cudnn

        pkgs.gcc.cc.lib
        pkgs.git
        pkgs.zlib
        pkgs.libGL
        pkgs.fontconfig
        pkgs.glib
      ];

      shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
        export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}

        export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.fontconfig.lib}/lib:${pkgs.glib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH

        if [ ! -d .venv ]; then
          echo "Creating venv (with access to nix-provided PySide6)..."
          ${pythonForVenv}/bin/python3 -m venv --system-site-packages .venv
        fi
        source .venv/bin/activate

        # The nixpkgs-provided PySide6/Qt build aborts on this machine's native
        # Wayland (Hyprland) backend, even though the session sets
        # QT_QPA_PLATFORM=wayland globally; XWayland (xcb) works reliably, so
        # force it here. Override manually after entering the shell if needed.
        export QT_QPA_PLATFORM=xcb

        # Triton locates libcuda.so.1 by shelling out to `/sbin/ldconfig -p`,
        # which does not exist on NixOS -- so `import triton` raises
        # FileNotFoundError before it ever looks at the GPU. The documented
        # override skips that probe entirely. The NVIDIA driver's libraries
        # live under /run/opengl-driver/lib on NixOS (already on
        # LD_LIBRARY_PATH above for PyTorch's own sake).
        export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

        echo "GPU PDE Lab dev environment ready. Python: $(python3 --version)"
      '';
    };
  };
}
