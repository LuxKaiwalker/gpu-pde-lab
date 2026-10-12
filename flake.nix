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

        echo "GPU PDE Lab dev environment ready. Python: $(python3 --version)"
      '';
    };
  };
}
