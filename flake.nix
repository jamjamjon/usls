{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = {nixpkgs, ...}: {
    devShell = nixpkgs.lib.genAttrs (nixpkgs.legacyPackages.x86_64-linux.onnxruntime.meta.platforms) (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in
      pkgs.mkShell {
        packages = with pkgs; [onnxruntime openssl pkg-config rustPlatform.bindgenHook];
      });
  };
}
