let
  # use a specific (although arbitrarily chosen) version of the Nix package collection
  default_pkgs = fetchTarball {
    url = "http://github.com/NixOS/nixpkgs/archive/nixpkgs-unstable.tar.gz";
    # the sha256 makes sure that the downloaded archive really is what it was when this
    # file was written
    sha256 = "0x5j9q1vi00c6kavnjlrwl3yy1xs60c34pkygm49dld2sgws7n0a";
  };
in { pkgs ? import default_pkgs { } }:
with pkgs;
let
  pythonBundle = python39.withPackages
    (ps: with ps; [ tensorflow-probability matplotlib numpy ipython ]);
in mkShell { buildInputs = [ pythonBundle ]; }
