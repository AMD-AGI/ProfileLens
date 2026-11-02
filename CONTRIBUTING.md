# Contributing to ProfileLens

Use [GitHub Issues](https://github.com/AMD-AGI/ProfileLens/issues) for bug reports
and feature requests. Include the version or commit, Python/ROCm version, GPU,
command, expected behavior, and a small reproducer. Remove credentials and
confidential workload or trace data. Report vulnerabilities privately using
[SECURITY.md](SECURITY.md).

## Development and pull requests

Use Linux x86-64 and Python 3.10 or later, following the prerequisites in
[README.md](README.md). In an isolated Python environment, install the source
with `python -m pip install -e .`.

Create a focused branch from `main` and open a pull request against `main`.
Describe the change and its validation, update relevant documentation, and
obtain review from the [code owners](.github/CODEOWNERS). The package CI checks
syntax, builds and installs the wheel, and runs CLI help/version commands.
Profiling changes also need validation on the relevant GPU and ROCm version.

Contributions are licensed under the existing [MIT license](LICENSE.TXT).
Submit only material you have the right to contribute, and retain applicable
copyright and third-party notices.
