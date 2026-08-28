# Agent instructions

Before modifying this repository, read `DEPENDENCY_AUDIT.md` and `PATCHES.md`.

- TorchTitanTurbo owns NPU-specific patches, kernels, graph compatibility, and
  profiling adapters. Do not modify TorchTitan framework code as a workaround.
- Resolve every patched upstream symbol against the current source-installed
  TorchTitan checkout and audit all torchtitan-test consumers listed in
  `DEPENDENCY_AUDIT.md`.
- Copied or overridden algorithms require behavioral contract tests against
  current upstream semantics.
- Do not silently swallow unexpected import or patch failures.
- Do not commit unless the user explicitly requests a commit.
