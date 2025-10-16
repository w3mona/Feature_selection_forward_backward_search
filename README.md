```markdown
# Feature Selection — Forward & Backward Search

A small C++ command-line tool implementing greedy wrapper-style feature-selection algorithms (forward selection and backward elimination), evaluated with Leave-One-Out cross-validation using a 1-Nearest-Neighbor classifier.

Repository: https://github.com/w3mona/Feature_selection_forward_backward_search

## Overview

This project provides an educational and practical implementation of wrapper-style feature selection in idiomatic C++. The program implements:

- Greedy forward selection
- Greedy backward elimination
- Evaluation using Leave-One-Out Cross-Validation (LOO-CV) with a 1‑Nearest‑Neighbor (1‑NN) classifier
- Simple data ingestion supporting whitespace- or comma-separated numeric files
- Per-feature z-score normalization (class label expected in column 0)
- A minimal command-line interface (filename + method choice)
- Basic parallelization of candidate evaluation using `std::thread`

The code is intended as a compact and readable reference implementation that you can extend for experiments, benchmarking, or integration into larger workflows.

## What this repo contains

A single C++ source that performs:
- Data reading/parsing
- Normalization
- Forward-selection and backward-elimination search
- Scoring via LOO-CV using 1-NN
- Simple console prompts for input filename and search choice

> Class label is assumed to be the first value on each row (column 0). Remaining columns are feature values.

## Build & Run

### Requirements
- A C++17-compatible compiler (g++, clang++)
- Standard C++ library (no external dependencies required)

### Build (single-file compile)
If the repository contains a single source file (for example `main.cpp`), compile with:

```bash
g++ -std=c++17 -O2 -pthread -o feature_selection main.cpp
```

Replace `main.cpp` with the actual source filename if different.

### Run

```bash
./feature_selection
```

The program will prompt for a filename and then ask you to choose:
1. Forward Selection
2. Backward Elimination

## Input data format

- Each row is one sample.
- The first value on each row is the class label (integer or numeric).
- Subsequent values are feature values (numeric).
- Values may be separated by whitespace or commas. Non-numeric cells are skipped with a warning.

Example (CSV or whitespace-separated):

```
1,5.1,3.5,1.4,0.2
0,6.2,3.4,5.4,2.3
```

## Implementation notes

- Scoring: Leave-One-Out CV combined with a 1‑Nearest‑Neighbor classifier is used to estimate accuracy for any candidate feature subset.
- Normalization: Feature columns (columns 1..) are z-score normalized across samples before any search to ensure features are comparable.
- Parallelism: Candidate evaluation is parallelized with a small custom `parallel_for` built on `std::thread` and mutexes to evaluate multiple candidate features concurrently.
- Data parsing: The reader replaces commas with spaces and attempts `stod` conversions, emitting warnings for invalid or out-of-range cells.

## Limitations (what this implementation does NOT currently provide)

To keep documentation accurate, note these items are not in the current code unless you add them:

- No unit tests or CI configuration included by default.
- No dedicated benchmarking scripts that quantify speedups from parallelism.
- The scoring function is fixed to LOO-CV + 1‑NN (not exposed as a pluggable interface).
- No specialized bit-level optimizations (no `std::bitset`-based search) and no OpenMP pragmas.
- No caching/memoization of distances across LOO folds is implemented.

## Suggestions to strengthen the project

- Add a `CMakeLists.txt` or `Makefile` to simplify building across platforms.
- Add unit tests (e.g., GoogleTest) and a GitHub Actions workflow for CI.
- Add benchmark scripts to measure runtime and compare single-threaded vs parallel runs.
- Make the scoring function pluggable so users can evaluate subsets with different classifiers/metrics (accuracy, AUC, MSE).
- Add example datasets and a short tutorial in the README showing a sample run and expected output.
- Add a LICENSE file (e.g., MIT) to clarify reuse terms.

## Contributing

Contributions are welcome. If you add features (tests, benchmarks, a pluggable scoring interface, build tooling), please update the README to reflect them and open a pull request.

## License

This repository is currently unlicensed. If you want the project to be open-source under a standard license, add a `LICENSE` file (e.g., MIT, Apache-2.0).

## Contact

Repo: https://github.com/w3mona/Feature_selection_forward_backward_search
```
