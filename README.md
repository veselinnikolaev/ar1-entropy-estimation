# AR(1) Entropy Estimation

Bachelor Semester Project investigating the effect of observation frequency (thinning) on entropy estimation for a Gaussian AR(1) process.

## Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager

## Setup & Installation

1. Clone the repository and navigate into the project folder:
   ```bash
   git clone https://github.com/veselinnikolaev/ar1-entropy-estimation.git
   cd ar1-entropy-estimation
   ```

2. Install dependencies and set up the editable environment:
   ```bash
   uv pip install -e .
   ```

## Running the Simulation

Execute the primary simulation script using `uv`:

```bash
uv run experiments/run_simulation.py
```
