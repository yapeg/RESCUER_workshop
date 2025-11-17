# Workshop 4 Collaboration Code

Small 2D advection–diffusion toy model for the **Collaborative Work on Research Code** workshop.

It is intentionally simple but realistic enough to practice:

- Git branches, forks, and pull requests
- Code review on a numerical code
- Basic testing with `pytest`

## Structure

```
code/
  computation/
    __init__.py
    solver.py        # numerical core (advection–diffusion step, boundary)
    io_utils.py      # config + CSV I/O
    plotting.py      # Matplotlib plotting helper
    run.py           # CLI entry point
  tests/
    __init__.py
    test_solver.py   # simple numerical sanity checks
  configs/
    base_config.csv  # simulation parameters
    make_ics.py      # initial condition generator
  requirements.txt
  README.md
 ```
## How to run

First, generate the desired initial condition in `test/make_ics` with the command:

```bash
python configs/make_ics.py
```

Then run the code using
```bash
python -m computation.run --config configs/base_config.csv --ic configs/base_ic.csv
```



