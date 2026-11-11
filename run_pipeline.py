"""Execute the pipeline notebooks in order, stopping at the first failure.
Usage:  python run_pipeline.py            (all steps 01-10)
        python run_pipeline.py 06 10      (a contiguous range, e.g. steps 06-10)"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB = Path(__file__).resolve().parent / "notebooks"
first, last = (sys.argv[1:3] + ["01", "10"][len(sys.argv[1:3]):])
for path in sorted(p for p in NB.glob("[0-9][0-9]_*.ipynb") if first <= p.name[:2] <= last):
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    except Exception as e:
        print(f"FAILED  {path.name}: {type(e).__name__}"); raise SystemExit(1)
    nbformat.write(nb, path)
    print(f"ok      {path.name}  ({time.time() - t0:.0f}s)")
print("pipeline complete")
