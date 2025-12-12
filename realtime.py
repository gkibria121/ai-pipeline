# realtime.py - show a progress bar reliably in terminals and notebooks
import sys
import time

try:
    # If running inside a notebook, use the notebook-compatible tqdm
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except Exception:
    # Fallback to a simple implementation if tqdm is not available
    tqdm = None

def main(n=100, delay=0.1):
    if tqdm is None:
        for i in range(n):
            print(f"Processing: {i+1}/{n}")
            sys.stdout.flush()
            time.sleep(delay)
        return

    # Force tqdm to write to stdout (helps when stdout is captured)
    use_notebook = "ipykernel" in sys.modules
    bar = tqdm(range(n), desc="Processing", file=sys.stdout, ncols=80)
    for _ in bar:
        time.sleep(delay)
        # ensure any buffered output is flushed immediately
        try:
            sys.stdout.flush()
        except Exception:
            pass

if __name__ == "__main__":
    main()
 