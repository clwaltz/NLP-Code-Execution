import subprocess
import re
import sys
import tempfile
import os
import textwrap


ADAPTER_DIR = "data_collection_2/lora_runs/run_253531/adapter"
BASE_CMD = [
    "py",
    "lora_interact.py",
    "--adapter_dir",
    ADAPTER_DIR,
]


def clean_code(code: str) -> str:
    forced_imports = [
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns"
    ]

    code = code.split('GENERATED CODE:')[1].split('Enter question (q to exit):')[0].strip()

    lines = code.splitlines()

    cleaned = []
    seen_imports = set()

    for line in lines:
        if "import pandas as pd" in line:
            seen_imports.add("pd")
            continue
        if "import matplotlib.pyplot as plt" in line:
            seen_imports.add("plt")
            continue

        cleaned.append(line)

    code = code.replace("plt.show()", "plt.savefig('output_plot.png')\nprint('\\n[Plot saved as output_plot.png]')\nos.startfile('output_plot.png')")

    final = forced_imports + cleaned
    return "\n".join(final)


def run_model(question: str, verbose=False):
    proc = subprocess.Popen(
        BASE_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out, _ = proc.communicate(question + "\n")

    code = out.strip()

    return code


def execute_code(code: str):
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    local_vars = {}

    try:
        exec(code, {"pd": __import__("pandas"), "plt": plt}, local_vars)
    except Exception as e:
        raise e


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("=== LORA INERACTIVE ===")
    if verbose:
        print("=== VERBOSE MODE ===")

    question = input("\nEnter question: ")

    retries = 3

    for attempt in range(retries):
        print(f"\n[Attempt {attempt+1}/{retries}] Generating code...\n")

        code = run_model(question, verbose=verbose)
        if verbose:
            print("\n=== RAW GENERATED CODE ===\n")
            print(code)
        code = clean_code(code)

        try:
            execute_code(code)
            print("\nSUCCESS\n")
            return
        except Exception as e:
            print("\nERROR DURING EXECUTION:")
            print(e)
            print("\nRetrying...\n")

    print("\nFAILED after retries.")


if __name__ == "__main__":
    main()