import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def run_notebook(notebook_filename):
    print(f"Reading notebook: {notebook_filename}")
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print("Initializing ExecutePreprocessor...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    print("Executing notebook cells...")
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(os.path.abspath(notebook_filename))}})

    print(f"Saving executed notebook with outputs to: {notebook_filename}")
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Notebook execution completed successfully!")

if __name__ == '__main__':
    target = 'mining_predictive_maintenance.ipynb'
    run_notebook(target)
