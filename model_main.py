import pandas as pd
from run_model_issue_binary import run_model_issue_binary

def vul_awareness_settings(owner, repository, option):
    if option == "issue":
        file_PATH = f"data/dataset_{owner}_{repository}_issue.csv"
        all_preds = run_model_issue_binary(file_PATH)
    elif option == "commit":
        file_PATH = f"data/dataset_{owner}_{repository}_commit.csv"
    else:
        raise ValueError("The option does not meet the requirements.")
