import logging
import json
import pandas as pd
from setup_logging import setup_logging
from utils import replace_tokens_simple_1

def preprocess_issue_dataset(owner, repository):
    setup_logging()
    logging.info(f"Preprocessing issue dataset for {owner}/{repository}")
    with open(f"issues/{owner}_{repository}_issues.json", 'r', encoding='utf-8') as f:
        issue_data = json.load(f)

    dataset = pd.DataFrame([
        {
            "issue_id": issue["url"].split('/')[-1].strip(),
            "title": issue["title"],
            "body": issue["body"]
        }
        for issue in issue_data
    ])

    print("begin process...")
    dataset["title"] = dataset["title"].map(replace_tokens_simple_1)
    dataset["body"] = dataset["body"].map(replace_tokens_simple_1)
    print("finish process...")

    dataset.to_csv(f"data/dataset_{owner}_{repository}_issue.csv", index=False)

if __name__ == "__main__":
    owner = "torvalds"
    repository = "linux"
    preprocess_issue_dataset(owner, repository)
