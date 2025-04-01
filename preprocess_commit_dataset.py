import logging
import json
import pandas as pd
from setup_logging import setup_logging

def preprocess_commit_dataset(owner, repository):
    setup_logging()
    logging.info(f"Preprocessing commit dataset for {owner}/{repository}")
    with open(f"commit_details/{owner}_{repository}_commit_details.json", 'r', encoding='utf-8') as f:
        commit_data = json.load(f)

    data_list = []

    for commit in commit_data:
        sha = commit["sha"]
        message = commit["message"]
        changes = commit["changes"]
        patches = '\n'.join(
            change["patch"]
            for change in changes
        )

        data_list.append({
            "sha": sha,
            "message": message,
            "patches": patches
        })

    dataset = pd.concat([
        pd.DataFrame(data_list),
        pd.DataFrame(columns=['sha', 'message', 'patches'])
    ], ignore_index=True)

    dataset.to_csv(f"data/dataset_{owner}_{repository}_commit.csv", index=False)
    with open(f"data/dataset_{owner}_{repository}_commit.json", 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    logging.info(f"成功处理并保存 {owner}/{repository} 的 commit 数据")

if __name__ == "__main__":
    owner = "torvalds"
    repository = "linux"
    preprocess_commit_dataset(owner, repository)
