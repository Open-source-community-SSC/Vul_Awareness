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

    data_list = []

    for issue in issue_data:
        issue_id = issue["url"].split('/')[-1].strip()
        title = issue["title"]
        body = issue["body"]

        data_list.append({
            "issue_id": issue_id,
            "title": title,
            "body": body if body is not None else ""
        })

    dataset = pd.concat([
        pd.DataFrame(data_list),
        pd.DataFrame(columns=['issue_id', 'title', 'body'])
    ], ignore_index=True)

    print("begin process...")
    dataset["title"] = dataset["title"].map(replace_tokens_simple_1)
    dataset["body"] = dataset["body"].map(replace_tokens_simple_1)
    print("finish process...")

    dataset.to_csv(f"data/dataset_{owner}_{repository}_issue.csv", index=False)
    with open(f"data/dataset_{owner}_{repository}_issue.json", 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    logging.info(f"成功处理并保存 {owner}/{repository} 的 issue report 数据")

if __name__ == "__main__":
    owner = "torvalds"
    repository = "linux"
    preprocess_issue_dataset(owner, repository)
