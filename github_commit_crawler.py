import os
import json
import time
import logging
import requests
from setup_logging import setup_logging

"""
    1、遍历 commits 清单，
    2、获取 commit message 和 patch
    3、保存到 json
"""

# 设置 Github 账户信息（建议使用 token 以避免 API 速率限制）
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def get_commit_details(sha, url):
    """
    获取 commit 详情
    :param sha: 
    :param url: 
    :return: 
    """
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        logging.error("Error when fetching commit %s details: %s" % (sha, response.text))
        print("Error when fetching commit %s details: %s" % (sha, response.text))
        return None

    data = response.json()
    message = data["commit"]["message"]
    files = data.get('files', [])

    patches = []
    for file in files:
        if "patch" in file:   # 只有文本文件才会有 patch
            patches.append({
                "filename": file["filename"],
                "patch": file["patch"]
            })

    logging.info("Commit %s details fetched successfully." % sha)
    return {"sha": sha, "message": message, "changes": patches}

def crawl_github_commit(owner, repository):
    """
    爬取 github 仓库的 commit 详情
    :param owner: 
    :param repository: 
    :return: 
    """
    setup_logging()
    # 日志
    logging.info("Crawling github commit details data of %s/%s" % (owner, repository))
    logging.info("Opening commits file of %s/%s" % (owner, repository))
    if not os.path.exists(f"commits/{owner}_{repository}_commits.json"):
        logging.error(f"文件 'commits/{owner}_{repository}_commits.json' 不存在！ 请先爬取 commits。")
        print(f"文件 'commits/{owner}_{repository}_commits.json' 不存在！ 请先爬取 commits。")
        return

    with open(f"commits/{owner}_{repository}_commits.json", 'r') as f:
        all_commits = json.load(f)
    logging.info("Commits file opened.")
    logging.info("Total commits: %d" % len(all_commits))

    logging.info("Now crawling commits...")

    commit_details = []
    for i, commit in enumerate(all_commits):
        sha = commit['sha']
        url = commit['url']
        details = get_commit_details(sha, url)
        logging.info(f"Index {i}: now fetching commit {sha} details...")
        if details:
            commit_details.append(details)
            logging.info(f"Index {i}: commit {sha} details fetched successfully.")

        if (i + 1) % 5 == 0:   # 每 5 次请求暂停 10 秒，避免触发 GitHub API 限制
            time.sleep(10)

    # 保存结果
    with open(f"commit_details/{owner}_{repository}_commit_details.json", 'w', encoding='utf-8') as f:
        json.dump(commit_details, f, ensure_ascii=False, indent=4)
    logging.info(f"All commit details saved to file commit_details/{owner}_{repository}_commit_details.json.")

if __name__ == '__main__':
    owner = "torvalds"
    repository = "linux"
    crawl_github_commit(owner, repository)
