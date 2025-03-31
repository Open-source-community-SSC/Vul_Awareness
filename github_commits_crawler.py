import json
import time
import logging
import requests
import datetime

"""
    1、获取最近 30 天的 commits
    2、提取 commit 的详细信息
    3、自动处理分页，确保所有 issue 都被获取到
"""

# 设置 Github 账户信息（建议使用 token 以避免 API 速率限制）
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# 设置时间间隔
time_interval = 30

# 计算最近 time_interval 天的时间戳
since_date = (datetime.datetime.utcnow() - datetime.timedelta(days=time_interval)).isoformat() + "Z"

def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,   # 设置日志级别为INFO，这样INFO及以上级别的日志都会输出
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',   # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S',   # 日期时间格式
        handlers=[
            # 同时将日志输出到文件和控制台
            logging.FileHandler('log/crawler.log'),
            logging.StreamHandler()
        ]
    )

def crawl_github_commits(owner, repository):
    # 日志
    logging.info("Crawling commits for %s/%s" % (owner, repository))

    url = f"https://api.github.com/repos/{owner}/{repository}/commits"
    params = {"since": since_date, "per_page": 100}

    # 爬取 commits
    all_commits = []
    page = 1
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print("Error: ", response.json())
            logging.error("Error fetching commits: %s of page %s" % (response.text, page))
            break
        commits = response.json()
        logging.info("Fetched %d commits of page %s" % (len(commits), page))
        all_commits.extend(commits)

        # 获取下一页 URL
        url = response.links.get("next", {}).get("url")
        logging.info("Next page: %s" % url)
        page += 1

        # 防止请求频率太高
        time.sleep(3)

    # 保存到文件
    logging.info("Saving commits to file.")
    output_file = f"commits/{owner}_{repository}_commits.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_commits, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(all_commits)} commits to {output_file}.")
    logging.info(f"Saved {len(all_commits)} commits to {output_file}.")
    logging.info("Done.")

if __name__ == "__main__":
    setup_logging()
    owner = 'torvalds'
    repository = 'linux'
    crawl_github_commits(owner, repository)
