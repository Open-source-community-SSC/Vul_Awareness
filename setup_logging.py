import logging

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
