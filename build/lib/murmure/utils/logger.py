import logging 

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

