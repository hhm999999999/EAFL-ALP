from loguru import logger
import sys

def init_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "{name}:{line} - <level>{message}</level>",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
