import functools
import json
import os
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(
    os.environ.get("TRITON_LOG_DIR", os.path.expanduser("~/.triton/logs"))
).absolute()
LOG_PATH = LOG_DIR / "debug.log"
TRITON_AOT_KERNEL_DIR = Path(
    os.environ.get("TRITON_AOT_KERNEL_DIR", os.path.expanduser("~/.triton/aot"))
).absolute()

if not TRITON_AOT_KERNEL_DIR.exists():
    TRITON_AOT_KERNEL_DIR.mkdir(parents=True, exist_ok=True)

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

open(LOG_PATH, "w").close()


def serialize(record):
    subset = {"function_info": record["extra"]}
    return json.dumps(subset)


def formatter(record):
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


logger.add(sys.stderr, level="INFO", format="{message}")
logger.add(
    LOG_PATH,
    level="DEBUG",
    # format="{extra}",  # formatter,
    backtrace=True,
    diagnose=True,
    serialize=True,
)


def trace_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with logger.contextualize(function=func.__name__):
            logger.bind(args=args).debug("args")
            logger.bind(kwargs=kwargs).debug("kwargs")
            result = func(*args, **kwargs)
            logger.debug("function.return_value", str(result))
        return result

    return wrapper
