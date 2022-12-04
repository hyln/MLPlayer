import logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

console_log = logging.StreamHandler()
console_log.setLevel(logging.DEBUG)

# file_log = logging.FileHandler("access.log")
# file_log.setLevel(logging.INFO)

formatter_console = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
formatter_file = logging.Formatter(
    "%(asctime)s - %(filename)s "
    "- %(funcName)s- %(lineno)d- "
    "-%(levelname)s - %(message)s"
)

console_log.setFormatter(formatter_console)
# file_log.setFormatter(formatter_file)

logger.addHandler(console_log)
# logger.addHandler(file_log)
