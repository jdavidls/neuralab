from rich import logging, console
from logging import getLogger as get_logger, basicConfig as log_config

console = console.Console()

log_config(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.RichHandler(console=console, show_path=False)],
)
