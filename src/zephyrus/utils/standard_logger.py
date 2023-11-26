# Make logger
import logging
import os
import threading
import time
from abc import ABC

from absl import logging as absl_logging
from rich.console import Console

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")
SLURM_LOCALID = os.environ.get("SLURM_STEP_ID", "0")

"""
Clone ABSL logging functions so we can add the slurm info and have a bit more control over the logging formating etc
"""
def get_initial_for_level(level):
  """Gets the initial that should start the log line for the given level.

  It returns:
  - 'I' when: level < STANDARD_WARNING.
  - 'W' when: STANDARD_WARNING <= level < STANDARD_ERROR.
  - 'E' when: STANDARD_ERROR <= level < STANDARD_CRITICAL.
  - 'F' when: level >= STANDARD_CRITICAL.

  Args:
    level: int, a Python standard logging level.

  Returns:
    The first initial as it would be logged by the C++ logging module.
  """
  if level < absl_logging.converter.STANDARD_INFO:
      return 'D'
  if level < absl_logging.converter.STANDARD_WARNING:
    return 'I'
  elif level < absl_logging.converter.STANDARD_ERROR:
    return 'W'
  elif level < absl_logging.converter.STANDARD_CRITICAL:
    return 'E'
  else:
    return 'F'


def get_log_prefix(record):
    """Returns the absl log prefix for the log record.

    Args:
      record: logging.LogRecord, the record to get prefix for.
    """
    created_tuple = time.localtime(record.created)
    created_microsecond = int(record.created % 1.0 * 1e6)

    critical_prefix = ''
    level = record.levelno
    if absl_logging._is_non_absl_fatal_record(record):
        # When the level is FATAL, but not logged from absl, lower the level so
        # it's treated as ERROR.
        level = logging.ERROR
        critical_prefix = absl_logging._CRITICAL_PREFIX
    severity = get_initial_for_level(level)
    sev_colours = {"I": "white", "D": "green"}
    sev_colour = sev_colours.get(severity, "not dim red")

    return '[dim]\[[%s bold]%c[/%s bold] %04d-%02d-%02d [white]%02d:%02d:%02d.%06d[/white] [yellow]%s.%s.%s[/yellow] [bright_magenta italic]%s[/bright_magenta italic]:%d][/dim] %s' % (
        sev_colour,
        severity,
        sev_colour,
        created_tuple.tm_year,
        created_tuple.tm_mon,
        created_tuple.tm_mday,
        created_tuple.tm_hour,
        created_tuple.tm_min,
        created_tuple.tm_sec,
        created_microsecond,
        SLURM_JOB_ID,
        SLURM_LOCALID,
        threading.current_thread().name,
        record.filename,
        record.lineno,
        critical_prefix)

console = Console(color_system="windows", force_interactive=True, width=10_000)

class LogFormatter(logging.Formatter):
    """
    Log formatter, we use rich to format the output and add some colour
    """
    def format(self, record: logging.LogRecord) -> str:
        prefix = get_log_prefix(record)
        log = super(LogFormatter, self).format(record)
        # Use rich to add colour etc
        with console.capture() as capture:
            console.print(prefix, end="", markup=True)
            console.print(log, end="")
        str_output = capture.get()
        return str_output


formatter = LogFormatter()

# ABSL - Log Format
absl_logging.use_absl_handler()
absl_logging.get_absl_handler().setFormatter(formatter)
logger = absl_logging.get_absl_logger()

def build_logger(*args, **kwargs):
    return logger

class Logged(ABC):
    def __init__(self, name=None):
        name = name if name else __name__
        self.logger = absl_logging.get_absl_logger()

