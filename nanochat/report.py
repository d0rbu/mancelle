"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import os
import re
import subprocess
import socket
import datetime
import platform
import argparse
from dataclasses import dataclass, field

import psutil
import torch

from loguru import logger
from nanochat.common import get_base_dir, get_dist_info


@dataclass(frozen=True)
class GitInfo:
    commit: str
    branch: str
    dirty: bool
    message: str


@dataclass(frozen=True)
class GPUInfo:
    available: bool
    count: int = 0
    names: list[str] = field(default_factory=list)
    memory_gb: list[float] = field(default_factory=list)
    cuda_version: str = ""


@dataclass(frozen=True)
class SystemInfo:
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cpu_count: int
    cpu_count_logical: int
    memory_gb: float
    user: str
    nanochat_base_dir: str
    working_dir: str


@dataclass(frozen=True)
class CostEstimate:
    hourly_rate: float
    gpu_type: str
    estimated_total: float | None


def run_command(cmd):
    """Run an external command (argv list, shell=False) and return stdout.

    SECURITY: This helper *intentionally* refuses ``shell=True`` and refuses
    string commands.  The previous implementation used ``subprocess.run(cmd,
    shell=True, ...)`` which made the helper a ready-made shell-injection
    primitive the moment any call site plumbed a config-/operator-controlled
    value into the format string.

    Parameters
    ----------
    cmd : list[str]
        Argument vector.  The first element is the program, the rest are
        arguments — each passed verbatim to the OS with NO shell
        interpretation.  Shell metacharacters (``;``, ``|``, ``$(...)``,
        backticks, ``>``/``<``) are treated as literal argv bytes.

    Returns
    -------
    str | None
        Stripped stdout on success, ``""`` if the command returned 0 with
        empty stdout, or ``None`` on a known failure (timeout, missing
        binary, non-zero exit with no stdout).

    Raises
    ------
    TypeError
        If ``cmd`` is not a list/tuple of strings.  This is deliberate: a
        string arg would suggest someone is re-introducing the shell=True
        pattern.  Unexpected errors (e.g. ``MemoryError``) propagate
        unchanged so operators notice them.
    """
    if not isinstance(cmd, (list, tuple)):
        raise TypeError(
            f"run_command requires a list/tuple argv (shell=False); "
            f"got {type(cmd).__name__}. Refusing to run under a shell."
        )
    if not all(isinstance(a, str) for a in cmd):
        raise TypeError("run_command argv entries must all be str")
    try:
        result = subprocess.run(
            list(cmd),
            shell=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Return stdout if we got output (even if the subprocess exited non-zero)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
        logger.debug(f"run_command({cmd!r}) failed: {e}")
        return None


def get_git_info() -> GitInfo:
    """Get current git commit, branch, and dirty status."""
    commit = run_command(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = run_command(["git", "status", "--porcelain"])
    dirty = bool(status) if status is not None else False

    # Get commit message
    message = run_command(["git", "log", "-1", "--pretty=%B"]) or ""
    message = message.split("\n")[0][:80]  # First line, truncated

    return GitInfo(commit=commit, branch=branch, dirty=dirty, message=message)


def get_gpu_info() -> GPUInfo:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return GPUInfo(available=False)

    num_devices = torch.cuda.device_count()
    names = []
    memory_gb = []

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        names.append(props.name)
        memory_gb.append(props.total_memory / (1024**3))

    # Get CUDA version
    cuda_version = torch.version.cuda or "unknown"

    return GPUInfo(
        available=True,
        count=num_devices,
        names=names,
        memory_gb=memory_gb,
        cuda_version=cuda_version,
    )


def get_system_info() -> SystemInfo:
    """Get system information."""
    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.system(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cpu_count=psutil.cpu_count(logical=False) or 1,
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        memory_gb=psutil.virtual_memory().total / (1024**3),
        user=os.environ.get("USER", "unknown"),
        nanochat_base_dir=os.environ.get("NANOCHAT_BASE_DIR", "out"),
        working_dir=os.getcwd(),
    )


def estimate_cost(
    gpu_info: GPUInfo, runtime_hours: float | None = None
) -> CostEstimate | None:
    """Estimate training cost based on GPU type and runtime."""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.available:
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info.names[0] if gpu_info.names else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info.count
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info.count  # Default estimate

    return CostEstimate(
        hourly_rate=hourly_rate,
        gpu_type=gpu_name,
        estimated_total=hourly_rate * runtime_hours if runtime_hours else None,
    )


def generate_header():
    """Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info.branch}
- Commit: {git_info.commit} {"(dirty)" if git_info.dirty else "(clean)"}
- Message: {git_info.message}

### Hardware
- Platform: {sys_info.platform}
- CPUs: {sys_info.cpu_count} cores ({sys_info.cpu_count_logical} logical)
- Memory: {sys_info.memory_gb:.1f} GB
"""

    if gpu_info.available:
        gpu_names = ", ".join(set(gpu_info.names))
        total_vram = sum(gpu_info.memory_gb)
        header += f"""- GPUs: {gpu_info.count}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info.cuda_version}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info.hourly_rate > 0:
        header += f"""- Hourly Rate: ${cost_info.hourly_rate:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info.python_version}
- PyTorch: {sys_info.torch_version}

"""

    # bloat metrics: count lines/chars in git-tracked source files only.
    # Pipe-free implementation: invoke ``git ls-files`` with ``--`` plus
    # one glob per extension (each glob is a literal argv entry, NOT
    # shell-expanded), then read+count each tracked file in Python so we
    # don't need a shell at all.
    extensions = ["py", "md", "rs", "html", "toml", "sh"]
    git_patterns = [f"*.{ext}" for ext in extensions]
    files_output = run_command(["git", "ls-files", "--", *git_patterns])
    file_list = [f for f in (files_output or "").split("\n") if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    for rel_path in file_list:
        try:
            with open(rel_path, "rb") as f:
                data = f.read()
        except (OSError, FileNotFoundError) as e:
            # Skip silently-vanished files (e.g. git-tracked but deleted on
            # disk) so a single missing file does not blank the metrics.
            logger.debug(f"bloat metric: skipping {rel_path!r}: {e}")
            continue
        # Match `wc -lc`: count newline bytes for lines and total bytes
        # for chars.  Binary-safe; decoding unknown encodings would lose
        # data.
        num_lines += data.count(b"\n")
        num_chars += len(data)
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists("uv.lock"):
        with open("uv.lock", "r", encoding="utf-8") as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header


# -----------------------------------------------------------------------------


def slugify(text):
    """Slugify a text string."""
    return text.lower().replace(" ", "-").replace("/", "-")


# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]


def extract(section, keys):
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys]  # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out


def extract_timestamp(content, prefix):
    """Extract timestamp from content with given prefix."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse timestamp {time_str!r}: {e}")
    return None


class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(
                f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        logger.info(f"Generating report to {report_file}")
        final_metrics = {}  # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(
                        r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL
                    )
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = (
                    None  # will cause us to not write the total wall clock time
                )
                bloat_data = "[bloat data missing]"
                logger.warning(
                    f"{header_file} does not exist. Did you forget to run `nanochat reset`?"
                )
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    logger.warning(f"{section_file} does not exist, skipping")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(
                        section, "GSM8K"
                    )  # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(
                all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x)
            )
            # Fixed column widths
            stages = ["base", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        logger.info(f"Reset report and wrote header to {header_file}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


class DummyReport:
    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        pass


def get_report(checkpoint_dir=None):
    """Return a Report for rank 0, DummyReport for others.

    If checkpoint_dir is provided, the report directory is
    ``{checkpoint_dir}/report/``. Otherwise, falls back to
    ``{get_base_dir()}/report/``.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        if checkpoint_dir is not None:
            report_dir = os.path.join(checkpoint_dir, "report")
        else:
            report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or reset nanochat training reports."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="generate",
        choices=["generate", "reset"],
        help="Operation to perform (default: generate)",
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default=None,
        help="Checkpoint directory containing report/ subdirectory. Default: {base_dir}",
    )
    args = parser.parse_args()
    report = get_report(checkpoint_dir=args.checkpoint_dir)
    if args.command == "generate":
        report.generate()
    elif args.command == "reset":
        report.reset()
