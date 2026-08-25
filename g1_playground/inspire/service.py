import logging
import os
import subprocess
import time

from g1_playground.utils import resolve_repo_path

logger = logging.getLogger("g1_playground")
SERVICE_BINARY = "third_party/dfx_inspire_service/build/inspire_g1"


class InspireService:
    """Own the DFX serial-to-DDS service process used by the real Inspire hands."""

    def __init__(self, net_if: str, domain_id: int, left_serial: str, right_serial: str):
        if domain_id != 0:
            raise ValueError(f"the DFX inspire service requires DDS domain 0, got {domain_id}")
        self.binary = resolve_repo_path(SERVICE_BINARY)
        if not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(
                f"{self.binary} is not executable; build it with "
                "'cd third_party/dfx_inspire_service && mkdir -p build && cd build && cmake .. && make -j4'"
            )
        self.command = [
            self.binary,
            "--network",
            net_if,
            "--left-serial",
            left_serial,
            "--right-serial",
            right_serial,
        ]
        self.process = None
        self.log = None

    def start(self) -> None:
        log_path = os.path.join("logs", "inspire_service.log")
        os.makedirs("logs", exist_ok=True)
        self.log = open(log_path, "w", encoding="utf-8")
        logger.warning("Starting the DFX inspire service, logging to %s", log_path)
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT)
        time.sleep(1.0)
        if self.process.poll() is not None:
            raise RuntimeError(
                f"the inspire service exited immediately (code {self.process.returncode}); see {log_path}. "
                "Are the serial ports free and is this session in the dialout group?"
            )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            logger.warning("Stopping the DFX inspire service")
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
