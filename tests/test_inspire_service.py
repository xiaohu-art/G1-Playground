import subprocess
import unittest
from unittest.mock import mock_open, patch

from g1_playground.inspire.service import InspireService


class TestInspireService(unittest.TestCase):
    def test_only_real_dds_domain_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "requires DDS domain 0"):
            InspireService("lo", 1, "/dev/left", "/dev/right")

    def test_start_and_stop_own_the_service_process(self):
        process = unittest.mock.Mock()
        process.poll.return_value = None
        output = mock_open()
        with (
            patch("g1_playground.inspire.service.resolve_repo_path", return_value="/repo/inspire_g1"),
            patch("g1_playground.inspire.service.os.access", return_value=True),
            patch("g1_playground.inspire.service.os.makedirs"),
            patch("g1_playground.inspire.service.open", output),
            patch("g1_playground.inspire.service.subprocess.Popen", return_value=process) as popen,
            patch("g1_playground.inspire.service.time.sleep") as sleep,
        ):
            service = InspireService("enP8p1s0", 0, "/dev/left", "/dev/right")
            service.start()
            service.stop()

        popen.assert_called_once_with(
            [
                "/repo/inspire_g1",
                "--network",
                "enP8p1s0",
                "--left-serial",
                "/dev/left",
                "--right-serial",
                "/dev/right",
            ],
            stdout=output(),
            stderr=subprocess.STDOUT,
        )
        sleep.assert_called_once_with(1.0)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3.0)
        output().close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
