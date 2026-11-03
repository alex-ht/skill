import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lib_paths  # noqa: E402


class TestLibPaths(unittest.TestCase):
    def test_explicit_workdir_wins(self):
        with patch.dict(os.environ, {"PINCHBENCH_WORKDIR": "/x/w", "SLURM_TMPDIR": "/y"}):
            self.assertEqual(lib_paths.pinchbench_workdir(), Path("/x/w"))
            self.assertEqual(lib_paths.run_root(), Path("/x/w/runs"))
            self.assertEqual(lib_paths.judge_root(), Path("/x/w/judge"))

    def test_slurm_job_dir_under_submit_dir(self):
        env = {"SLURM_JOB_ID": "42", "SLURM_SUBMIT_DIR": "/sub"}
        with patch.dict(os.environ, env):
            os.environ.pop("PINCHBENCH_WORKDIR", None)
            os.environ.pop("SLURM_TMPDIR", None)
            self.assertEqual(
                lib_paths.pinchbench_workdir(), Path("/sub/.pinchbench-work/job-42")
            )

    def test_never_defaults_to_tmp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(str(lib_paths.pinchbench_workdir()).startswith("/tmp/"))

    def test_openclaw_state_dir_env(self):
        with patch.dict(os.environ, {"OPENCLAW_STATE_DIR": "/s/oc"}):
            os.environ.pop("OPENCLAW_CONFIG_PATH", None)
            self.assertEqual(lib_paths.openclaw_state_dir(), Path("/s/oc"))
            self.assertEqual(lib_paths.openclaw_config_path(), Path("/s/oc/openclaw.json"))

    def test_ensure_tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"PINCHBENCH_WORKDIR": d, "TMPDIR": "/nonexistent-x"}):
                saved = tempfile.tempdir
                try:
                    got = lib_paths.ensure_tmpdir()
                    self.assertEqual(got, Path(d) / "tmp")
                    self.assertEqual(os.environ["TMPDIR"], str(Path(d) / "tmp"))
                finally:
                    tempfile.tempdir = saved


class TestNextRunId(unittest.TestCase):
    def test_skips_ids_already_in_output_dir(self):
        import benchmark

        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d) / "runs", Path(d) / "out"
            out.mkdir()
            (out / "0003_model.json").write_text("{}")
            (out / "0004_transcripts").mkdir()
            self.assertEqual(benchmark._next_run_id(root, out), "0005")
            self.assertEqual(benchmark._next_run_id(root), "0001")


if __name__ == "__main__":
    unittest.main()
