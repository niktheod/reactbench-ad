"""Run names: one scheme for ReactNAV and ReactPAI, the frame count k per dataset."""
import unittest

from reactbench.runs import TAUS_S, main, parse_run_name, run_name, timestep


class RunNameTests(unittest.TestCase):
    def test_round_trip(self):
        for dataset, frames in (("navsim", (1, 2, 3, 4)), ("pai-av", (5, 10, 15, 20))):
            for tau, k in zip(TAUS_S, frames):
                name = run_name("car", tau, dataset)
                self.assertEqual(name, f"car_tau{tau:.1f}s")
                self.assertEqual(parse_run_name(name, dataset), ("car", k, tau))
        self.assertEqual(run_name("crossing_pedestrian", 1.5, "pai-av"), "crossing_pedestrian_tau1.5s")
        self.assertEqual(parse_run_name("standing_animal_tau2.0s", "navsim"), ("standing_animal", 4, 2.0))

    def test_other_folders(self):
        for name in ("original", "car_tau1.0s_cot_swap", "car_timestep_2", "bicycle_tau1.0s", "car_tau1s"):
            self.assertIsNone(parse_run_name(name, "navsim"))

    def test_whole_frames_only(self):
        self.assertEqual(timestep(0.3, "pai-av"), 3)
        for tau, dataset in ((0.3, "navsim"), (0.25, "pai-av"), (0.0, "navsim")):
            with self.assertRaises(ValueError):
                timestep(tau, dataset)
        with self.assertRaises(ValueError):
            parse_run_name("car_tau0.3s", "navsim")

    def test_cli(self):
        from contextlib import redirect_stdout
        from io import StringIO
        out = StringIO()
        with redirect_stdout(out):
            main(["pai-av", "standing_pedestrian", "1.5"])
        self.assertEqual(out.getvalue().split(), ["standing_pedestrian_tau1.5s", "15"])


if __name__ == "__main__":
    unittest.main()
