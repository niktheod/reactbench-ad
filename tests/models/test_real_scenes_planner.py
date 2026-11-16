"""Real-scenes ReCogDrive planner loading: every planner weight from the checkpoint, null states optional."""

import unittest

import numpy as np

from reactbench.models.recogdrive.real_scenes import planner_state


def tensor(*shape):
    return np.zeros(shape)


class PlannerStateTest(unittest.TestCase):
    MODEL = {"action_head.w": tensor(2, 3), "action_head.null_history": tensor(1, 1, 4), "backbone.x": tensor(1)}

    def test_upstream_checkpoint_without_null_states(self):
        state = planner_state(self.MODEL, {"agent.action_head.w": tensor(2, 3), "agent.old_policy.w": tensor(2, 3)})
        self.assertEqual(sorted(state), ["action_head.null_history", "action_head.w"])
        self.assertIs(state["action_head.null_history"], self.MODEL["action_head.null_history"])

    def test_missing_or_reshaped_weight(self):
        with self.assertRaises(ValueError):
            planner_state(self.MODEL, {"agent.action_head.null_history": tensor(1, 1, 4)})
        with self.assertRaises(ValueError):
            planner_state(self.MODEL, {"agent.action_head.w": tensor(3, 2)})


if __name__ == "__main__":
    unittest.main()
