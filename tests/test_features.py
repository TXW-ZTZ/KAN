import unittest
import torch
from world_dynamics.features import DYNAMICS_FEATURE_NAMES,build_dynamics_features


class FeatureTests(unittest.TestCase):
    def test_shape_and_no_future_leakage(self):
        torch.manual_seed(2); semantics=torch.randn(12,21); actions=torch.randn(12,6); base=build_dynamics_features(semantics,actions); self.assertEqual(base.shape,(12,len(DYNAMICS_FEATURE_NAMES))); semantics[8:]+=100; actions[8:]-=100; changed=build_dynamics_features(semantics,actions); self.assertTrue(torch.equal(base[:8],changed[:8]))
    def test_past_summaries_start_zero(self):
        result=build_dynamics_features(torch.randn(5,21),torch.randn(5,6)); self.assertTrue(torch.equal(result[0,27:30],torch.zeros(3))); self.assertTrue(torch.equal(result[0,39:45],torch.zeros(6)))


if __name__=="__main__": unittest.main()
