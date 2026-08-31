import unittest
from types import SimpleNamespace
import torch
from world_dynamics.frozen_world_model import FrozenWorldModel


class WorldModelTests(unittest.TestCase):
    def test_expected_imagine_shapes_and_seeded_uniforms(self):
        torch.manual_seed(4); model=FrozenWorldModel(obs_dim=38,action_dim=6,hidden_dim=16,deter_dim=12,stoch_dim=3,classes=4,scalar_bins=9,scalar_min=-2,scalar_max=2); h,z=model.initial(5,torch.device("cpu"),torch.float32); action=torch.randn(5,6); uniform=torch.rand(5,7,3,4); a=model.expected_imagine(h,z,action,uniform); b=model.expected_imagine(h,z,action,uniform); self.assertEqual(a[0].shape,(5,7,38)); self.assertEqual(a[1].shape,(5,7,1)); self.assertTrue(torch.equal(a[0],b[0]))
    def test_reset_carry_matches_recorded_policy_path(self):
        model=FrozenWorldModel(hidden_dim=16,deter_dim=12,stoch_dim=3,classes=4,scalar_bins=9); h,z=model.initial(2,torch.device("cpu"),torch.float32); self.assertTrue(torch.equal(h,torch.zeros_like(h))); self.assertTrue(torch.equal(z,torch.zeros_like(z)))


if __name__=="__main__": unittest.main()
