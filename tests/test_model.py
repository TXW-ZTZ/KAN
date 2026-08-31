import unittest
import torch
from world_dynamics.model import AdditiveDynamicsKAN


class ModelTests(unittest.TestCase):
    def test_edges_exactly_reconstruct_output(self):
        torch.manual_seed(3); model=AdditiveDynamicsKAN(45,11); x=torch.randn(13,45).clamp(-3.9,3.9); reconstructed=model.edge_contributions(x).sum(-1)+model.output_bias; self.assertTrue(torch.allclose(model(x),reconstructed,atol=1e-6,rtol=1e-6))


if __name__=="__main__": unittest.main()
