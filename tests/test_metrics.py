import unittest
import torch
from world_dynamics.metrics import regression_report


class MetricTests(unittest.TestCase):
    def test_constant_target_has_no_r2(self):
        report=regression_report(torch.ones(10,2),torch.stack((torch.arange(10.),torch.zeros(10)),1),("varying","constant")); self.assertIsNotNone(report["per_output"][0]["r2"]); self.assertIsNone(report["per_output"][1]["r2"]); self.assertEqual(report["constant_target_outputs"],["constant"])

    def test_r2_denominator_and_rmse_identity_are_explicit(self):
        target=torch.tensor([[1.0],[2.0],[4.0],[8.0]],dtype=torch.float32)
        prediction=torch.tensor([[1.5],[1.0],[5.0],[7.0]],dtype=torch.float32)
        row=regression_report(prediction,target,("signal",))["per_output"][0]
        expected_sse=float((target.double()-prediction.double()).square().sum())
        expected_sst=float((target.double()-target.double().mean()).square().sum())
        self.assertAlmostEqual(row["sse"],expected_sse,places=12)
        self.assertAlmostEqual(row["sst"],expected_sst,places=12)
        self.assertAlmostEqual(row["r2"],1.0-expected_sse/expected_sst,places=12)
        self.assertAlmostEqual(row["r2"],1.0-row["rmse_over_target_std"]**2,places=12)


if __name__=="__main__": unittest.main()
