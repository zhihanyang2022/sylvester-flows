import unittest
import core
import torch
from torch.autograd.functional import jacobian
from functools import partial
from scipy.stats import ortho_group
import numpy as np
import torch.nn.functional as F


class TestLinalg(unittest.TestCase):
    pass


class TestPlanar(unittest.TestCase):

    def test_transform(self):
        def transform(z, u, w, b):
            return core.planar.transform(z, u, w, b)[:-1]

        D = 100

        for seed in range(100):
            torch.manual_seed(seed)

            z = torch.randn(D).double()
            u = torch.randn(D).double()
            w = torch.randn(D).double()
            b = torch.randn(1).double()

            logabsdet_efficient = float(core.planar.transform(z, u, w, b)[-1])
            logabsdet_truth = float(torch.slogdet(jacobian(partial(transform, u=u, w=w, b=b), z))[1])

            self.assertAlmostEqual(logabsdet_efficient, logabsdet_truth, places=10)


class TestSylvester(unittest.TestCase):

    def test_transform(self):
        """
        test shape
        test logabsdet computation
        """

        def transform(z, Q, R1, R2, b):
            return core.sylvester.transform(z, Q, R1, R2, b)[:-1]

        D, M = 100, 50

        for seed in range(100):  # ensuring that the difference is small across many random seeds

            torch.manual_seed(seed)
            np.random.seed(seed)

            z = torch.randn(D).double()  # not using double can cause the difference in logabsdet to be about 1e6 ~ 1e-7
            Q = torch.from_numpy(ortho_group.rvs(D)[:, :M]).double()  # random matrix with orthogonal matrix
            R1 = torch.triu(torch.tanh(torch.randn(M, M))).double()
            R2 = torch.triu(torch.tanh(torch.randn(M, M))).double()
            b = torch.randn(M).double()

            if seed == 0:  # just need to test shape once

                out = core.sylvester.transform(z, Q, R1, R2, b)
                self.assertEqual(tuple(out.shape), (D + 1,))

            logabsdet_efficient = float(core.sylvester.transform(z, Q, R1, R2, b)[-1])
            logabsdet_truth = float(torch.slogdet(jacobian(partial(transform, Q=Q, R1=R1, R2=R2, b=b), z))[1])

            self.assertAlmostEqual(logabsdet_efficient, logabsdet_truth, places=10)

    def test_batch_transform(self):
        """
        test shape
        """
        bs, num_samples, D, M = 32, 10, 100, 50

        torch.manual_seed(42)
        np.random.seed(42)

        z = torch.randn(bs, num_samples, D).double()
        Q = torch.stack([torch.from_numpy(ortho_group.rvs(D)[:, :M]) for i in range(bs)])
        R1 = torch.triu(torch.tanh(torch.randn(bs, M, M))).double()
        R2 = torch.triu(torch.tanh(torch.randn(bs, M, M))).double()
        b = torch.randn(bs, M).double()

        out = core.sylvester.batch_transform(z, Q, R1, R2, b)

        self.assertEqual(tuple(out.shape), (bs, num_samples, D + 1))

    def test_batch_sample_from_flow(self):
        """
        test shape
        """
        bs, num_samples, D, M, K = 32, 10, 100, 50, 8

        torch.manual_seed(42)
        np.random.seed(42)

        μ = torch.randn(bs, D).double()
        σ = F.softplus(torch.randn(bs, D).double())
        Q = torch.stack([torch.from_numpy(ortho_group.rvs(D)[:, :M]) for i in range(bs * K)]).reshape(bs, K, D, M)
        R1 = torch.triu(torch.tanh(torch.randn(bs, K, M, M))).double()
        R2 = torch.triu(torch.tanh(torch.randn(bs, K, M, M))).double()
        b = torch.randn(bs, K, M).double()

        samples, logps = core.sylvester.batch_sample_from_flow(μ, σ, Q, R1, R2, b, num_samples)

        self.assertEqual(tuple(samples.shape), (bs, num_samples, D))
        self.assertEqual(tuple(logps.shape), (bs, num_samples))


if __name__ == '__main__':
    unittest.main()
