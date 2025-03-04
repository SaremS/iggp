import unittest
import torch
import gpytorch

#run tests from root directory
from src.integrated_gradients.iggp_softmax import * 
from src.svgp_models.multioutput_svgp import MultiOutputSVGP


class TestGetIggpSoftmax(unittest.TestCase):
    def test_softmax_riemann_batched(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        X = torch.randn(50, 5)
        y = torch.tensor(np.random.choice([0,1],50))
        m_svgp = MultiOutputSVGP(X, 2, 20)

        optimizer = torch.optim.Adam(m_svgp.parameters(), lr=0.01)

        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=2, num_classes=2, mixing_weights=False)
        mll = gpytorch.mlls.VariationalELBO(likelihood, m_svgp, y.numel())

        for i in range(2):
            optimizer.zero_grad()
            output = m_svgp(X)
            loss = -mll(output, y)
            loss.backward()

            optimizer.step()
        
        
        for i in range(3):
            for c in range(2):
                eval_point = X[i,:]
                base_point = torch.zeros(5)
                iggps_batched = get_iggp_softmax_riemann_batched(m_svgp,
                                                                 eval_point,
                                                                 base_point,
                                                                 c,
                                                                 20,
                                                                 5) 

                iggps_standard = get_iggp_softmax(m_svgp,
                                                  eval_point,
                                                  base_point,
                                                  c,
                                                  20)
                
                self.assertTrue(torch.all(torch.isclose(iggps_batched, iggps_standard)))


    def test_softmax_riemann_batched_large(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        X = torch.randn(50, 10)
        y = torch.tensor(np.random.choice([0,1,2],50))
        m_svgp = MultiOutputSVGP(X, 3, 20)

        optimizer = torch.optim.Adam(m_svgp.parameters(), lr=0.01)

        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=3, num_classes=3, mixing_weights=False)
        mll = gpytorch.mlls.VariationalELBO(likelihood, m_svgp, y.numel())

        for i in range(2):
            optimizer.zero_grad()
            output = m_svgp(X)
            loss = -mll(output, y)
            loss.backward()

            optimizer.step()

        for i in range(3):
            for c in range(3):
                eval_point = X[i,:]
                base_point = torch.zeros(10)
                iggps_batched = get_iggp_softmax_riemann_batched(m_svgp,
                                                                 eval_point,
                                                                 base_point,
                                                                 c,
                                                                 20,
                                                                 5) 

                iggps_standard = get_iggp_softmax(m_svgp,
                                                  eval_point,
                                                  base_point,
                                                  c,
                                                  20)
                
                self.assertTrue(torch.all(torch.isclose(iggps_batched, iggps_standard)))
