import unittest
import torch

from src.iggp_mean_calculations.approx_sigmoid_deriv_mean import * #run tests from root directory

class TestSigmoidDerivProductMeanApprox(unittest.TestCase):
    def test_independent_isotropic(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.0, 0.0])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        approx_result = sigmoid_deriv_product_mean_approx(mu, cov)
        quadrature_result = sigmoid_deriv_product_mean_quadrature(mu, cov)
        mc_result = sigmoid_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, approx_result, places=3)



    def test_independent_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        approx_result = sigmoid_deriv_product_mean_approx(mu, cov)
        quadrature_result = sigmoid_deriv_product_mean_quadrature(mu, cov)
        mc_result = sigmoid_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, approx_result, places=3)    
       

    def test_correlated_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.05], [0.05, 0.1]])
        
        approx_result = sigmoid_deriv_product_mean_approx(mu, cov)
        quadrature_result = sigmoid_deriv_product_mean_quadrature(mu, cov)
        quad_batch_result = sigmoid_deriv_product_mean_quadrature_batch(
                mu[0], mu[1],
                cov[0,0], cov[1,1], cov[0,1]).item()
        mc_result = sigmoid_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, quad_batch_result, places=3)
        self.assertAlmostEqual(quadrature_result, quad_batch_result, places=10)


    def test_negative_correlation_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, -0.1])
        cov = torch.FloatTensor([[0.1, -0.05], [-0.05, 0.1]])
        
        approx_result = sigmoid_deriv_product_mean_approx(mu, cov)
        quadrature_result = sigmoid_deriv_product_mean_quadrature(mu, cov)
        quad_batch_result = sigmoid_deriv_product_mean_quadrature_batch(
                mu[0], mu[1],
                cov[0,0], cov[1,1], cov[0,1]).item()
        mc_result = sigmoid_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, approx_result, places=3)
        self.assertAlmostEqual(quadrature_result, quad_batch_result, places=3)
        self.assertAlmostEqual(quadrature_result, quad_batch_result, places=10)



