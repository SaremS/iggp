import unittest
import torch

from src.iggp_mean_calculations.exact_means import * #run tests from root directory

class TestGaussianPdfProductMean(unittest.TestCase):
    def test_independent_isotropic(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.0, 0.0])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = gaussian_pdf_product_mean(mu, cov)
        mc_result = gaussian_pdf_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = gaussian_pdf_product_mean(mu, cov)
        mc_result = gaussian_pdf_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_correlated_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.05], [0.05, 0.1]])
        
        exact_result = gaussian_pdf_product_mean(mu, cov)
        mc_result = gaussian_pdf_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_negative_correlation_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, -0.1])
        cov = torch.FloatTensor([[0.1, -0.05], [-0.05, 0.1]])
        
        exact_result = gaussian_pdf_product_mean(mu, cov)
        mc_result = gaussian_pdf_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


class TestExpProductMean(unittest.TestCase):
    def test_independent_isotropic(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.0, 0.0])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = exp_product_mean(mu, cov)
        mc_result = exp_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = exp_product_mean(mu, cov)
        mc_result = exp_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_correlated_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.05], [0.05, 0.1]])
        
        exact_result = exp_product_mean(mu, cov)
        mc_result = exp_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_negative_correlation_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, -0.1])
        cov = torch.FloatTensor([[0.1, -0.05], [-0.05, 0.1]])
        
        exact_result = exp_product_mean(mu, cov)
        mc_result = exp_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)



class TestSquareDerivProductMean(unittest.TestCase):
    def test_independent_isotropic(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.0, 0.0])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = square_deriv_product_mean(mu, cov)
        mc_result = square_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.0], [0.0, 0.1]])
        
        exact_result = square_deriv_product_mean(mu, cov)
        mc_result = square_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_correlated_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, 0.1])
        cov = torch.FloatTensor([[0.1, 0.05], [0.05, 0.1]])
        
        exact_result = square_deriv_product_mean(mu, cov)
        mc_result = square_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_negative_correlation_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.FloatTensor([0.1, -0.1])
        cov = torch.FloatTensor([[0.1, -0.05], [-0.05, 0.1]])
        
        exact_result = square_deriv_product_mean(mu, cov)
        mc_result = square_deriv_product_mean_mc(mu, cov, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)



if __name__ == "__main__":
    unittest.main()
