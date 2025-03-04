import unittest
import torch

from src.iggp_mean_calculations.approx_softmax_deriv_mean import * #run tests from root directory

class TestSoftmaxDerivProductMeanApprox(unittest.TestCase):
    def test_independent_isotropic(self):
        torch.manual_seed(123)

        mu = torch.zeros(4) 
        cov = torch.zeros(4,4) 
        cov.diagonal().copy_(torch.ones(4)*0.1)
        
        exact_result = softmax_deriv_c_product_mean_approx(mu, cov, 0)
        mc_result = softmax_deriv_c_product_mean_mc(mu, cov, 0, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_nonzeromean(self):
        torch.manual_seed(123)

        mu = torch.tensor(np.array([0.1,-0.1,0.1,-0.1]), dtype=torch.float)
        cov = torch.zeros(4,4)
        cov.diagonal().copy_(torch.tensor(np.array([0.05,0.1,0.05,0.1]), dtype=torch.float))
        
        exact_result = softmax_deriv_c_product_mean_approx(mu, cov, 0)
        mc_result = softmax_deriv_c_product_mean_mc(mu, cov, 0, int(1e6))
        
        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_isotropic_cj(self):
        torch.manual_seed(123)

        mu = torch.zeros(4) 
        cov = torch.zeros(4,4) 
        cov.diagonal().copy_(torch.ones(4)*0.1)
        
        exact_result = softmax_deriv_cj_product_mean_approx(mu, cov, 0, 1)
        mc_result = softmax_deriv_cj_product_mean_mc(mu, cov, 0, 1, int(1e6))

        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_independent_nonzeromean_cj(self):
        torch.manual_seed(123)

        mu = torch.tensor(np.array([0.1,-0.1,0.1,-0.1]), dtype=torch.float)
        cov = torch.zeros(4,4)
        cov.diagonal().copy_(torch.tensor(np.array([0.05,0.1,0.05,0.1]), dtype=torch.float))
        
        exact_result = softmax_deriv_cj_product_mean_approx(mu, cov, 0, 1)
        mc_result = softmax_deriv_cj_product_mean_mc(mu, cov, 0, 1, int(1e6))
        
        self.assertAlmostEqual(mc_result, exact_result, places=3)


    def test_softmax_c_gradient_torchgrad(self):
        torch.manual_seed(123)

        for c in range(10):
            x = torch.randn(10, requires_grad=True)
            output = torch.softmax(x,0)[c]
            output.backward()

            torch_grad = x.grad

            with torch.no_grad():
                eval_grad = softmax_c_gradient(x,c)
                self.assertTrue(
                        torch.all(
                            torch.abs(torch_grad-eval_grad)<1e-6
                            )
                        )


    def test_softmax_c_hessian_torchgrad(self):
        torch.manual_seed(123)

        for c in range(5):
            for row in range(5):
                x = torch.randn(10, requires_grad=True)
                output = softmax_c_gradient(x,c)[row]
                output.backward()

                torch_hess_row = x.grad

                with torch.no_grad():
                    eval_hess_row = softmax_c_hessian(x,c)[row,:]
                    self.assertTrue(
                            torch.all(
                                torch.abs(torch_hess_row-eval_hess_row)<1e-6
                                )
                            )


    def test_softmax_target_fun_gradient_torchgrad(self):
        torch.manual_seed(123)

        for c in range(4):
            x = torch.randn(4, requires_grad=True)

            mu = torch.randn(5, requires_grad=False)
            L = torch.randn(5,5, requires_grad=False)
            cov = L@L.T

            output = softmax_target_fun(mu, cov, x, c)
            output.backward()

            torch_grad = x.grad

            with torch.no_grad():
                eval_grad = softmax_target_fun_gradient(mu, cov, x,c)
                self.assertTrue(
                        torch.all(
                            torch.abs(torch_grad-eval_grad)<1e-6
                            )
                        )
    

    def test_softmax_target_fun_cj_gradient_torchgrad(self):
        torch.manual_seed(123)

        for c in range(4):
            x = torch.randn(4, requires_grad=True)

            mu = torch.randn(5, requires_grad=False)
            L = torch.randn(5,5, requires_grad=False)
            cov = L@L.T

            output = softmax_target_fun_cj(mu, cov, x, c, 0)
            output.backward()

            torch_grad = x.grad

            with torch.no_grad():
                eval_grad = softmax_target_fun_cj_gradient(mu, cov, x,c,0)
                self.assertTrue(
                        torch.all(
                            torch.abs(torch_grad-eval_grad)<1e-6
                            )
                        )

    def test_softmax_target_fun_hessian_torchgrad(self):
        torch.manual_seed(123)

        for c in range(4):
            for row in range(4):
                x = torch.randn(4, requires_grad=True)

                mu = torch.randn(5, requires_grad=False)
                L = torch.randn(5,5, requires_grad=False)
                cov = L@L.T

                output = softmax_target_fun_gradient(mu, cov, x, c)[row]
                output.backward()

                torch_grad = x.grad

                with torch.no_grad():
                    eval_grad = softmax_target_fun_hessian(mu, cov, x,c)[row,:]
                    self.assertTrue(
                            torch.all(
                                torch.abs(torch_grad-eval_grad)<1e-6
                                )
                            )


    def test_softmax_target_fun_cj_hessian_torchgrad(self):
        torch.manual_seed(123)

        for c in range(4):
            for row in range(4):
                x = torch.randn(4, requires_grad=True)

                mu = torch.randn(5, requires_grad=False)
                L = torch.randn(5,5, requires_grad=False)
                cov = L@L.T

                output = softmax_target_fun_cj_gradient(mu, cov, x, c, 0)[row]
                output.backward()

                torch_grad = x.grad

                with torch.no_grad():
                    eval_grad = softmax_target_fun_cj_hessian(mu, cov, x,c, 0)[row,:]
                    self.assertTrue(
                            torch.all(
                                torch.abs(torch_grad-eval_grad)<1e-6
                                )
                            )
