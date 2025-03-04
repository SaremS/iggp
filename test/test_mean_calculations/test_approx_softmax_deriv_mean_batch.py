import unittest
import torch

#run tests from root directory
from src.iggp_mean_calculations.approx_softmax_deriv_mean import * 
from src.iggp_mean_calculations.approx_softmax_deriv_mean_batch import *


class TestSoftmaxDerivProductMeanApproxBatch(unittest.TestCase):
    def test_softmax_c_gradient_batch(self):
        #batch output should equal non-batch output
        torch.manual_seed(42)
        
        X = torch.randn(5, 10)

        batched_grads = softmax_c_gradient_batch(X, 0)

        for d in range(5):
            simple_grads = softmax_c_gradient(X[d,:], 0)
            self.assertTrue(torch.equal(batched_grads[d,:], simple_grads))


    def test_softmax_c_hessian_batch(self):
        #batch output should equal non-batch output
        torch.manual_seed(42)
        
        X = torch.randn(5, 10)

        batched_hessians = softmax_c_hessian_batch(X, 0)

        for d in range(5):
            simple_hessians = softmax_c_hessian(X[d,:], 0)
            self.assertTrue(torch.equal(batched_hessians[d,:,:], simple_hessians))


    def test_softmax_target_fun_batch(self):
        B = 4 #batch-dimension
        N = 5 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_target_fun_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    x, c
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_target_fun(mu_b, full_cov_b, x_b, c)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b]-fun_output)<1e-6
                                ) 
                            )


    def test_softmax_target_fun_cj_batch(self):
        B = 4 #batch-dimension
        N = 5 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            for j in range(c+1, N-1):
                batch_fun_output = softmax_target_fun_cj_batch(
                        mu_X, mu_Y,
                        K_XX, K_XY,
                        x, c, j
                    )
                
                for b in range(B):
                    mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                    full_cov_b = full_cov[b,:,:]
                    x_b = x[b,:]

                    fun_output = softmax_target_fun_cj(mu_b, full_cov_b, x_b, c, j)

                    self.assertTrue(
                                torch.all(
                                    torch.abs(batch_fun_output[b]-fun_output)<1e-6
                                    ) 
                                )


    def test_softmax_target_fun_gradient_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_target_fun_gradient_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    x, c
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_target_fun_gradient(mu_b, full_cov_b, x_b, c)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b,:]-fun_output)<1e-6
                                ) 
                            )


    def test_softmax_target_fun_cj_gradient_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_target_fun_cj_gradient_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    x, c, 1
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_target_fun_cj_gradient(mu_b, full_cov_b, x_b, c, 1)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b,:]-fun_output)<1e-6
                                ) 
                            )


    def test_softmax_target_fun_hessian_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_target_fun_hessian_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    x, c
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_target_fun_hessian(mu_b, full_cov_b, x_b, c)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b,:,:]-fun_output)<1e-6
                                ) 
                            )


    def test_softmax_target_fun_cj_hessian_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_target_fun_cj_hessian_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    x, c, 0
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_target_fun_cj_hessian(mu_b, full_cov_b, x_b, c, 0)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b,:,:]-fun_output)<1e-6
                                ) 
                            )



    def test_softmax_deriv_c_product_mean_approx_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_deriv_c_product_mean_approx_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    c
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_deriv_c_product_mean_approx(mu_b, full_cov_b, c)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b]-fun_output)<1e-6
                                ) 
                            )


    def test_softmax_deriv_cj_product_mean_approx_batch(self):
        B = 4 #batch-dimension
        N = 3 #elements per batch, i.e. softmax inputs

        torch.manual_seed(123)

        x = torch.randn(B,N-1)

        mu_X = torch.randn(B,N-1)
        mu_Y = torch.randn(B)
        
        L = torch.randn(B,N,N)
        full_cov = torch.einsum("bij,bkj->bik", L, L)

        K_XX = full_cov[:,:-1,:-1]
        K_XY = full_cov[:,:-1,-1]

        for c in range(N-1):
            batch_fun_output = softmax_deriv_cj_product_mean_approx_batch(
                    mu_X, mu_Y,
                    K_XX, K_XY,
                    c, 0
                )
            
            for b in range(B):
                mu_b = torch.cat([mu_X[b,:], mu_Y[b].unsqueeze(0)],0)
                full_cov_b = full_cov[b,:,:]
                x_b = x[b,:]

                fun_output = softmax_deriv_cj_product_mean_approx(mu_b, full_cov_b, c, 0)

                self.assertTrue(
                            torch.all(
                                torch.abs(batch_fun_output[b]-fun_output)<1e-6
                                ) 
                            )
