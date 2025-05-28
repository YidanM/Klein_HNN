"""Klein manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh, tanh, artanh

class Klein(Manifold):
    """
    Klein manifold class.

    We use the following convention: x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Klein ball radius.
    """

    def __init__(self):
        super(Klein, self).__init__()
        self.name = 'Klein'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def klein_metric_tensor(self, x, c):
        """
        Compute the Klein metric tensor in matrix form G(x) at point x.
        """
        norm_sq = torch.sum(x**2, dim=-1, keepdim=True)  
        dim = x.shape[-1]
        I = torch.eye(dim, device=x.device).expand(*x.shape[:-1], dim, dim) 
        outer = x.unsqueeze(-1) @ x.unsqueeze(-2)  
        denom1 = 1 - norm_sq
        denom2 = denom1 ** 2
        G = I / denom1.unsqueeze(-1).clamp_min(self.min_norm) + outer / denom2.unsqueeze(-1).clamp_min(self.min_norm)
        return G

    def klein_norm(self, x, v, c):
        """
        Compute the norm of tangent vector v at point x.
        """
        G = self.klein_metric_tensor(x, c)  
        v = v.unsqueeze(-1)
        vTGv = torch.matmul(v.transpose(-2, -1), torch.matmul(G, v))
        return torch.sqrt(vTGv.unsqueeze(-1).unsqueeze(-1)) 

    def sqdist(self, x, y, c):
        """
        Compute the geodesic distance between two points x and y in the Klein model.
        """
        x_dot_y = torch.sum(x * y, dim=-1)  
        norm_x_sq = torch.sum(x**2, dim=-1)  
        norm_y_sq = torch.sum(y**2, dim=-1) 
        numerator = 1 - x_dot_y
        denominator = torch.sqrt((1 - norm_x_sq) * (1 - norm_y_sq)).clamp_min(self.min_norm)
        dist = arcosh(numerator / denominator)
        return dist ** 2
    
    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 1 / torch.sqrt((1. - x_sqnorm)).clamp_min(self.min_norm)
    
    def egrad2rgrad(self, x, egrad, c):
        G = self.klein_metric_tensor(x) 
        G_inv = torch.linalg.inv(G) 
        rgrad = torch.matmul(G_inv, egrad.unsqueeze(-1)).squeeze(-1)
        return rgrad

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u
    
    def expmap(self, x, v, c):
        norm_v = self.klein_norm(x, v, c).unsqueeze(-1)
        lambda_x = self._lambda_x(x, c)
        x_dot_v = (x * v).sum(dim=-1, keepdim=True) 
        sinh_norm = sinh(norm_v)
        cosh_norm = cosh(norm_v)
        numerator = sinh_norm * v / norm_v.clamp_min(self.min_norm)
        denominator = cosh_norm + (lambda_x ** 2) * x_dot_v * sinh_norm / norm_v.clamp_min(self.min_norm)
        return x + numerator / denominator.clamp_min(self.min_norm)
    
    def logmap(self, x, y, c):
        diff = y - x
        norm_diff = self.klein_norm(x, diff, c).clamp_min(self.min_norm)
        dist_xy = torch.sqrt(self.sqdist(x, y, c)).unsqueeze(-1)
        return dist_xy * diff / norm_diff

    def expmap0(self, u, c):
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(u_norm) * u / u_norm
        return gamma_1
        
    def logmap0(self, p, c):
        p_norm = torch.clamp_min(p.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        scale = arcosh(self._lambda_x(p, c)) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        """
        We adopt the same function name as the counterparts in the Poincare ball and hyperboloid models for ease of use. 
        This function in fact implements the Einstein addition of two vectors in the Klein model.
        """
        xy = (x * y).sum(dim=dim, keepdim=True)
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        gamma_x = 1/(1-x2)**0.5
        return 1/(1+xy).clamp_min(self.min_norm) * (x + 1/gamma_x.clamp_min(self.min_norm) * y+ gamma_x/(1+gamma_x).clamp_min(self.min_norm) * xy * x)
    
    def mobius_matvec(self, m, x, c):
        """
        We adopt the same function name as the counterparts in the Poincare ball and hyperboloid models for ease of use.
        This function in fact implements the Einstein matrix-vector multiplication in the Klein model.
        """
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(2 * mx_norm / x_norm * artanh(x_norm/(1+(1-x_norm**2)**0.5).clamp_min(self.min_norm))) * mx / mx_norm
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res
    
    def ptransp0(self, x, u, c):
        x_dot_u = torch.sum(x * u, dim=-1, keepdim=True)  
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)  
        sqrt_term = torch.sqrt(1 - x_norm_sq) 
        coeff = (x_dot_u * (sqrt_term - 2)) / (1 - sqrt_term).clamp_min(self.min_norm) 
        return coeff * x + sqrt_term * u
