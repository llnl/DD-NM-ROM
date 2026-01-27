import numpy as np
import dill as pickle
import scipy.linalg as la
import scipy.sparse as sp

from time import time
from dd_nm_rom import solvers


def compute_basis_from_svd(data_dict, ec=1e-6, nbasis=-1):
    if nbasis <= 0:
        s = data_dict['sing_vals']
        ss = s*s
        dim = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
    else:
        dim = nbasis
    return data_dict['left_vecs'][:, :dim]

# select sample nodes for hyper-reduction
def select_sample_nodes(residual_basis, nz, ncol=-1):
    '''
    Greedy algorithm to select sample nodes for hyper reduction.
    inputs:
        residual_basis: (Nr, nr)-array of residual basis vectors 
        nz: integer of desired number of sample nodes
        ncol: number of working columns of residual_basis
        
    outputs:
        s: array of sample nodes
    '''
    ncol = residual_basis.shape[1] if ncol<0 else ncol
    s    = np.array([], dtype=np.int)
    
    na       = nz - s.size                             # number of additional nodes to sample
    nb       = 0                                       # intializes counter for the number of working basis vectors used
    nit      = np.min([ncol, na])                      # number of greedy iterations to perform
    nrhs     = int(np.ceil(ncol/na))                   # max number of RHS in least squares problem
    ncj_min  = int(np.floor(ncol/nit))                 # minimum number of working basis vectors per iteration
    naj_min  = int(np.floor(na*nrhs/ncol))             # minimum number of sample nodes to add per iteration
    full_ind = np.arange(residual_basis.shape[0])      # array of all node inidices

    for j in range(nit):
        ncj = ncj_min            # number of working basis vectors for current iteration
        if j <= (ncol % nit - 1):
            ncj += 1
        naj = naj_min            # number of sample nodes to add during current iteration
        if np.logical_and(nrhs == 1, j<=(na % ncol - 1)):
            naj += 1
        if j == 0:
            R = residual_basis[:, 0:ncj]
        else:
            for q in range(ncj):
                vec     = np.linalg.lstsq(residual_basis[s, 0:nb], residual_basis[s, nb+q-1], rcond=None)[0]
                R[:, q] = residual_basis[:, nb+q-1]-residual_basis[:, 0:nb]@vec
        for k in range(naj):
            # choose node with largest average error
            ind = np.setdiff1d(full_ind, s)
            n   = np.where(np.isin(full_ind, ind))[0][np.argmax(np.sum(R[ind]*R[ind], axis=1))]
            s   = np.append(s, n)
        nb += ncj
        
    return np.sort(s)

def sp_diag(vec):
        '''
        Helper function to compute sparse diagonal matrix.
        
        input:
        vec: vector to compute sparse diagonal matrix of
        
        output:
        D: sparse diagonal matrix with diagonal vec
        '''
        return sp.spdiags(vec, 0, vec.size, vec.size)
    
class monolithic_LSROM:
    '''
    Class for generating a non-hyper-reduced subdomain of the DD-LS-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    fom: class of full-order model
    basis: basis for interior states on subdomain
    
    methods:
    res_jac: compute residual its jacobian on the subdomain
    '''
    def __init__(self, fom, basis):
        self.hxy = fom.hxy
        self.n_residual = fom.nxy
        self.n_state = basis.shape[1]
        
        # separate bases for u and v
        self.basis = basis
        self.u_basis = basis[:fom.nxy]
        self.v_basis = basis[fom.nxy:]
        
        # compute reused matrices
        self.Bx_u = fom.Bx@self.u_basis
        self.By_u = fom.By@self.u_basis
        self.Cx_u = fom.Cx@self.u_basis
        self.Cy_u = fom.Cy@self.u_basis
        
        self.Bx_v = fom.Bx@self.v_basis
        self.By_v = fom.By@self.v_basis
        self.Cx_v = fom.Cx@self.v_basis
        self.Cy_v = fom.Cy@self.v_basis
        
        self.b_ux1 = fom.b_ux1
        self.b_uy1 = fom.b_uy1
        self.b_ux2 = fom.b_ux2
        self.b_uy2 = fom.b_uy2
        
        self.b_vx1 = fom.b_vx1
        self.b_vy1 = fom.b_vy1
        self.b_vx2 = fom.b_vx2
        self.b_vy2 = fom.b_vy2
        
    def res_jac(self, w):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w: (n_interior,) vector - reduced state
        
        outputs:
        res: (n_residual,) vector - residual on subdomain
        jac: (n_residual, n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        '''
        # assemble u and v on residual subdomains
        u_res = self.u_basis@w
        v_res = self.v_basis@w
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_u@w
        Byu = self.By_u@w
        Cxu = self.Cx_u@w
        Cyu = self.Cy_u@w
        
        Bxv = self.Bx_v@w
        Byv = self.By_v@w
        Cxv = self.Cx_v@w
        Cyv = self.Cy_v@w
        
        # computes u and v residuals
        res_u = u_res*(Bxu - self.b_ux1) + v_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = u_res*(Bxv - self.b_vx1) + v_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # compute jacobians with respect to u and v interior and interface states  
        jac_u = sp.spdiags(Bxu-self.b_ux1, 0, self.n_residual, self.n_residual)@self.u_basis \
                 + sp.spdiags(u_res, 0, self.n_residual, self.n_residual)@self.Bx_u \
                 + sp.spdiags(Byu-self.b_uy1, 0, self.n_residual, self.n_residual)@self.v_basis \
                 + sp.spdiags(v_res, 0, self.n_residual, self.n_residual)@self.By_u \
                 + self.Cx_u \
                 + self.Cy_u
        
        jac_v = sp.spdiags(Bxv-self.b_vx1, 0, self.n_residual, self.n_residual)@self.u_basis \
                 +sp.spdiags(u_res, 0, self.n_residual, self.n_residual)@self.Bx_v \
                 +sp.spdiags(Byv-self.b_vy1, 0, self.n_residual, self.n_residual)@self.v_basis \
                 +sp.spdiags(v_res, 0, self.n_residual, self.n_residual)@self.By_v \
                 +self.Cx_v \
                 +self.Cy_v
        
        jac = np.vstack([jac_u, jac_v])
                             
        return res, jac
    
    def solve(self, w0, tol=1e-6, maxit=20, print_hist=False):
        w, conv_hist, step_hist, iter = solvers.gauss_newton(self.res_jac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
        return w, conv_hist, step_hist, iter
    
class monolithic_LSROM_HR:
    '''
    Class for generating a non-hyper-reduced subdomain of the DD-LS-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    fom: class of full-order model
    state_basis: basis for state
    residual_basis: basis for residuals (used for HR)
    sample_ratio: [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
    n_samples:    [optional] specify number of hyper reduction sample nodes. If n_samples<0, the number of samples is determined by the 
                  sample ratio. Default is -1. 
    methods:
    res_jac: compute residual its jacobian on the subdomain
    '''
    def __init__(self, fom, state_basis, residual_basis, sample_ratio=2, n_samples=-1):
        
        self.n_residual = fom.nxy
        self.n_state = state_basis.shape[1]
        
        # separate bases for u and v
        self.basis = state_basis
        self.u_basis = state_basis[:fom.nxy]
        self.v_basis = state_basis[fom.nxy:]
        
        # select HR nodes
        self.n_hr_nodes = sample_ratio*residual_basis.shape[1] if n_samples<0 else n_samples
        self.hr_ind = select_sample_nodes(residual_basis, self.n_hr_nodes)
        self.hr_ind_u = self.hr_ind[self.hr_ind < self.n_residual]
        self.hr_ind_v = self.hr_ind[self.hr_ind >= self.n_residual] - self.n_residual
        
        # compute reused matrices
        self.Bx_u = fom.Bx[self.hr_ind_u]@self.u_basis
        self.By_u = fom.By[self.hr_ind_u]@self.u_basis
        self.Cx_u = fom.Cx[self.hr_ind_u]@self.u_basis
        self.Cy_u = fom.Cy[self.hr_ind_u]@self.u_basis
        
        self.Bx_v = fom.Bx[self.hr_ind_v]@self.v_basis
        self.By_v = fom.By[self.hr_ind_v]@self.v_basis
        self.Cx_v = fom.Cx[self.hr_ind_v]@self.v_basis
        self.Cy_v = fom.Cy[self.hr_ind_v]@self.v_basis
        
        self.b_ux1 = fom.b_ux1[self.hr_ind_u]
        self.b_uy1 = fom.b_uy1[self.hr_ind_u]
        self.b_ux2 = fom.b_ux2[self.hr_ind_u]
        self.b_uy2 = fom.b_uy2[self.hr_ind_u]
        
        self.b_vx1 = fom.b_vx1[self.hr_ind_v]
        self.b_vy1 = fom.b_vy1[self.hr_ind_v]
        self.b_vx2 = fom.b_vx2[self.hr_ind_v]
        self.b_vy2 = fom.b_vy2[self.hr_ind_v]
        
        self.uu_incl = self.u_basis[self.hr_ind_u]
        self.uv_incl = self.v_basis[self.hr_ind_u]
        self.vu_incl = self.u_basis[self.hr_ind_v]
        self.vv_incl = self.v_basis[self.hr_ind_v]
        
        
    def res_jac(self, w):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w: (n_interior,) vector - reduced state
        
        outputs:
        res: (n_residual,) vector - residual on subdomain
        jac: (n_residual, n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        '''
        # assemble u and v on residual subdomains
        uu_res = self.uu_incl@w
        uv_res = self.uv_incl@w
        vu_res = self.vu_incl@w
        vv_res = self.vv_incl@w
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_u@w
        Byu = self.By_u@w
        Cxu = self.Cx_u@w
        Cyu = self.Cy_u@w
        
        Bxv = self.Bx_v@w
        Byv = self.By_v@w
        Cxv = self.Cx_v@w
        Cyv = self.Cy_v@w
        
        # computes u and v residuals
        res_u = uu_res*(Bxu - self.b_ux1) + uv_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = vu_res*(Bxv - self.b_vx1) + vv_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # compute jacobians with respect to u and v
        jac_u = sp_diag(Bxu-self.b_ux1)@self.uu_incl \
                 + sp_diag(uu_res)@self.Bx_u \
                 + sp_diag(Byu-self.b_uy1)@self.uv_incl \
                 + sp_diag(uv_res)@self.By_u \
                 + self.Cx_u \
                 + self.Cy_u
        
        jac_v = sp_diag(Bxv-self.b_vx1)@self.vu_incl \
                 +sp_diag(vu_res)@self.Bx_v \
                 +sp_diag(Byv-self.b_vy1)@self.vv_incl \
                 +sp_diag(vv_res)@self.By_v \
                 +self.Cx_v \
                 +self.Cy_v
        
        jac = np.vstack([jac_u, jac_v])
                             
        return res, jac
    
    def solve(self, w0, tol=1e-6, maxit=20, print_hist=False):
        w, conv_hist, step_hist, iter = solvers.gauss_newton(self.res_jac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
        return w, conv_hist, step_hist, iter