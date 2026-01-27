import numpy as np
import dill as pickle
import scipy.linalg as la
import scipy.sparse as sp

from time import time
from dd_nm_rom import solvers


def save_svd_data(ddmdl, snapshots, residuals, save_dir='./', nsnaps=-1):
    '''
    Saves left singular vectors and singular values of snapshot and residual data for each subdomain.
    
    inputs: 
    ddmdl: instance of DD_model class
    snapshots: list where snapshots[i] = snapshot vector corresponding to ith parameter set
    residuals: list where residuals[i][j] = Newton residual at jth iteration corresponding to ith parameter set
    save_dir: [optional] directory to save data to. Default is current working directory.
    '''
    # gets indices for training snapshots
    if nsnaps>=0:
        idx = np.linspace(0, len(snapshots), num=nsnaps, endpoint=False, dtype=int)
        snapshots = list(np.vstack(snapshots)[idx])
    
    Nsnaps = len(snapshots)
    Nres   = len(residuals)
    
    # sets file names
    res_file =save_dir+f'res_svd.p'
    intr_file=save_dir+f'intr_svd_nsnaps_{Nsnaps}.p'        
    intf_file=save_dir+f'intf_svd_nsnaps_{Nsnaps}.p'
    port_file=save_dir+f'port_svd_nsnaps_{Nsnaps}.p'
    
    intr_leftvecs = []
    intr_singvals = []
    intf_leftvecs = []
    intf_singvals = []
    port_leftvecs = []
    port_singvals = []
    res_leftvecs = []
    res_singvals = []
    
    for i, s in enumerate(ddmdl.subdomains):    # loop over subdomains
        intr_snaps = []
        intf_snaps = []
        res_snaps  = []
        
        # extracts interior/interface/residual snapshots restricted to current subdomain
        print(f'Restricting snapshots to subdomain {i}...')
        for j in range(Nsnaps):
            intr_snaps.append(np.concatenate([snapshots[j][s.indices["interior"]], 
                                              snapshots[j][s.indices["interior"]+ddmdl.nxy]]))
            intf_snaps.append(np.concatenate([snapshots[j][s.indices["interface"]], 
                                              snapshots[j][s.indices["interface"]+ddmdl.nxy]]))
        for j in range(Nres):
            res_snaps.append(np.hstack([residuals[j][:, s.indices["res"]], 
                                        residuals[j][:, s.indices["res"]+ddmdl.nxy]]))
        print('Done!')
        
        # computes SVD
        print(f'Computing subdomain {i} interior SVD...')
        u_intr, s_intr, vh_intr = la.svd(np.vstack(intr_snaps).T, full_matrices=False, check_finite=False)
        print('Done!')
        print(f'Computing subdomain {i} interface SVD...')
        u_intf, s_intf, vh_intf = la.svd(np.vstack(intf_snaps).T, full_matrices=False, check_finite=False)
        print('Done!')
        print(f'Computing subdomain {i} residual SVD...')
        u_res, s_res, vh_res = la.svd(np.vstack(res_snaps).T, full_matrices=False, check_finite=False)
        print('Done!')
        
        intr_leftvecs.append(u_intr)
        intr_singvals.append(s_intr)
        intf_leftvecs.append(u_intf)
        intf_singvals.append(s_intf)
        res_leftvecs.append(u_res)
        res_singvals.append(s_res)
        
    for p in ddmdl.ports:    # loop over ports
        port_snaps = []

        # extracts port snapshots
        print(f'Restricting snapshots to port {p}...')
        for j in range(Nsnaps):
            port_snaps.append(np.concatenate([snapshots[j][ddmdl.ports_dict[p]], 
                                              snapshots[j][ddmdl.ports_dict[p]+ddmdl.nxy]]))
        print('Done!')

        # computes SVD
        print(f'Computing port {p} SVD...')
        u_port, s_port, vh_port = la.svd(np.vstack(port_snaps).T, full_matrices=False, check_finite=False)
        print('Done!')

        port_leftvecs.append(u_port)
        port_singvals.append(s_port)
        
    # puts left singular vectors and singular values into dictionaries
    intr_dict = {'left_vecs': intr_leftvecs, 
                 'sing_vals': intr_singvals}
    intf_dict = {'left_vecs': intf_leftvecs, 
                 'sing_vals': intf_singvals}
    ports_dict = {'left_vecs': port_leftvecs, 
                 'sing_vals': port_singvals}
    res_dict = {'left_vecs': res_leftvecs, 
                'sing_vals': res_singvals}
    
    # saves data
    pickle.dump(intr_dict, open(intr_file,'wb'))
    pickle.dump(intf_dict, open(intf_file,'wb'))
    pickle.dump(ports_dict, open(port_file, 'wb'))
    pickle.dump(res_dict, open(res_file,'wb'))    

# puts left singular vectors and singular values into dictionaries

# compute POD bases given SVD data
def compute_bases_from_svd(data_dict,
                           ec=1e-8, 
                           nbasis=-1):
    '''
    Computes POD bases given saved SVD data.
    
    inputs:
    data_dict: dictionary with fields
                'left_vecs' : list of matrices of left singular vectors of snapshot data for each subdomain
                'sing_vals' : list of vectors of singular values of snapshot data for each subdomain
    ec: [optional] energy criterior for choosing size of basis. Default is 1e-8
    nbasis: [optional] size of basis. Setting to -1 uses energy criterion. Default is -1.
    
    output:
    bases: list of matrices containing POD basis for each subdomain     
    '''
    bases = []
    for i, U in enumerate(data_dict['left_vecs']):
        if nbasis <= 0:
            s = data_dict['sing_vals'][i]
            ss = s*s
            dim = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
        else:
            dim = nbasis
        bases.append(U[:, :dim])
    return bases
    
# compute POD basis for given snapshot data
def POD(data, ec=1e-5, n_basis=-1):
    '''
    Compute POD basis given snapshot data.
    inputs:
        data: (N, M)-array of snapshot data
        ec: [optional] energy criterion for selecting number of basis vectors. Default is 1e-5
        n_basis: [optional] specify number of basis vectors. The default of -1 uses the energy criterion. 
                 Setting n_basis > 0 overrides the energy criterion.

    outputs:
        U: (N, n_basis)-array containing POD basis
        s: singular values of data
    '''
    u, s, vh = la.svd(data, full_matrices=False, check_finite=False)
    if n_basis <= 0:
        ss = s*s
        n_basis = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
    return u[:, 0:n_basis], s

def compute_bases(ddmdl, 
                  snapshots,
                  residuals=None,
                  ec_res=1e-8, 
                  ec_intr=1e-6, 
                  ec_intf=1e-6,
                  nbasis_res=-1, 
                  nbasis_intr=-1, 
                  nbasis_intf=-1,
                  intf_type='full_intf'):
    '''
    Compute residual, interior, and interface bases for each subdomain in the DD model.
    
    inputs: 
    ddmdl: instance of DD_model class
    snapshots: list where snapshots[i] = snapshot vector corresponding to ith parameter set
    residuals: [optional] list where residuals[i][j] = Newton residual at jth iteration correpsonding to ith parameter set
                          if residuals == None, then the residual basis is not computed.
    ec_res: [optional] residual basis energy criterion. Default is 1e-8
    ec_intr: [optional] interior basis energy criterion. Default is 1e-4
    ec_intf: [optional] interface basis energy criterion. Default is 1e-4
    nbasis_res: [optional] number of basis residual basis vectors. Default is -1. See POD documentation.
    nbasis_intr: [optional] number of basis interior basis vectors. Default is -1. See POD documentation.
    nbasis_intf: [optional] number of basis interface basis vectors. Default is -1. See POD documentation.
    intf_type: [optional] type of interface basis used. 'full_intf' or 'port'. Default is 'full_intf'
    
    outputs:
    interior_bases: list where interior_bases[i] is interior basis for ith subdomain
    interface_bases: list where interface_bases[i] is interface basis for ith subdomain
    residual_bases: list where residual_bases[i] is residual basis for ith subdomain
    
    '''
    
    Ns = len(snapshots)
    
    # separate full-domain snapshot data into subdomain snapshot data
    # The separated data corresponding to each subdomain and parameter pair is organized as follows:
    # - Mu[j] = the $j$th $(a_1, \lambda)$ parameter pair
    # - interior[i][j] = interior state snapshot on subdomain $i$ corresponding to parameters Mu[j]
    # - interface[i][j] = interface state snapshot on subdomain $i$ corresponding to parameters Mu[j]
    # - residual[i][j][k] = $k$th residual snapshot on subdomain $i$ corresponding to parameters Mu[j]
    # - ports[k][j] = state on port $k$ corresponding to parameters Mu[j]
    # - skeleton[j] = skeleton state (i.e. union of all interface states) corresponding to parameters Mu[j]+
    
    # collect port snapshots
    if intf_type=='port':
        port_states = []
        for port in ddmdl.ports:
            port_i = []
            for j in range(Ns):
                port_i.append(np.concatenate([snapshots[j][ddmdl.ports_dict[port]],
                                              snapshots[j][ddmdl.ports_dict[port] + ddmdl.nxy]]))
            port_states.append(port_i)
    
    elif intf_type=='skeleton':
        # collect skeleton snapshots
        skeleton = [np.concatenate([snapshots[j][ddmdl.skeleton], 
                                    snapshots[j][ddmdl.skeleton+ddmdl.nxy]]) for j in range(len(Mu))]
    
    # collect interior and interface snapshots on each subdomain
    interior = []
    interface = []
    for s in ddmdl.subdomains:
        interior_i = []
        interface_i = []
        for j in range(Ns):
            interior_i.append(np.concatenate([snapshots[j][s.indices["interior"]], 
                                              snapshots[j][s.indices["interior"]+ddmdl.nxy]]))
            interface_i.append(np.concatenate([snapshots[j][s.indices["interface"]], 
                                              snapshots[j][s.indices["interface"]+ddmdl.nxy]]))
        interior.append(interior_i)
        interface.append(interface_i)
        
    # Compute POD bases for each subdomain
    print('Computing bases for each subdomain...')
    
    if residuals != None:
        # collect residual snapshots on each subdomain
        Nr = len(residuals)
        residual = []
        for s in ddmdl.subdomains:
            residual_i = []
            for j in range(Nr):
                residual_i.append(np.hstack([residuals[j][:, s.indices["res"]], 
                                             residuals[j][:, s.indices["res"]+ddmdl.nxy]]))
            residual.append(residual_i)

        # residual bases
        residual_bases  = list([])
        for i in range(ddmdl.n_sub):
            res_i, s = POD(np.vstack(residual[i]).T, ec=ec_res, n_basis=nbasis_res)
            residual_bases.append(res_i)
            print(f'residual_bases[{i}].shape={residual_bases[i].shape}')
            
    # interior bases
    interior_bases  = list([])
    for i in range(ddmdl.n_sub):
        intr_i, s = POD(np.vstack(interior[i]).T, ec=ec_intr, n_basis=nbasis_intr)
        interior_bases.append(intr_i)
        print(f'interior_bases[{i}].shape={interior_bases[i].shape}')

    # interface bases
    if intf_type=='full_intf':
        interface_bases = list([])
        for i in range(ddmdl.n_sub):
            intf_i, s = POD(np.vstack(interface[i]).T, ec=ec_intf, n_basis=nbasis_intf)
            interface_bases.append(intf_i)
            print(f'interface_bases[{i}].shape={interface_bases[i].shape}')

    elif intf_type=='port':
        port_bases = list([])
        for i in range(len(port_states)):
            port_i, s = POD(np.vstack(port_states[i]).T, ec=ec_intf)
            port_bases.append(port_i)

        interface_bases = []
        for s in ddmdl.subdomains:
            shift = 0
            basis = np.zeros((2*s.n_interface, np.sum([port_bases[j].shape[1] for j in s.in_ports])))
            for j in s.in_ports:
                nodes = ddmdl.ports_dict[ddmdl.ports[j]]
                row = np.where(np.isin(s.indices["interface"], nodes))[0]
                row = np.concatenate([row, row+s.n_interface])
                col = np.arange(port_bases[j].shape[1])+shift
                basis[np.ix_(row, col)] = port_bases[j]
                shift += port_bases[j].shape[1]
            interface_bases.append(basis)
            print(f'interface_bases[{i}].shape={basis.shape}')

    print('Bases computed!')
    
    if residuals != None:
        return interior_bases, interface_bases, residual_bases
    else:
        return interior_bases, interface_bases

# select sample nodes for hyper-reduction
def select_sample_nodes(subdomain, residual_basis, nz, ncol, n_corners=1):
    '''
    Greedy algorithm to select sample nodes for hyper reduction.
    inputs:
        subdomain: instance of class subdomain_model
        residual_basis: (Nr, nr)-array of residual basis vectors corresponding to subdomain
        nz: integer of desired number of sample nodes
        ncol: number of working columns of residual_basis
        n_corners: number of interface nodes to include in sample nodes
        
    outputs:
        s: array of sample nodes
    '''
    s = np.array([], dtype=np.int32)
    
    # greedily sample corner nodes
    if n_corners > 0:
        corners = np.concatenate([subdomain.inter["res_interface"], subdomain.inter["res_interface"]+subdomain.n_indices["res"]])
        corner_ind = select_sample_nodes(subdomain, residual_basis[corners], n_corners, ncol, n_corners=0)
        s = np.union1d(s, corners[corner_ind])
        
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

# compute face-splitting product
def face_splitting(A, B):
    '''
    Compute face-splitting product of matrices A and B. Useful in the following identity:
        for matrices A, B and vectors x, y
        hadamard(Ax, By) = face_splitting(A, B) @ kron(x, y).
        
    It is related to the Khatri-Rao product by
        face_splitting(A, B).T = khatri_rao(A.T, B.T)
    and is equivalent to the row-wise kronecker product of A and B.
    
    inputs:
        A: (m, n) array
        B: (m, p) array
    outputs:
        C: (m, np) array 
    '''
    return la.khatri_rao(A.T, B.T).T

# class for non-hyper-reduced subdomain of the DD-LS-ROM
class subdomain_LS_ROM:
    '''
    Class for generating a non-hyper-reduced subdomain of the DD-LS-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    subdomain: subdomain class of full-order DD model
    interior_basis: basis for interior states on subdomain
    interface_basis: basis for interface states on subdomain
    constraint_mat: constraint matrix
    
    methods:
    res_jac: compute residual its jacobian on the subdomain
    '''
    def __init__(self, subdomain, interior_basis, interface_basis, constraint_mat, scaling=1):
        self.hxy = subdomain.hxy
        self.scaling = scaling
        self.in_ports  = subdomain.in_ports
        
        self.n_indices["residual"]  = subdomain.n_indices["residual"]
        self.n_interior  = interior_basis.shape[1]
        self.n_interface = interface_basis.shape[1]
        
        self.indices["interior"]  = subdomain.indices["interior"]
        self.indices["interface"] = subdomain.indices["interface"]
        
        # separate interior and interface bases for u and v
        self.intr_basis   = interior_basis
        self.intf_basis   = interface_basis
        self.u_intr_basis = interior_basis[0:subdomain.n_interior]
        self.v_intr_basis = interior_basis[subdomain.n_interior:]
        self.u_intf_basis = interface_basis[0:subdomain.n_interface]
        self.v_intf_basis = interface_basis[subdomain.n_interface:]
        
        # compute reused matrices
        self.Bx_intr_u = subdomain.Bx_interior@self.u_intr_basis
        self.By_intr_u = subdomain.By_interior@self.u_intr_basis
        self.Cx_intr_u = subdomain.Cx_interior@self.u_intr_basis
        self.Cy_intr_u = subdomain.Cy_interior@self.u_intr_basis
        
        self.Bx_intr_v = subdomain.Bx_interior@self.v_intr_basis
        self.By_intr_v = subdomain.By_interior@self.v_intr_basis
        self.Cx_intr_v = subdomain.Cx_interior@self.v_intr_basis
        self.Cy_intr_v = subdomain.Cy_interior@self.v_intr_basis
        
        self.Bx_intf_u = subdomain.Bx_interface@self.u_intf_basis
        self.By_intf_u = subdomain.By_interface@self.u_intf_basis
        self.Cx_intf_u = subdomain.Cx_interface@self.u_intf_basis
        self.Cy_intf_u = subdomain.Cy_interface@self.u_intf_basis
        
        self.Bx_intf_v = subdomain.Bx_interface@self.v_intf_basis
        self.By_intf_v = subdomain.By_interface@self.v_intf_basis
        self.Cx_intf_v = subdomain.Cx_interface@self.v_intf_basis
        self.Cy_intf_v = subdomain.Cy_interface@self.v_intf_basis
        
        self.u_intr_incl = subdomain.I_interior@self.u_intr_basis
        self.v_intr_incl = subdomain.I_interior@self.v_intr_basis
        self.u_intf_incl = subdomain.I_interface@self.u_intf_basis
        self.v_intf_incl = subdomain.I_interface@self.v_intf_basis
        
        self.b_ux1 = subdomain.b_ux1
        self.b_uy1 = subdomain.b_uy1
        self.b_ux2 = subdomain.b_ux2
        self.b_uy2 = subdomain.b_uy2
        
        self.b_vx1 = subdomain.b_vx1
        self.b_vy1 = subdomain.b_vy1
        self.b_vx2 = subdomain.b_vx2
        self.b_vy2 = subdomain.b_vy2
        
        self.constraint_mat = constraint_mat
        
    def res_jac(self, w_intr, w_intf, lam):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w_intr: (n_interior,) vector - reduced interior state
        w_intf: (n_interface,) vector - reduced interface state
        lam   : (n_constraints,) vector of lagrange multipliers
        
        outputs:
        res: (n_indices["residual"],) vector - residual on subdomain
        jac: (n_indices["residual"], n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ax : constraint matrix times interface state
        '''
        # assemble u and v on residual subdomains
        u_res = self.u_intr_incl@w_intr + self.u_intf_incl@w_intf
        v_res = self.v_intr_incl@w_intr + self.v_intf_incl@w_intf
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_intr_u@w_intr + self.Bx_intf_u@w_intf
        Byu = self.By_intr_u@w_intr + self.By_intf_u@w_intf
        Cxu = self.Cx_intr_u@w_intr + self.Cx_intf_u@w_intf
        Cyu = self.Cy_intr_u@w_intr + self.Cy_intf_u@w_intf
        
        Bxv = self.Bx_intr_v@w_intr + self.Bx_intf_v@w_intf
        Byv = self.By_intr_v@w_intr + self.By_intf_v@w_intf
        Cxv = self.Cx_intr_v@w_intr + self.Cx_intf_v@w_intf
        Cyv = self.Cy_intr_v@w_intr + self.Cy_intf_v@w_intf
        
        # computes u and v residuals
        res_u = u_res*(Bxu - self.b_ux1) + v_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = u_res*(Bxv - self.b_vx1) + v_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # compute jacobians with respect to u and v interior and interface states  
        jac_intr_u = sp.spdiags(Bxu-self.b_ux1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.u_intr_incl \
                     + sp.spdiags(u_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.Bx_intr_u \
                     + sp.spdiags(Byu-self.b_uy1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.v_intr_incl \
                     + sp.spdiags(v_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.By_intr_u \
                     + self.Cx_intr_u \
                     + self.Cy_intr_u
        
        jac_intr_v = sp.spdiags(Bxv-self.b_vx1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.u_intr_incl \
                     +sp.spdiags(u_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.Bx_intr_v \
                     +sp.spdiags(Byv-self.b_vy1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.v_intr_incl \
                     +sp.spdiags(v_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.By_intr_v \
                     +self.Cx_intr_v \
                     +self.Cy_intr_v
        
        jac_intf_u = sp.spdiags(Bxu-self.b_ux1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.u_intf_incl \
                     +sp.spdiags(u_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.Bx_intf_u \
                     +sp.spdiags(Byu-self.b_uy1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.v_intf_incl \
                     +sp.spdiags(v_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.By_intf_u \
                     +self.Cx_intf_u \
                     +self.Cy_intf_u
        
        jac_intf_v = sp.spdiags(Bxv-self.b_vx1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.u_intf_incl \
                     +sp.spdiags(u_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.Bx_intf_v \
                     +sp.spdiags(Byv-self.b_vy1, 0, self.n_indices["residual"], self.n_indices["residual"])@self.v_intf_incl \
                     +sp.spdiags(v_res, 0, self.n_indices["residual"], self.n_indices["residual"])@self.By_intf_v \
                     +self.Cx_intf_v \
                     +self.Cy_intf_v
        
        jac_intr = np.vstack([jac_intr_u, jac_intr_v])
        jac_intf = np.vstack([jac_intf_u, jac_intf_v])
        
        # compute terms needed for SQP solver
        Ax = self.constraint_mat@w_intf
        
        jac = np.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res), 
                              self.scaling*(jac_intf.T@res)+self.constraint_mat.T@lam])
                             
        return res, jac, H, rhs, Ax
    
# generate hyper-reduced subdomain LS-ROM
class subdomain_LS_ROM_HR:
    '''
    Class for generating a hyper-reduced subdomain of the DD-LS-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    subdomain: subdomain class of full-order DD model
    residual_basis: basis for residual snapshots
    interior_basis: basis for interior states on subdomain
    interface_basis: basis for interface states on subdomain
    hr_type: hyper-reduction type. Either 'gappy_POD' or 'collocation'
    constraint_mat: constraint matrix for subdomain
    nz: number of hyper reduction indices
    ncol: number of working columns for sample node selection algorithm
    n_corners: [optional] number of interface columns to include in sample nodes. Default is 5
    
    methods:
    res_jac: compute residual and its jacobian on the subdomain
    '''
    def __init__(self, subdomain, 
                 residual_basis, 
                 interior_basis, 
                 interface_basis, 
                 hr_type,
                 constraint_mat, 
                 nz, 
                 ncol, 
                 n_corners=5,
                 scaling=1):
        self.hxy = subdomain.hxy
        self.scaling = scaling
        
        # select residual nodes for hyper reduction
        self.hr_ind = select_sample_nodes(subdomain, residual_basis, nz, ncol, n_corners=n_corners)
        self.hr_ind_u = self.hr_ind[self.hr_ind < subdomain.n_indices["residual"]]
        self.hr_ind_v = self.hr_ind[self.hr_ind >= subdomain.n_indices["residual"]]-subdomain.n_indices["residual"]
        
        # reused quantities
        self.in_ports = subdomain.in_ports
        
        # stores bases
        self.residual_basis = residual_basis
        self.intr_basis = interior_basis
        self.intf_basis = interface_basis
        
        self.n_indices["residual"] = subdomain.n_indices["residual"]
        self.n_interior = interior_basis.shape[1]
        self.n_interface = interface_basis.shape[1]
        
        self.indices["interior"] = subdomain.indices["interior"]
        self.indices["interface"] = subdomain.indices["interface"]
        
        # residual weighting matrix corresponding to HR type
        if hr_type == 'gappy_POD':
            self.res_weight   = la.pinv(self.residual_basis[self.hr_ind])
        elif hr_type == 'collocation':
            self.res_weight = sp.eye(self.hr_ind.size)
        
        # separate interior and interface bases for u and v
        self.u_intr_basis = interior_basis[0:subdomain.n_interior]
        self.v_intr_basis = interior_basis[subdomain.n_interior:]
        self.u_intf_basis = interface_basis[0:subdomain.n_interface]
        self.v_intf_basis = interface_basis[subdomain.n_interface:]
        
        self.constraint_mat = constraint_mat
        
        # compute reused matrices
        Bx_intr_u = subdomain.Bx_interior[self.hr_ind_u]@self.u_intr_basis
        By_intr_u = subdomain.By_interior[self.hr_ind_u]@self.u_intr_basis
        Cx_intr_u = subdomain.Cx_interior[self.hr_ind_u]@self.u_intr_basis
        Cy_intr_u = subdomain.Cy_interior[self.hr_ind_u]@self.u_intr_basis
        
        Bx_intr_v = subdomain.Bx_interior[self.hr_ind_v]@self.v_intr_basis
        By_intr_v = subdomain.By_interior[self.hr_ind_v]@self.v_intr_basis
        Cx_intr_v = subdomain.Cx_interior[self.hr_ind_v]@self.v_intr_basis
        Cy_intr_v = subdomain.Cy_interior[self.hr_ind_v]@self.v_intr_basis
        
        Bx_intf_u = subdomain.Bx_interface[self.hr_ind_u]@self.u_intf_basis
        By_intf_u = subdomain.By_interface[self.hr_ind_u]@self.u_intf_basis
        Cx_intf_u = subdomain.Cx_interface[self.hr_ind_u]@self.u_intf_basis
        Cy_intf_u = subdomain.Cy_interface[self.hr_ind_u]@self.u_intf_basis
        
        Bx_intf_v = subdomain.Bx_interface[self.hr_ind_v]@self.v_intf_basis
        By_intf_v = subdomain.By_interface[self.hr_ind_v]@self.v_intf_basis
        Cx_intf_v = subdomain.Cx_interface[self.hr_ind_v]@self.v_intf_basis
        Cy_intf_v = subdomain.Cy_interface[self.hr_ind_v]@self.v_intf_basis
        
        uu_intr_incl = subdomain.I_interior[self.hr_ind_u]@self.u_intr_basis
        uv_intr_incl = subdomain.I_interior[self.hr_ind_u]@self.v_intr_basis
        uu_intf_incl = subdomain.I_interface[self.hr_ind_u]@self.u_intf_basis
        uv_intf_incl = subdomain.I_interface[self.hr_ind_u]@self.v_intf_basis
        
        vu_intr_incl = subdomain.I_interior[self.hr_ind_v]@self.u_intr_basis
        vv_intr_incl = subdomain.I_interior[self.hr_ind_v]@self.v_intr_basis
        vu_intf_incl = subdomain.I_interface[self.hr_ind_v]@self.u_intf_basis
        vv_intf_incl = subdomain.I_interface[self.hr_ind_v]@self.v_intf_basis
        
        b_ux1 = subdomain.b_ux1[self.hr_ind_u]
        b_uy1 = subdomain.b_uy1[self.hr_ind_u]
        b_ux2 = subdomain.b_ux2[self.hr_ind_u]
        b_uy2 = subdomain.b_uy2[self.hr_ind_u]
        
        b_vx1 = subdomain.b_vx1[self.hr_ind_v]
        b_vy1 = subdomain.b_vy1[self.hr_ind_v]
        b_vx2 = subdomain.b_vx2[self.hr_ind_v]
        b_vy2 = subdomain.b_vy2[self.hr_ind_v]
        
        # compute matrices used for residual and jacobian computation
        # u-block matrices
        u_intr_intr = face_splitting(uu_intr_incl, Bx_intr_u) \
                     +face_splitting(uv_intr_incl, By_intr_u)
        u_intr_intf = face_splitting(uu_intr_incl, Bx_intf_u) + face_splitting(Bx_intr_u, uu_intf_incl) \
                     +face_splitting(uv_intr_incl, By_intf_u) + face_splitting(By_intr_u, uv_intf_incl)
        u_intf_intf = face_splitting(uu_intf_incl, Bx_intf_u) \
                     +face_splitting(uv_intf_incl, By_intf_u)
        u_intr = -self.hr_diag(b_ux1)@uu_intr_incl \
                 -self.hr_diag(b_uy1)@uv_intr_incl \
                 +Cx_intr_u \
                 +Cy_intr_u
        u_intf = -self.hr_diag(b_ux1)@uu_intf_incl \
                 -self.hr_diag(b_uy1)@uv_intf_incl \
                 +Cx_intf_u \
                 +Cy_intf_u
        u_b = b_ux2 + b_uy2
        
        # v-block matrices
        v_intr_intr = face_splitting(vu_intr_incl, Bx_intr_v) \
                     +face_splitting(vv_intr_incl, By_intr_v)
        v_intr_intf = face_splitting(vu_intr_incl, Bx_intf_v) + face_splitting(Bx_intr_v, vu_intf_incl) \
                     +face_splitting(vv_intr_incl, By_intf_v) + face_splitting(By_intr_v, vv_intf_incl)
        v_intf_intf = face_splitting(vu_intf_incl, Bx_intf_v) \
                     +face_splitting(vv_intf_incl, By_intf_v)
        v_intr = -self.hr_diag(b_vx1)@vu_intr_incl \
                 -self.hr_diag(b_vy1)@vv_intr_incl \
                 +Cx_intr_v \
                 +Cy_intr_v
        v_intf = -self.hr_diag(b_vx1)@vu_intf_incl \
                 -self.hr_diag(b_vy1)@vv_intf_incl \
                 +Cx_intf_v \
                 +Cy_intf_v
        v_b = b_vx2 + b_vy2
        
        # full matrices
        self.I_intr = sp.eye(self.n_interior)
        self.I_intf = sp.eye(self.n_interface)
        self.intr_intr = self.res_weight@np.vstack([u_intr_intr, v_intr_intr])
        self.intr_intf = self.res_weight@np.vstack([u_intr_intf, v_intr_intf])
        self.intf_intf = self.res_weight@np.vstack([u_intf_intf, v_intf_intf])
        self.intr = self.res_weight@np.vstack([u_intr, v_intr])
        self.intf = self.res_weight@np.vstack([u_intf, v_intf])
        self.b = self.res_weight@np.concatenate([u_b, v_b])
        
    def hr_diag(self, vec):
        '''
        Helper function to compute sparse diagonal matrix.
        
        input:
        vec: vector to compute sparse diagonal matrix of
        
        output:
        D: sparse diagonal matrix with diagonal vec
        '''
        return sp.spdiags(vec, 0, vec.size, vec.size)
    
    def res_jac(self, w_intr, w_intf, lam):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w_intr: (n_interior,) vector - reduced interior state
        w_intf: (n_interface,) vector - reduced interface state
        lam   : (n_constraints,) vector of lagrange multipliers
        
        outputs:
        res: (nz,) vector - residual on subdomain
        jac: (n_indices["residual"], n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ax : constraint matrix times interface state
        '''
        # compute residual
        res = self.intr_intr@np.kron(w_intr, w_intr) + self.intr_intf@np.kron(w_intr, w_intf) \
             + self.intf_intf@np.kron(w_intf, w_intf) + self.intr@w_intr + self.intf@w_intf + self.b
        
        w_intr_col = w_intr.reshape(-1, 1)
        w_intf_col = w_intf.reshape(-1, 1)
        
        # compute jacobians with respect to interior and interface states
        jac_intr = self.intr_intr@(sp.kron(self.I_intr, w_intr_col) + sp.kron(w_intr_col, self.I_intr)) \
                   + self.intr_intf@sp.kron(self.I_intr, w_intf_col) + self.intr
        
        jac_intf = self.intf_intf@(sp.kron(self.I_intf, w_intf_col) + sp.kron(w_intf_col, self.I_intf)) \
                   + self.intr_intf@sp.kron(w_intr_col, self.I_intf) + self.intf
                             
        # compute terms needed for SQP solver
        Ax = self.constraint_mat@w_intf
        
        jac = np.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res),
                              self.scaling*(jac_intf.T@res)+self.constraint_mat.T@lam])
                             
        return res, jac, H, rhs, Ax 
    
class DD_LS_ROM:
    '''
    Compute DD-LS-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.
    
    inputs:
    ddmdl: DD model class corresponding to full order DD model. 
    residual_bases: list of residual bases where residual_bases[i] is the residual basis for the ith subdomain
    interior_bases: list of interior bases where interior_bases[i] is the interior basis for the ith subdomain
    interface_bases: list of interface bases where interface_bases[i] is the interface basis for the ith subdomain
    hr: [optional] Boolean to specify if hyper reduction is applied. Default is False
    hr_type: [optional] Hyper-reduction type. Either 'gappy_POD' or 'collocation'. 
                Only takes effect if hr=True. Default is 'collocation'
    sample_ratio: [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
    n_samples: [optional] specify number of hyper reduction sample nodes. 
                If n_samples is an array with length equal to the number of subdomains, then 
                n_samples[i] is the number of HR samples on the ith subdomain.
                
                If n_samples is a positive integer, then each subdomain has n_samples HR nodes. 
                
                Otherwise, the number of samples is determined by the sample ratio. 
                Default is -1. 
                
    n_corners: [optional] Number of interface nodes included in the HR sample nodes. 
                If n_corners is an array with length equal to the number of subdomains, then 
                n_corners[i] is the number of interface HR nodes on the ith subdomain.
                
                If n_corners is a positive integer, then each subdomain has n_corners interface HR nodes. 
                
                Otherwise, the number of interface HR nodes on each subdomain is determined by n_samples
                multiplied by the ratio of the number of interface nodes contained in the residual nodes 
                to the total number of residual nodes.
                
                Default is -1. 
    
    constraint_type: [optional] 'weak' or 'strong' compatibility constraints. Default is weak. 
    n_constraints: [optional] number of 'weak' constraints
    
    fields:
    n_constraints: number of (equality) constraints for NLP
    constraint_mult: matrix multiplier corresponding to constraint_type. 
                     if constraint_type == 'weak':
                         constraint_mult = (n_constraints, nA) Gaussian random matrix
                     elif constraint_type == 'strong':
                         constraint_mult = identity
    subdomain: list of subdomain_LS_ROM or subdomain_LS_ROM_HR classes corresponding to each subdomain, i.e.
               subdomain[i] = reduced subdomain class corresponding to subdomain [i]
               
    methods:
    FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
    solve: solves for the reduced states of the DD-LS-ROM using the Lagrange-Newton-SQP method. 
    '''
    def __init__(self, ddmdl, 
                 residual_bases,
                 interior_bases, 
                 interface_bases,
                 port_bases=[],
                 hr=False, 
                 hr_type='collocation',
                 sample_ratio=2, 
                 n_samples=-1, 
                 n_corners=-1,
                 constraint_type='weak', 
                 n_constraints=1,
                 seed=None, 
                 scaling=1):
        self.hr = hr
        self.nxy = ddmdl.nxy
        self.hxy = ddmdl.hxy
        self.n_sub = ddmdl.n_subs
        self.ports = ddmdl.ports
        self.constraint_type = constraint_type 
        self.scaling = self.hxy if scaling < 0 else scaling
        
        if constraint_type=='weak': # weak FOM-port constraints
            self.n_constraints = n_constraints
            rng = np.random.default_rng(seed)
            self.constraint_mult = rng.standard_normal(size=(self.n_constraints, ddmdl.n_constraints))
            constraint_mat_list = [self.constraint_mult@s.constraint_mat@interface_bases[j] for j, s in enumerate(ddmdl.subdomains)]
            
        else: # strong ROM-port constraints
            assert len(port_bases) == len(ddmdl.ports)
            
            # assign coupling conditions to ROM states
            # rom_port_ind[j][p] = indices of intf state on subdomain j corresponding to port p
            self.rom_port_ind = []
            for s in ddmdl.subdomains:
                rom_port_dict = {}
                shift = 0
                for p in s.in_ports:
                    rom_port_dict[p] = np.arange(port_bases[p].shape[1])+shift
                    shift += port_bases[p].shape[1]
                self.rom_port_ind.append(rom_port_dict)   
               
            # assemble interface bases using port bases
            interface_bases = []
            n_intf_list = []
            for i, s in enumerate(ddmdl.subdomains):
                n_intf = np.sum([len(self.rom_port_ind[i][p]) for p in s.in_ports])
                n_intf_list.append(n_intf)
                basis  = np.zeros((2*s.n_interface, n_intf))
                for p in s.in_ports:
                    p_ind = np.concatenate([ddmdl.ports_dict[ddmdl.ports[p]], ddmdl.ports_dict[ddmdl.ports[p]]+ddmdl.nxy])
                    s_ind = np.concatenate([s.indices["interface"], s.indices["interface"]+ddmdl.nxy])
                    fomport2intf = np.nonzero(np.isin(s_ind, p_ind))[0]
                    basis[np.ix_(fomport2intf, self.rom_port_ind[i][p])] += port_bases[p]
                interface_bases.append(basis)
            
            # assemble ROM-port constraint matrices
            self.n_constraints  = np.sum([(len(port)-1)*port_bases[k].shape[1] for k, port in enumerate(ddmdl.ports)])
            constraint_mat_list = [sp.coo_matrix((self.n_constraints, n_intf)) for n_intf in n_intf_list]
            
            shift = 0 
            for j, p in enumerate(ddmdl.ports):
                port = list(p)
                npj = port_bases[j].shape[1]
                for i in range(len(port)-1):
                    s1   = port[i]
                    constraint_mat_list[s1].col  = np.concatenate((constraint_mat_list[s1].col, self.rom_port_ind[s1][j]))
                    constraint_mat_list[s1].row  = np.concatenate((constraint_mat_list[s1].row, np.arange(npj)+shift))
                    constraint_mat_list[s1].data = np.concatenate((constraint_mat_list[s1].data, np.ones(npj)))   

                    s2   = port[i+1]
                    constraint_mat_list[s2].col  = np.concatenate((constraint_mat_list[s2].col, self.rom_port_ind[s2][j]))
                    constraint_mat_list[s2].row  = np.concatenate((constraint_mat_list[s2].row, np.arange(npj)+shift))
                    constraint_mat_list[s2].data = np.concatenate((constraint_mat_list[s2].data, -np.ones(npj)))

                    shift += npj
        
        self.subdomain=list([])
        if hr:
            if hr_type in ['gappy_POD', 'collocation']:
                self.hr_type = hr_type
            else: 
                print("Error: hr_type must be 'gappy_POD' or 'collocation'.")
                return 
        
            # compute parameters for hyper reduction
            ncol = np.array([rb.shape[1] for rb in residual_bases])             # number of working columns per subdomain
            nz_max = np.array([rb.shape[0] for rb in residual_bases])
            
            # number of sample nodes per subdomain
            if isinstance(n_samples, int):
                if n_samples > 0:
                    nz = n_samples*np.ones(self.n_sub)
                else:
                    nz = sample_ratio*ncol
            else:
                nz = n_samples
            nz = np.maximum(np.minimum(nz, nz_max), ncol)
            
            # number of corner nodes per subdomain
            n_corners_max = nz_max - np.array([b.shape[0] for b in interior_bases])
            if isinstance(n_corners, int):
                if n_corners > 0:
                    n_corners = n_corners*np.ones(self.n_sub)
                else:
                    n_corners = np.round(nz*n_corners_max/nz_max)
            n_corners = np.minimum(n_corners, n_corners_max)
            
            # generate subdomain models
            for i in range(ddmdl.n_sub):
                self.subdomain.append(subdomain_LS_ROM_HR(ddmdl.subdomains[i], 
                                                          residual_bases[i],
                                                          interior_bases[i], 
                                                          interface_bases[i], 
                                                          self.hr_type,
                                                          constraint_mat_list[i],
                                                          int(nz[i]),
                                                          int(ncol[i]), 
                                                          n_corners=int(n_corners[i]), 
                                                          scaling=self.scaling))
                
        else:
            # generate subdomain models
            for i in range(ddmdl.n_sub):
                self.subdomain.append(subdomain_LS_ROM(ddmdl.subdomains[i], 
                                                       interior_bases[i], 
                                                       interface_bases[i], 
                                                       constraint_mat_list[i], 
                                                       scaling=self.scaling))
    # compute RHS and jacobian for SQP solver
    def FJac(self, w):
        '''
        Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
        
        inputs: 
        w: vector of all interior and interface states for each subdomain 
           and the lagrange multipliers lam in the order
            [intr[0], intf[0], ..., intr[n_sub], intf[n_sub], lam]
            
        outputs:
        val: RHS of the KKT system
        full_jac: KKT matrix
        runtime: "parallel" runtime to assemble KKT system
        '''
        start = time()
        shift = 0
        val = list([])
        A_list = list([])
        H_list = list([])
        
        constraint_res = np.zeros(self.n_constraints)
        lam = w[-self.n_constraints:]
        runtime = time()-start
        stimes = np.zeros(self.n_sub)
        
        for i, s in enumerate(self.subdomain):
            start = time()
            indices["interior"]  = np.arange(s.n_interior)
            indices["interface"] = np.arange(s.n_interface)

            w_intr = w[indices["interior"]+shift]
            shift += s.n_interior

            w_intf = w[indices["interface"]+shift]
            shift += s.n_interface
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs, Ax = s.res_jac(w_intr, w_intf, lam)
            stimes[i] = time()-start
            
            # RHS block for KKT system
            start = time()
            val.append(rhs)
            constraint_res += Ax
            
            H_list.append(H)
            A_list += [sp.csr_matrix((self.n_constraints, s.n_interior)), s.constraint_mat]                            
            runtime += time()-start
        
        start = time()
        val.append(constraint_res)
        val = np.concatenate(val)
        H_block = sp.block_diag(H_list)
        A_block = sp.hstack(A_list)
        full_jac = sp.bmat([[H_block, A_block.T], [A_block, None]]).tocsr()
        runtime += time()-start
        runtime += stimes.max()
        
        return val, full_jac, runtime
    
    # solve SQP using Lagrange-Newton SQP
    def solve(self, w0, tol=1e-5, maxit=20, print_hist=False, rhs_hist=False):
        '''
        Solves for the reduced states of the DD-LS-ROM using the Lagrange-Newton-SQP algorithm. 
        
        inputs: 
        w0: initial interior/interface states and lagrange multipliers
        tol: [optional] stopping tolerance for Newton solver. Default is 1e-5
        maxit: [optional] max number of iterations for newton solver. Default is 20
        print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False
        rhs_hist: [optional] Boolean to return RHS vectors for each iteration. Default is False

        outputs:
        u_full: u state mapped to full domain
        v_full: v state mapped to full domain
        w_interior: list of interior states for each subdomain, i.e
                    w_interior[i] = interior state on subdomain i
        w_interface: list of interface states for each subdomain, i.e
                     w_interface[i] = interface state on subdomain i
        lam:    vector of optimal lagrange multipliers
        runtime: solve time for Newton solver
        itr: number of iterations for Newton solver
        '''
        
        print('Starting Newton solver...')
        y, res_vecs, res_hist, step_hist, itr, runtime = solvers.newton(self.FJac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
        print(f'Newton solver terminated after {itr} iterations with residual {res_hist[-1]:1.4e}.')
        
        # assemble solution on full domain from DD-ROM solution
        u_full = np.zeros(self.nxy)
        v_full = np.zeros(self.nxy)
        w_interior = list([])
        w_interface = list([])
        lam = y[-self.n_constraints:]

        shift = 0
        for s in self.subdomain:
            w_intr = y[np.arange(s.n_interior)+shift]
            shift += s.n_interior
            w_intf = y[np.arange(s.n_interface)+shift]
            shift += s.n_interface
            
            u_full[s.indices["interior"]] = s.u_intr_basis@w_intr
            v_full[s.indices["interior"]] = s.v_intr_basis@w_intr
            
            u_full[s.indices["interface"]] = s.u_intf_basis@w_intf
            v_full[s.indices["interface"]] = s.v_intf_basis@w_intf
            
            w_interior.append(w_intr)
            w_interface.append(w_intf)
        
        if rhs_hist:
            return u_full, v_full, w_interior, w_interface, lam, runtime, itr, res_vecs
        
        else:
            return u_full, v_full, w_interior, w_interface, lam, runtime, itr
    
    # Compute error between ROM and FOM solutions
    def compute_error(self, w_intr, w_intf, u_intr, v_intr, u_intf, v_intf):
        '''
        Compute error between ROM and FOM DD solutions.
        
        inputs:
        w_intr: list of reduced interior states where
                   w_intr[i] = (self.subdomain[i].n_interior,) vector of ROM solution interior states
        w_intf: list of reduced interface states where
                   w_intf[i] = (self.subdomain[i].n_interface,) vector of ROM solution interface states
                   
        u_intr: list of FOM u interior states where
                   u_intr[i] = (ddmdl.subdomains[i].n_interior,) vector of FOM u solution interior states
        v_intr: list of FOM v interior states where
                   v_intr[i] = (ddmdl.subdomains[i].n_interior,) vector of FOM v solution interior states
        u_intf: list of FOM u interface states where
                   u_intf[i] = (ddmdl.subdomains[i].n_interface,) vector of FOM u solution interface states
        v_intf: list of FOM v interface states where
                   v_intf[i] = (ddmdl.subdomains[i].n_interface,) vector of FOM v solution interface states
                   
        output:
        error: square root of mean squared relative error on each subdomain
        '''
        error = 0.0
        for i in range(self.n_sub):
            num = np.sum(np.square(u_intr[i]-self.subdomain[i].u_intr_basis@w_intr[i])) +\
                  np.sum(np.square(v_intr[i]-self.subdomain[i].v_intr_basis@w_intr[i])) +\
                  np.sum(np.square(u_intf[i]-self.subdomain[i].u_intf_basis@w_intf[i])) +\
                  np.sum(np.square(v_intf[i]-self.subdomain[i].v_intf_basis@w_intf[i]))

            den = np.sum(np.square(u_intr[i])) +\
                  np.sum(np.square(v_intr[i])) +\
                  np.sum(np.square(u_intf[i])) +\
                  np.sum(np.square(v_intf[i]))

            error += num/den
        error = np.sqrt(error/self.n_sub)
        return error