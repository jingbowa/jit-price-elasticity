import numpy as np
from NumbaMinpack import minpack_sig, lmdif
from numba import cfunc, njit, prange

def Complements_data(m_price = np.array([0.5, 0.5, 0.5])):
    m, j = 20000, m_price.shape[0]
    u = np.random.rand(m, j) - 0.5
    z = np.random.rand(m, j) - 0.5
    p_jt = m_price + 0.5*z + 0.5*u + 0.5*np.random.rand(m, j) - 0.25
    p_jt_aug = np.hstack((p_jt, (p_jt[:,[0]] + p_jt[:,[1]])))
    cfd = np.hstack((u, (u[:,[0]] + u[:,[1]]))) 
    delta_i, alpha_i = 0.4, 3
    gamma = np.array([0., 0., 0., 1.5]).reshape(1, 4)
    u_jt = delta_i - alpha_i*p_jt_aug + gamma + cfd

    temp = np.exp(u_jt)
    temp[:, 0] = temp[:, 0] + temp[:, -1]
    temp[:, 1] = temp[:, 1] + temp[:, -1]
    temp = temp[:, :-1]
    s_jt = temp/(1 + np.sum(temp, axis = 1, keepdims=True))
    return s_jt[:,0], s_jt[:,1], np.hstack((p_jt, z[:,[0]])), p_jt[:,0], z[:,0]

def sales_p_1d_extra(price, u):
    p_jt_aug = np.array(list(price) + [price[0] + price[1]])
    
    cfd = np.hstack((u, (u[:,[0]] + u[:,[1]])))
    delta_i, alpha_i = 0.4, 3 
    gamma = np.array([0., 0., 0., 1.5]).reshape(1, 4)

    u_jt = delta_i - alpha_i*p_jt_aug + gamma + cfd
    
    temp = np.exp(u_jt)
    temp[:, 0] = temp[:, 0] + temp[:, -1]
    temp[:, 1] = temp[:, 1] + temp[:, -1]
    temp = temp[:, :-1]
    s_jt = temp/(1 + np.sum(temp, axis = 1, keepdims=True))
    return s_jt

def Complements_stats(price = np.array([0.5, 0.5, 0.5])):    
    m, j = 20000, 3
    u = np.random.rand(m, j) - 0.5
    
    step = 0.1

    price_f, price_b = np.copy(price), np.copy(price)
    price_f[0] = price_f[0] + step/2
    price_b[0] = price_b[0] - step/2

    s_jt = sales_p_1d_extra(price, u)
    s_jt_f, s_jt_b = sales_p_1d_extra(price_f, u), sales_p_1d_extra(price_b, u)

    elasticity = np.mean((s_jt_f - s_jt_b)/s_jt*price[0]/step, axis = 0)

    own, cross = elasticity[0], elasticity[1]
    return own, cross

# (0, alpha_1); (1, alpha_2); (2, alpha_3)
# (3, gamma_1); (4, gamma_2); (5, gamma_3)
# (6, p_1); (7, p_2); (8, p_3)
# (9, cfd_1); (10, cfd_2); (11, cfd_3); (12, e)

@cfunc(minpack_sig)
def optimize_full(x, fvec, args):
    a = args[12] - x[0]*args[6] - x[1]*args[7]
    b = args[8]*args[5]
    temp = args[11]*((a/b + 1) ** (args[2] - 1))/args[8]

    fvec[0] = ((x[0]/args[3] + 1) ** (args[0] - 1))*args[9] - args[6]*temp
    fvec[1] = ((x[1]/args[4] + 1) ** (args[1] - 1))*args[10] - args[7]*temp
funcptr = optimize_full.address

@cfunc(minpack_sig)
def optimize_23(x, fvec, args):
    a = args[12] - x[1]*args[7]
    b = args[8]*args[5]
    temp = args[11]*((a/b + 1) ** (args[2] - 1))/args[8]

    fvec[0] = - x[0] ** 2 #auxiliary 
    fvec[1] = ((x[1]/args[4] + 1) ** (args[1] - 1))*args[10] - args[7]*temp
funcptr_23 = optimize_23.address

@cfunc(minpack_sig)
def optimize_13(x, fvec, args):
    a = args[12] - x[0]*args[6]
    b = args[8]*args[5]
    temp = args[11]*((a/b + 1) ** (args[2] - 1))/args[8]

    fvec[0] = ((x[0]/args[3] + 1) ** (args[0] - 1))*args[9] - args[6]*temp
    fvec[1] = - x[1] ** 2 #auxiliary
funcptr_13 = optimize_13.address

@cfunc(minpack_sig)
def optimize_12(x, fvec, args):
    a = args[12] - x[0]*args[6]
    b = args[7]*args[4]
    temp = args[10]*((a/b + 1) ** (args[2] - 1))/args[7]

    fvec[0] = ((x[0]/args[3] + 1) ** (args[1] - 1))*args[9] - args[6]*temp
    fvec[1] = -x[1] ** 2 #auxiliary
funcptr_12 = optimize_12.address

@cfunc(minpack_sig)
def optimize_aux(x, fvec, args):
    a = args[0]
    fvec[0] = -x[0] ** 2 #auxiliary
    fvec[1] = -x[1] ** 2 #auxiliary
funcptr_aux = optimize_aux.address

@njit
def refine(args, p, e):
    init_0 = np.array((1.5, 1.5))
    init_1 = np.array((1.5, 0.))
    init_2 = np.array((0., 1.5))
    init_3 = np.array((0., 0.))
    x1, x2 = lmdif(funcptr, init_0, 2, args)[0]
    x3 = (e - x1*p[0] - x2*p[1])/p[2]
    while x1<0 or x2<0 or x3<0:
        if x1<0 and x2 >=0 and x3 >=0:
            x1, x2 = lmdif(funcptr_23, init_2, 2, args)[0]
            x3 = (e - x2*p[1])/p[2]
        elif x1>=0 and x2 <0 and x3 >=0:
            x1, x2 = lmdif(funcptr_13, init_1, 2, args)[0]
            x3 = (e - x1*p[0])/p[2]
        elif x1>=0 and x2 >0 and x3 <0:
            x1, x3 = lmdif(funcptr_12, init_1, 2, args)[0]
            x2 = (e - x1*p[0])/p[1]
        elif x1<0 and x2<0 and x3 >=0:
            x1, x2 = lmdif(funcptr_aux, init_3, 2, args)[0]
            x3 = e/p[2]
        elif x1<0 and x2>=0 and x3 <0:
            x1, x3 = lmdif(funcptr_aux, init_3, 2, args)[0]
            x2 = e/p[1]
        elif x1>=0 and x2<0 and x3 <0:
            x1 = e/p[0]
            x2, x3 = lmdif(funcptr_aux, init_3, 2, args)[0]
    return np.array((x1, x2, x3))

@njit(parallel = True)
def sales_p_jt(p_jt, alpha, gamma, u, e):
    m, j = u.shape
    j, n_i = alpha.shape
    purchases = np.zeros((m, j, n_i))
    for ind in range(n_i):
        alpha_i = alpha[:, ind].copy().reshape(1, j)
        gamma_i = gamma[:, ind].copy().reshape(1, j)
        e_i = e[ind]
        e_stack = np.array(e[ind]).reshape(1, 1)
        for mkt in range(m):
            p = p_jt[mkt, :]
            cfd = u[mkt, :]
            args=np.hstack((alpha_i, gamma_i, p.reshape(1, j), cfd.reshape(1, j), e_stack)).reshape(13,)
            purchases[mkt, :, ind]= refine(args, p, e_i)    
    s_jt = purchases
    return s_jt

@njit(parallel = True)
def sales_p_1d(price, alpha, gamma, u, e):
    m, j = u.shape
    j, n_i = alpha.shape
    purchases = np.zeros((m, j, n_i))
    for ind in range(n_i):
        alpha_i = alpha[:, ind].copy().reshape(1, j)
        gamma_i = gamma[:, ind].copy().reshape(1, j)
        e_i = e[ind]
        e_stack = np.array(e[ind]).reshape(1, 1)
        for mkt in range(m):
            p = price
            cfd = u[mkt, :]
            args=np.hstack((alpha_i, gamma_i, p.reshape(1, j), cfd.reshape(1, j), e_stack)).reshape(13,)
            purchases[mkt,:, ind]= refine(args, p, e_i)
    s_jt = purchases
    return s_jt

def Variety_data(m_price = np.array([0.5, 0.5, 0.5])):
    m, j, n_i = 20000, 3, 100
    
    e = np.random.rand(n_i)*0.1 + 5.0

    u = np.random.rand(m, j)*0.1 + 1.0 #uniform [1.0, 1.1)
    z = np.random.rand(m, j) - 0.5 #uniform [-0.5, 0.5)

    m_price = np.array([0.5, 0.5, 0.5])

    p_jt = m_price + 0.5*z + 0.5*(u-1) # [0, 1)

    alpha = np.zeros((j, n_i))+np.array((0.2, 0.4, 0.6))[:, np.newaxis]
    gamma = np.random.rand(j, n_i)*0.1 + 1.1

    s_jt = np.mean(sales_p_jt(p_jt, alpha, gamma, u, e), axis = 2)
    return s_jt[:,0], s_jt[:,1], np.hstack((p_jt, z[:,[0]])), p_jt[:,0], z[:,0]

def Variety_stats(price = np.array([0.5, 0.5, 0.5])):
    m, j, n_i = 20000, 3, 100
    e = np.random.rand(n_i)*0.1 + 5.0
    step = 0.1

    price_f, price_b = np.copy(price), np.copy(price)
    price_f[0] = price_f[0] + step/2
    price_b[0] = price_b[0] - step/2

    u = np.random.rand(m, j)*0.1 + 1.0
    alpha = np.zeros((j, n_i))+np.array((0.2, 0.4, 0.6))[:, np.newaxis]
    gamma = np.random.rand(j, n_i)*0.1 + 1.1

    s_jt = np.mean(sales_p_1d(price, alpha, gamma, u, e),axis=0)
    s_jt_f = np.mean(sales_p_1d(price_f, alpha, gamma, u, e),axis=0)
    s_jt_b = np.mean(sales_p_1d(price_b, alpha, gamma, u, e),axis=0)

    elasticity = (s_jt_f - s_jt_b)/s_jt*price[0]/step

    own = np.mean(elasticity, axis = 1)[0]
    cross = np.mean(elasticity, axis = 1)[1]
    return own, cross#, np.mean(s_jt, axis = 1)[0]