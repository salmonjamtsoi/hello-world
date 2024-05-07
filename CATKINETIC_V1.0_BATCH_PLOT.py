# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:48:44 2023

@author: huangxiaoyuan
"""
import os
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.integrate import odeint
import sympy
from sympy import symbols,solve,nsolve
import json
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time

#start_time = time.time()

open('Output_Scaling.csv', 'w').close()
params = pd.read_excel('INPUT_Scaling.xlsx')
param_G0 = params.iloc[:,2:11].values
param_Ea = params.iloc[:,11:20].values
def Kinetic_fsolve_module():
    # Ea[1,4], G0[1,4]
    for param_index in range(len(params)):
        Ea = param_Ea[param_index, :]
        G0 = param_G0[param_index, :]  
        print('Index',param_index)
        
        Reactions,Parameters,Initial = ReadINCAR(input_dict) # input_dict is from INPUT.json file
        Matrix, myBasic, mySite = ExtractCoefficient(Reactions)
        nReactions = len(Reactions)  # numbers of reactions
        T = Parameters['T']
        kB = 1.3806505e-23
        NA = 6.0221367e23
        h = 6.62606957e-34
        e = 1.60217653e-19
        Pa2bar = 1 / 100000
        keV = e / T/kB # not very understanding -- xu
        RT = kB * T * NA * Pa2bar
        kBT_h = kB / h * T # not very understanding -- xu
        kf, kr = RateConstCalc(Ea,G0,keV,kBT_h)
    
        [fp,fc,fq,fpr,fcr,fqr,r,r0,Qsum,osym0,ODEFun,Q_var,yNeed] = CreateRateEquations(Matrix,myBasic,mySite,RT,kf,kr,Parameters,Initial)
        cgMatrix,dGc = My_Check_Reaction2(Matrix, myBasic, mySite, G0)
        Q0 = Parameters['Q0']
        EssentialVar = ['Q0', 'T']
        for var in EssentialVar:
            if eval(f'not {var}'):
                print(f'\nThe essential variable: {var} is not assigned in INCAR.m')
                if var == 'Q0':
                    print('\nSet Initial Surface Active Site to 1')
                    Q0 = [1] * len(mySite)
                elif var == 'T':
                    print('\nSet Reaction Temperature to Room Temperature 298.15K')
                    T = 298.15
                else:
                    return
        if nReactions > len(Ea):
            print(f'\nThe number of Energy barriers (Ea) {len(Ea)} is less than the number of Reaction Equation {nReactions}')
            return
        if nReactions > len(G0):
            print(f'\nThe number of Gibbs free Energy (G0) {len(G0)} is less than the number of Reaction Equation {nReactions}')
            return
        y0 = Q_var
        t = [0,100]
        y0_solved = odeint(ODEFun, y0, t)
        yInit_array = y0_solved[1:]
        a,b = yInit_array.shape
        yInit_list  = []
        for i in range(a):
            for k in range(b):
                yInit_list.append(yInit_array[i,k])
        yInit = []
        for i in range(len(yNeed)):
            if yNeed[i] == True:
                yInit.append(yInit_list[i])
        
        yChan, yp,Q_need= Initialization(osym0,fpr, fcr, fqr,Parameters)    
        
        fqr = list(fqr)
        fqt=[]
        for i in range(len(fqr)):
            if fqr[i] != 0:
                fqt.append(fqr[i])
        n = len(Qsum)
        for ix in range(n):
            fq[ ix,-n + ix] = Qsum[ix]- Q0[ix]
        # Return variables
        new_fp = np.array(fp).reshape(-1,1)
        new_fc = np.array(fc).reshape(-1,1)
        new_fq = np.array(fq).reshape(-1,1)
        F = np.column_stack((new_fp, new_fc, new_fq))
        F = (F.T).flatten()
        F = F[yp]
              
        
        F = F[yChan].reshape(-1,1)
        ix,iy = F.shape
        
        Fs = []
        for i in range (ix):
            for k in range (iy):
                Fs.append(F[i,k])
        Q_s = []
        for i in range(len(Q_need)):
            Q_s.append(symbols(Q_need[i]))
#        solved_value = solve(Fs,Q_s)
#        print('solved_value',solved_value)
    
        solved_value_matrix = nsolve(Fs, Q_s,yInit,verify =False)
        solved_value = list(solved_value_matrix)
#    print('solved_value2',solved_value)
    
    #过滤无效值，有效值转为字典存储
        result_dict = dict()
        for i in range(len(solved_value)):
            result_dict[Q_s[i]]= solved_value[i]
            if solved_value[i] < 0 or solved_value[i] > 1.0:
                print('!!Attention!!Solution is out og range (0,1),please treat Rnet results with caution!')
       
        print("\nresult_dict:\n",result_dict)
    
        r = r.reshape(-1,1)
        m,n = r.shape
        Rnet = np.ones((m, n))
        for i in range(m):
            for j in range(n):
                Rnet[i,j]=abs(r[i,j].evalf(subs=result_dict))
        
        l,p = r0.shape
        Rfr = np.ones((l, p))
        for i in range(l):
            for j in range(p):
                Rfr[i,j]=r0[i,j].evalf(subs=result_dict)
        min_value = np.zeros((2,1))
        min_value[0,0] = min(Rnet[:,0])
        min_value[1,0] = math.log10(min_value[0,0])
        R_net = np.vstack((Rnet,min_value))
        R_net = pd.DataFrame(R_net)
        R_net = R_net.T
        R_net.to_csv('Output_Scaling.csv', mode='a', header=False, index=None)

    x = params['x1']
    y = params['x2']
    data_output = pd.read_csv('Output_Scaling.csv',header=None)
    z = data_output.iloc[:,10]
    
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    xi,yi = np.meshgrid(xi,yi)
    os.startfile('Output_Scaling.csv')
    
    print("Before griddata call:")
    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("xi:", xi)
    print("yi:", yi)
    
    zi = griddata((x,y),z,(xi,yi),method= 'cubic')
    
    print("After griddata call:")
    print("zi:", zi)
    
    plt.contourf(xi,yi,zi,21,cmap = 'plasma')
    C = plt.contour(xi,yi,zi,21,cmap = 'plasma')
    
    plt.clabel(C, inline = True,colors = 'k',fmt = '%1.2f')
    plt.colorbar(C)
    plt.title('log10(Rnet)Volcano_Curve')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.show()

      
def ReadINCAR(input_dict):

    Reactions = []
    Parameters = []
    Initial = []

    n_Reactions = len(input_dict['Reactions'])   
    for i in range(n_Reactions):  
        varlue_list = input_dict['Reactions'][i].values()
        Reactions += varlue_list

    Parameters = input_dict['ReactionConditions'][0]
    Initial = input_dict['InitialConditions'][0] 
    return Reactions,Parameters,Initial

def ExtractCoefficient(Reactions): # extract coefficient from reactions that are extracted from INCAR.json -- xu
    Sbasic = '#'
    mySite = []
    myBasic = []
    pofArrows = np.zeros(len(Reactions), dtype=int) # why creat an all-zero array of length len(Reactions)? -- xu
    blist = []
    mySite.append(Sbasic) # add the value of Sbasic at the bottom of mySite -- xu

    def mystrMatch(mystring, searchstring): # to find searchtring in mystring and give back the index -- xu
        strIsfound = 0                      # for what purpose of this code? -- xu
        for ix in range(len(mystring)):
            if mystring[ix] == searchstring:
                strIsfound = ix + 1
                break
        return strIsfound

    for ix in range(len(Reactions)):
        a = Reactions[ix]  # list a is a reaction -- xu
        pofArrows[ix] = a.find('<->')+1 # what purpose? -- xu
        a = a.replace('<->', '+') # why use + as a separator? -- xu 
        alist = a.split('+')
        blist.append(alist)
        Ns = len(mySite)


        for ik in range(Ns):
            alist = [item.replace(mySite[ik], '') for item in alist]

        for il in range(len(alist)):
            alist[il] = alist[il].replace('(p)', '')
            alist[il] = alist[il].replace('(c)', '')

            if Sbasic in alist[il]:
                Ns = len(mySite)
                mySite.append(Sbasic + str(Ns + 1))
                alist = [item.replace(mySite[Ns], '') for item in alist]

        for jx in range(len(alist)):
            if alist[jx]:
                strIsfound = mystrMatch(myBasic, alist[jx])
            else:
                break

            if strIsfound == 0:
                myBasic.append(alist[jx])
    
    myBasic.sort()  
    Ns = len(mySite)
    Nlist = len(blist)
    Matrix = np.zeros((Ns + 2,Nlist, len(myBasic) + Ns ))
    for ix in range(len(blist)):
        mark = 0
        for jx in range(len(blist[ix])):
            alist = blist[ix]
            mark += len(alist[jx]) + 1

            if mark <= pofArrows[ix]:
                sign = -1
            else:
                sign = 1

            psite0 = mystrMatch(myBasic, alist[jx]) + mystrMatch([item + "(p)" for item in myBasic], alist[jx])
            if psite0:
                Matrix[-1, ix, psite0-1] += sign
                continue

            psite1 = mystrMatch([item + "(c)" for item in myBasic], alist[jx])

            if psite1:
                Matrix[-2, ix,psite1 ] += sign
                continue

            for kx in range(Ns):
                psitea = mystrMatch([item + mySite[kx] for item in myBasic], alist[jx])

                if psitea:
                    Matrix[kx, ix, psitea-1] += sign
                    break

            psites = mystrMatch(mySite, alist[jx])
            if psites:
                Matrix[ psites-1, ix,len(myBasic)-1 + psites] += sign

    return Matrix, myBasic, mySite # not very understanding ExtractCoefficient function need further reading -- xu

def RateConstCalc(Ea, G0,keV,kBT_h):
    Keq=[]
    kf = []
    kr = []
    for i in range (len(G0)):
        Keq.append( np.exp(-G0[i] * keV))
        kf.append( np.exp(-Ea[i] * keV) * kBT_h)
        kr.append( kf[i] / Keq[i])
    return kf, kr

def CreateRateEquations(Matrix,myBasic,mySite,RT,kf,kr,Parameters,Initial):

    Ns,M, N = Matrix.shape
    mySite0 = [ 'v' if i == '#' else i for i in mySite]
    myBasic = myBasic + mySite0

    P = ['P_'+myBasic[i] for i in range(len(myBasic))]
    P_var=[0 for _ in range (N)]
    P_Froz = []
    for key in Parameters:
        if key in P:
            Index = P.index(key) 
            P_var[Index] = Parameters[key]
            P_Froz.append(key)
    Q = ['Q_'+myBasic[i] for i in range(len(myBasic))]
    Q_list = []
    for i in range(len(Q)):
        Q_list.append(sympy.symbols(Q[i]))
    
    Q_var=[0 for _ in range (N)]
    for key in Initial:
        if key in Q:
            Index = Q.index(key) 
            Q_var[Index] = Initial[key]
#    C = ['C_'+myBasic[i] for i in range(len(myBasic))]
    C_var = [0 for _ in range (N)]
       
    rs = sympy.zeros(M,1)
    for ix in range(M):
        ixs = str(ix + 1)
        order = f"({ixs})"
        rs[ix,0] = sympy.symbols(f"r{order}")

    pMatrix = Matrix[Ns - 1,:, :]
    cMatrix = Matrix[Ns - 2,:, :]
    sMatrix = Matrix[Ns - 3,:, :]

    pfMatrix = np.copy(pMatrix)
    prMatrix = np.copy(pMatrix)
    cfMatrix = np.copy(cMatrix)
    crMatrix = np.copy(cMatrix)
    sfMatrix = np.copy(sMatrix)
    srMatrix = np.copy(sMatrix)

    pf = pMatrix > 0
    pr = pMatrix < 0
    cf = cMatrix > 0
    cr = cMatrix < 0
    sf = sMatrix > 0
    sr = sMatrix < 0

    pfMatrix[pr] = 0
    prMatrix[pf] = 0
    cfMatrix[cr] = 0
    crMatrix[cf] = 0
    sfMatrix[sr] = 0
    srMatrix[sf] = 0

    Pm = np.ones((M, 1))*P_var
    rF = Pm ** -prMatrix
    rR = Pm ** pfMatrix
    Cm = np.ones((M, 1)) *C_var
    rF *= Cm ** -crMatrix
    rR *= Cm ** cfMatrix
    for ix in range (Ns - 2):
        Qm = np.ones((M, 1))*Q_list
        rF = rF * Qm ** -srMatrix
        rR = rR * Qm ** sfMatrix
        
    tF = np.array(kf).reshape(-1, 1)
    tR = np.array(kr).reshape(-1, 1)

    for ix in range(N):
        tF = tF * rF[:, ix]
        tR = tR * rR[:, ix]
    tF = np.diagonal(tF)
    tR = np.diagonal(tR)

    r0 = np.column_stack([tF, tR])
    r = tF - tR  
    
    r_str = ''.join(map(str,r))
    r_list = r_str.split('*') 
    osym = []
    for i in range(len(r_list)):
        if r_list[i] in Q:
            osym.append(r_list[i])
    osym = list(set(osym))
    osym = sorted(osym)
    
    yNeed = [False]*len(Q)
    for i in range(len(Q)):
        if Q[i] in osym:
            yNeed[i]=True
       
    osym0 = sorted(P_Froz+osym)
    
    fp = np.zeros((1, N))
    tp = (RT * r).reshape(-1,1) * np.ones((1, N)) * pMatrix
    for iy in range(M):
        fp = fp + tp[iy, :]

    fc = np.zeros((1, N))
    tc = r.reshape(-1,1) * np.ones((1, N)) * cMatrix
    for iy in range(M):
        fc = fc + tc[iy, :]


    fq = sympy.zeros(Ns-2, N)
    for ix in range(Ns - 2):
        tq = r.reshape(-1,1) * np.ones((1, N)) * sMatrix
        for iy in range(M):
            test_tq = tq[iy,:].reshape(-1,1)
            fq[ix,:] = fq[ix,:] + test_tq.T      
       
    fpr = RT * np.sum(np.multiply(rs * np.ones((1, N)) , pMatrix),axis=0)
    fcr = np.sum(np.multiply(rs * np.ones((1, N)), cMatrix), axis=0)

    fqr = sympy.zeros(Ns-2, N)
    Qsum = sympy.zeros(1, Ns-2)
    
    Q_new = np.array(Q_list).reshape(1, -1)
    for ix in range(Ns - 2):
        fqr = np.sum(np.multiply(rs * np.ones((1, N)), sMatrix), axis=0)
        Qsum[ix] = np.sum(Q_new[ix, fqr != 0])
    def ODEFun(y,t):
        Q = y
        Ps = np.ones((M, 1))*P_var
        rFs = Ps ** -prMatrix
        rRs = Ps ** pfMatrix
        rFs *= Cm ** -crMatrix
        rRs *= Cm ** cfMatrix
        
        for ix in range (Ns - 2):
            Qs = np.ones((M, 1))*y
            rFs = rFs * Qs ** -srMatrix
            rRs = rRs * Qs ** sfMatrix
            
        tFs = np.array(kf).reshape(-1, 1)
        tRs = np.array(kr).reshape(-1, 1)

        for ix in range(N):
            tFs = tFs * rFs[:, ix]
            tRs = tRs * rRs[:, ix]
        tFs = np.diagonal(tFs)
        tRs = np.diagonal(tRs)

        rs = tFs - tRs  
        rss = rs.reshape(-1, 1)
        
        fqs = np.zeros((Ns-2, N))
        for ix in range(Ns - 2):
            fqs = np.sum(np.multiply(rss * np.ones((1, N)), sMatrix), axis=0) 
        
        fqs = fqs.reshape(-1, 1)
        a,b = fqs.shape 
        dydt = []
        for i in range(a):
            for k in range(b):
                dydt.append(fqs[i,k])
        return dydt
        
    return fp,fc,fq,fpr,fcr,fqr,r,r0,Qsum,osym0,ODEFun,Q_var,yNeed


def Initialization(osym0, fpr, fcr, fqr, Parameters):
    Nsyms = len(osym0)
    yChan = [False] * Nsyms

    for i in range(Nsyms):
        if osym0[i] in Parameters.keys():
            yChan[i] = False
        else:
            yChan[i] = True
    Q_need = [osym0[i] for i in range(Nsyms) if yChan[i]]

    new_fpr = np.array(fpr).reshape(-1,1)
    new_fcr = np.array(fcr).reshape(-1,1)
    new_fqr = np.array(fqr).reshape(-1,1)
    dy =np.column_stack((new_fpr, new_fcr, new_fqr))
    
    dy = (dy.T).flatten()
    yp = np.array(dy) != 0
    dy = dy[yp]
    dy = dy[yChan]

    return  yChan, yp,Q_need


def My_Check_Reaction2(Matrix,myBasic,mySite,G0 ):
    l, _, n = Matrix.shape
    Mcg = np.concatenate((Matrix[-2, :, :], Matrix[-1, :, :]), axis=1)
    for il in range(l - 2):
        Msur=Matrix[il, :, :]
    Msurt = Msur.T
    Msurt = sympy.Matrix(Msurt)
    solR_o =Msurt.nullspace()

    solR = np.array(solR_o[0])
    for i in range (1,len(solR_o)):
        solR = np.hstack((solR,np.array(solR_o[i])))
    solR = solR.T
 
    tsol = solR @ Mcg
    column_indices_1 = np.sum(np.abs(Mcg), axis=0) < 0
    column_indices_2 = np.sum(np.abs(Mcg), axis=0) > 0
    AT = np.hstack((-tsol[ :,column_indices_1], tsol[ :,column_indices_2]))
    AT = np.mat(AT)
    P, L = new_func1(AT)
    PL = P@L
    a_L,b_L = L.shape
    if a_L == b_L:
        LTM = L
    else:
        uper_half = np.eye(a_L-b_L)
        lower_half = np.zeros((b_L,a_L-b_L))
        half_L = np.vstack((uper_half,lower_half))
        LTM = np.hstack((PL,half_L))
    new_U = np.linalg.inv(LTM)
    solR = new_U @ solR
    tsol = solR @ Mcg
    AT = np.hstack((-tsol[:,column_indices_1], tsol[ :,column_indices_2]))
    
    Npsum = AT.shape[0]
    Asign = []
    
    for jm in range(Npsum):
        if np.all(AT[jm, :] == 0):
            Asign .append( 0 )
        elif np.all(AT[jm, :] > 0):
            Asign.append( -1 )
        elif np.all(AT[jm, :] < 0):
            Asign.append( 1)
        else:
            Asign .append( 1j)
    psign = []
    for i in range(len(Asign)):
        psign .append( Asign[i] != 0)
    psign = np.array(psign)
    
    ReaX = tsol[psign]
    Rsol = solR[psign] 
    Mcg_less = np.sum(Mcg,axis=0) < 0 
    Mcg_less_est  = Mcg_less.reshape(-1,1)
    RX_est = (psign * Mcg_less_est).T
    RX_row = -tsol[RX_est] 
    RX_row = RX_row.reshape(1,-1)
    a = len(ReaX)
    b,c= RX_row.shape
    d = int(c/a)
    if b == a :
        RX = RX_row
    else:
        RX = RX_row.reshape(a,d)
    Mcg_greater = np.sum(Mcg,axis=0) > 0 
    Mcg_greater_est  = Mcg_greater.reshape(-1,1) 
    PX_est = (psign * Mcg_greater_est).T
    PX_row = tsol[PX_est]
    PX_row = PX_row.reshape(1,-1)
    e,f = PX_row.shape
    g = int(f/a)
    if e == a:
        PX = PX_row
    else:
        PX = PX_row.reshape(a,g)
    MX = mynormMat(RX, PX)
    Rsum = MX @ ReaX
    Msum = MX @ Rsol 
    InCycle = solR[~psign, :]
    G0 = np.array(G0)
    dGc = InCycle @ G0.T
    if len(dGc) == 0 :
      dGc=np.array([0])
    
    statue = 0  
    if abs(dGc.any()) > 0.2:
        statue = 1
    for im in range(InCycle.shape[0]):
        dispRlist(InCycle[im, :], dGc[im])

    Nsite = len(mySite)
    Npsum = Msum.shape[0]
    cgMatrix = np.zeros((l,Npsum, n))

    for jm in range(Npsum):
        gMat = np.reshape(Rsum[jm, :], (2, n))
        cgMatrix[Nsite ,jm, :] = gMat[0, :]
        cgMatrix[Nsite + 1,jm, :] = gMat[1, :]
    cgMatrix = np.round(cgMatrix,3)
        
    Rs = Matrix2Reaction(cgMatrix, myBasic, mySite, 2)
    dGm = Msum @ G0.T
    
    for km in range(Npsum):
        print(f"Reaction : {km+1} : {Rs[km]}")
        dispRlist(Msum[km, :], float(dGm[km]))
    
    if statue == 1:
        print("Thermodynamic cycle(s) are not conserved. Pay attention to the results!")
    return cgMatrix,dGc

def new_func1(AT):
    P, L = new_func(AT)
    return P,L

def new_func(AT):
    print("AT type:", type(AT))
    print("AT shape:", AT.shape)
    P,L,U = sp.linalg.lu(AT)
    return P,L

def mynormMat(RX, PX):
    t0 = np.concatenate((RX, PX ),axis=1)
    t = np.copy(t0)
    nt, mt = t0.shape
    for jm in range(nt):
        nn = np.sum(t[jm, :]<0)
        nm = np.sum(t[jm, :] > 0)
        if np.all(t[jm, :] > 0) or nn > nm:
            t[jm, :] = -t[jm, :]

    ix = 0

    while ix < mt and np.any(t < 0):
        ix += 1
        for jm in range(nt):
            pn = np.where(t[jm, :] < 0)[0]
            for km in range(pn.size):
                if t[jm, pn[km]] < 0:
                    qn = np.where(t[:, pn[km]] > 0)[0]
                    if qn.size > 0:
                        cn = t[jm, pn[km]] / t[qn[0], pn[km]]
                        t[jm, :] = t[jm, :] - cn * t[qn[0], :]
                    else:
                        qn = np.where(t[:, pn[km]] < 0)[0]
                        qn = qn[qn != jm]
                        if qn.size > 0:
                            qn = qn[0]
                            cn = t[jm, pn[km]] / t[qn, pn[km]]
                            t[jm, :] = t[jm, :] + cn * t[qn, :]
                        else:
                            t[jm, :] = -t[jm, :]
    t = t.astype(float)
    t0 = t0.astype(float)
    if not np.any(t < 0):
        MX = np.linalg.lstsq(t0.T,t.T,rcond=None)[0]
        MX = MX.T
    else:
        MX = 1
    MX = np.round(MX,3)
    return MX


def dispRlist(RMat, dG):
    Rstr = f"dG = {dG:.2f} eV : "
    RMat = RMat.astype(float)
    RMat = np.round(RMat,3)
    RMat = RMat.reshape(1,-1) 
    for lm in range(RMat.shape[1]):
        nstr = ''        
        if RMat[0,lm] == 0:
            continue
        elif RMat[0,lm] == 1:
            nstr = ''
            addstr = ' + '
        elif RMat[0,lm] > 0:
            nstr = str(RMat[0,lm]) + ' '
            addstr = ' + '
        elif RMat[0,lm] == -1:
            nstr = ''
            addstr = ' - '
        elif RMat[0,lm] < 0:
            nstr = str(-RMat[0,lm]) + ' '
            addstr = ' - '               
        Rstr += addstr + nstr + f"R{lm+1} "
    
    print(Rstr)


def Matrix2Reaction(Matrix, myBasic, mySite, ptype=1):
    lM,mM, nM = Matrix.shape
    Mstr = [mySite[il] for il in range(lM - 2)]
    mySitet = ['' for _ in range(lM - 2)]
    Mstr.append('(c)')
    Mstr.append('(p)')
    mySpecies = myBasic + mySitet
    Rs = []

    for im in range(mM):
        Rt = []
        for fr in range(2):
            for jm in range(lM, 0, -1):
                for k in np.where((-1) ** (fr+1) * Matrix[ jm - 1,im, :] > 0)[0]:
                    nx = abs(Matrix[jm - 1,im, k ])
                    if ptype == 1:
                        for ns in range(nx):
                            Rt.append(mySpecies[k] + Mstr[jm - 1] + ' + ')
                    else:
                        nstr = ' '
                        if nx != 1:
                            nstr = str(nx) + ' '
                        Rt.append(nstr + mySpecies[k] + Mstr[jm - 1] + ' + ')

            Rt.append('<->')
        Rt.append(' - ')
        Rt = ''.join(Rt)
        Rt = Rt.replace(' + <->', ' <-> ')
        Rt = Rt.replace(' <->  - ', ' ')
        Rt = Rt.replace('<-><-> - ', ' ')
        Rt = Rt.replace(' <-> - ', ' ')
        Rs.append(Rt)

    return Rs

try:
    if __name__ == '__main__':
        with open('INCAR_PDH_full.json', encoding='UTF-8') as fid:
            input_dict = json.load(fid)
        Kinetic_fsolve_module()

finally:
    end_time = time.time()

    execution_time = end_time - start_time

    print('Execution Time: ',execution_time, "s")
