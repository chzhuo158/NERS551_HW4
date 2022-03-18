#!/usr/bin/python3

import os

import sys

import numpy as np

import matplotlib.pyplot as plt



if sys.version_info[0] < 3:

    raise Exception("Python 3 or a more recent version is required.")



# Parameters

reactivity_inputfile='symmetric_ramp_data_gen.txt'
# reactivity_inputfile='symmetric_ramp_data.txt'

P_0 = 1.0               # Initial power



## one-group parameters

Lambdas=[0.49405] # Group decay constants (1/s)

Betas = [0.0076]     # Delayed neutron fractions



## six-group parameters

# Lambdas=[0.0128,0.0318,0.119,0.3181,1.4027,3.9286]

# Betas=[0.0002584,0.00152,0.0013908,0.0030704,0.001102,0.0002584]



## Precursor concentration

Zetas=np.zeros(len(Betas))

for i in range(len(Betas)):

    Zetas[i]=Betas[i]/Lambdas[i] # Initial precursor concentration = beta*P/lambda



LAMBDA_0 = 2.6E-15      # Neutron lifetime (s)

lambd = 0.49405

beta_eff = np.sum(Betas)



assert(len(Lambdas) == len(Betas)), "Need the same number of lambdas and betas"

assert(len(Lambdas) == len(Zetas)), "Need the same number of lambdas and zetas"



lambd_H = 0.0

gamma_d = 0.0



ffp = 1.0

ffp_prev = 1.0



theta = 0.5





def analytic_PJA(P_s, rho_s, lambd, drho_dt, dt):

    ###########################################################################

    # NOTE: Solves the Prompt Jump Approximation (PJA) for linear ramps.

    # This means one delayed neutron group and no temperature feedback.

    ###########################################################################

    # PARAMETERS:

    # P_s       = Power at time linear ramp drho_dt is imposed

    # rho_s     = Reactivity at time drho_dt is imposed ($)

    # lambd     = 1 group delayed neutron precursor decay constant (1/s). 

    #               Note lambda is a reserved word, hence use of lambd

    # drho_dt   = Constant linear reactivity change ($/s)

    # dt        = Time after linear ramp drho_dt is imposed, for which power is 

    #               to be found (s)

    ###########################################################################

    

    # Check for proper function arguments

    assert(P_s > 0), "Power must be greater than 0."

    assert(lambd >= 0), "Delayed neutron precursor decay constant must be non-negatve."

    assert(dt >= 0), "Time to evaluate must be non-negative." 



    if drho_dt != 0:

        tau = (1 - rho_s)/drho_dt

        c = lambd/drho_dt + 1

        #print('pja',P_s, rho_s, lambd, drho_dt, dt)

        return P_s*np.exp(-lambd*dt)*(tau/(tau - dt))**c

    else:

        assert(rho_s != 1), "For a ramp rate of 0, initial reactivity cannot be a dollar."

        return P_s*np.exp(lambd*rho_s*dt/(1-rho_s))



def analytic_PJA_2part(P_0, rho_0, lambd, rho_1, t_s, rho_2, t):

    ###########################################################################

    # NOTE: Solves the Prompt Jump Approximation (PJA) for two part linear 

    # ramps. This means one delayed neutron group and no temperature feedback.

    ###########################################################################

    # PARAMETERS:

    # P_0       = Power at time linear the first linear ramp (rho_1) is imposed

    # rho_0     = Reactivity at time rho_1 is imposed ($)

    # lambd     = 1 group delayed neutron precursor decay constant (1/s). 

    #               Note lambda is a reserved word, hence use of lambd

    # rho_1     = First constant linear reactivity change ($/s)

    # t_s       = Time at which second reactivity change rho_2 starts (s)

    # rho_2     = Second constant linear reactivity change ($/s)

    # t         = Time after linear ramp rho_1 is imposed, for which power is 

    #               to be found (s)

    ###########################################################################

    if t <= t_s:

        return analytic_PJA(P_0, rho_0, lambd, rho_1, t)

    else:

        rho_01 = t_s*rho_1

        P_01 = analytic_PJA(P_0, rho_0, lambd, rho_1, t_s)

        return analytic_PJA(P_01, rho_01, lambd, rho_2, t-t_s)







def k0(x):

    return 1.0-x/2.0+x*x/6.0-x*x*x/24.0+x*x*x*x/120.0-x*x*x*x*x/720.0

    #return (1-np.exp(-x))/x



def k1(x):

    return 0.5-x/6.0+x*x/24.0-x*x*x/120.0+x*x*x*x/720.0

    #return (1-k0(x))/x



# Load reactivity insertion data

current_dir = os.getcwd()

data = np.loadtxt(current_dir + "/" + reactivity_inputfile, skiprows=1, delimiter=',')

t = data[:,0]                      # Times

rho_im = data[:,1]*beta_eff          # Inserted reactivity at that time



# Allocate arrays based on data and parameters

rho = np.zeros(len(t))             # Total reactivity = inserted + feedback

P = np.zeros(len(t)) + P_0         # Power

LAMBDA = np.zeros(len(t)) + LAMBDA_0    # Neutron lifetime

tau = np.zeros(len(t))

S_d = np.zeros(len(t)) 

S_dhat = np.zeros(len(t)) 



groups = len(Lambdas)

beta = np.zeros(( len(t), groups )) 

for index, val in enumerate(Betas):

    beta[:,index] = val





zeta = np.zeros(( len(t), groups )) 

for index, val in enumerate(Zetas):

    zeta[:,index] = val



zeta_hat = np.zeros(( len(t), groups )) 



lambdas = np.zeros(( len(t), groups ))

lambda_t = np.zeros(( len(t), groups )) 

for index, val in enumerate(Lambdas):

    lambdas[:, index] = val





OMEGA = np.zeros(( len(t), groups ))

G = np.zeros(( len(t),groups))



fout = open('Variables.txt','w')

fout.write(" n       t(s)    delta t    alpha_n lambda_h_tilda lambda_h_hat  zeta_n     rho_n       p_n\n")



# Compute solution to PKE

for n in range(1,len(t)):

    # Step 1

    dt_nm1 = (t[n] - t[n-1])    # Delta t_(n-1)

    alpha_n = (1.0/dt_nm1)*np.log(P[n-1]/P[n-2])    # Eq 23

    if n==len(t)-1:

        dt_n = (t[n] - t[n-1])

    else:

        dt_n = (t[n+1] - t[n])

    ## self defined reactivity insertion

    #if i<=int(t1/dt):

    #    rho_im[n] = 5.0*beta_eff*i*dt

    #else:

    #    rho_im[n] = 5.0*t1*beta_eff-5.0*beta_eff*(i-int(t1/dt))*dt

    beta_eff_n = np.sum(beta[n,:])

    

    for k in range(groups):

        lambda_t[n,k] = (lambdas[n,k] + alpha_n)*dt_n                      # Eq 9

        OMEGA[n,k] = (LAMBDA_0/LAMBDA[n])*beta[n,k]*dt_n*k1(lambda_t[n,k]) # Eq 15

        G[n-1,k] = (LAMBDA_0/LAMBDA[n-1])*beta[n,k]*P[n-1]                 # Eq 8

        zeta_hat[n,k] = np.exp(-lambdas[n,k]*dt_n)*zeta[n-1,k]\
            + np.exp(alpha_n*dt_n)*dt_n*G[n-1,k]\
            *(k0(lambda_t[n,k])-k1(lambda_t[n,k]))                         # Eq 16

 

    # Step 2

    for k in range(groups):

        tau[n] = tau[n] + lambdas[n,k]*OMEGA[n,k]                          # Eq 18

        S_dhat[n] =  S_dhat[n] + lambdas[n,k]*zeta_hat[n,k]                # Eq 18

        S_d[n-1] = S_d[n-1]+lambdas[n-1,k]*zeta[n-1,k]                     # Eq 19

        

    # Step 3

    lambd_H_tilda = (lambd_H+alpha_n)*dt_n                                 # Eq 29

    lambd_H_hat = lambd_H*dt_n                                             # Eq 28

    rho_d_prev = rho[n-1]-rho_im[n-1]                                      # Eq 27



    a1 = P[n]/P[0]*gamma_d*dt_n*k1(lambd_H_tilda)                          # Eq 32

    b1 = rho_im[n]+np.exp(-lambd_H_hat)*rho_d_prev\
        -P[0]*gamma_d*dt_n*k0(lambd_H_hat)*P_0 \
        +np.exp(alpha_n*dt_n)*gamma_d*dt_n \
        *(k0(lambd_H_tilda)-k1(lambd_H_tilda))                             # Eq 33

 

    # step 4

    a = theta*dt_n*a1/LAMBDA[n]                                            # Eq 36

    b = theta*dt_n*(((b1-beta_eff_n)/LAMBDA[n]-alpha_n)+tau[n]/LAMBDA_0)-1   # Eq 37

    c = theta*dt_n/LAMBDA_0*S_dhat[n]+np.exp(alpha_n*dt_n)\
        *((1-theta)*dt_n*((((rho[n-1]-beta_eff_n)/LAMBDA[n-1]-alpha_n)\
        *P[n-1]+S_d[n-1]/LAMBDA_0)+P[n-1]))                                # Eq 38



    # step 5

    if a<-1e-14:

        P[n] = (-b-np.sqrt(b*b-4*a*c))/(2*a)                               # Eq 39

    else:

        P[n] = -c/b                                                        # Eq 40



    # step 6

    delta_p_left = np.abs(P[n] - np.exp(alpha_n*dt_n)*P[n-1])

    #if dt_n < 0.00001:

    #    dt_n = t[i-1]-t[i-2]

    delta_p_right = np.abs(P[n]-P[n-1]-(P[n-1]-P[n-2])*dt_nm1/dt_n)

    #while delta_p_left > delta_p_right:

    if delta_p_left > delta_p_right:

        alpha_n = 0.0

        # Step 3

        lambd_H_tilda = (lambd_H+alpha_n)*dt_n                                 # Eq 29

        lambd_H_hat = lambd_H*dt_n                                             # Eq 28

        rho_d_prev = rho[n-1]-rho_im[n-1]                                      # Eq 27



        a1 = P[n]/P[0]*gamma_d*dt_n*k1(lambd_H_tilda)                          # Eq 32

        b1 = rho_im[n]+np.exp(-lambd_H_hat)*rho_d_prev\
            -P[0]*gamma_d*dt_n*k0(lambd_H_hat)*P_0\
            +np.exp(alpha_n*dt_n)*gamma_d*dt_n\
            *(k0(lambd_H_tilda)-k1(lambd_H_tilda))                             # Eq 33

    

        # step 4

        a = theta*dt_n*a1/LAMBDA[n]                                            # Eq 36

        b = theta*dt_n*(((b1-beta_eff_n)/LAMBDA[n]-alpha_n)+tau[n]/LAMBDA_0)-1   # Eq 37

        c = theta*dt_n/LAMBDA_0*S_dhat[n]+np.exp(alpha_n*dt_n)\
            *((1-theta)*dt_n*((((rho[n-1]-beta_eff_n)/LAMBDA[n-1]-alpha_n)\
            *P[n-1]+S_d[n-1]/LAMBDA_0)+P[n-1]))                                # Eq 38

    

        # step 5

        if a<0:

            P[n] = (-b-np.sqrt(b*b-4*a*c))/(2*a)                               # Eq 39

        elif a==0:

            P[n] = -c/b     

            

        #delta_p_left = np.abs(P[n] - np.exp(alpha_n*dt_n)*P[n-1])

        #delta_p_right = np.abs(P[n]-P[n-1]-(P[n-1]-P[n-2])*dt_nm1/dt_n)                                         # Eq 40

     

    rho[n] = a1*P[n]+b1

    for k in range(groups):

        zeta[n,k] = P[n]*OMEGA[n,k] + zeta_hat[n,k]



    # print(S_dhat[n])



    fout.write("%3i  %9.6f  %9.6f  %9.6f  %9.6f     %9.6f  %9.6f  %9.6f  %9.6f %9.6f\n" % (n, t[n], dt_n, alpha_n, lambd_H_tilda, lambd_H_hat, zeta[n,0], rho[n], P[n], G[n-1,0]))



fout.close()

 

# Generate analytical solution

N = len(t)

drho_dt = 0.5  # slope of reactivity insertion rate

rho_s = 0.0    # starting time

P_s = P_0      # starting power

t1 = 1.0      # peak of reactivity ramp

result_analytical = np.zeros(N)

for i in range(N):

    if t[i] <= t1:

        result_analytical[i] = analytic_PJA(P_s, rho_s, lambd, drho_dt, t[i])

        P_s1 = result_analytical[i]

    else:

        rho_s1 = drho_dt*t1

        result_analytical[i] = analytic_PJA(P_s1, rho_s1, lambd, -drho_dt/5.0, t[i]-t1)



deltaP = (P[int((N-1)/6)]-result_analytical[int((N-1)/6)])/result_analytical[int((N-1)/6)]

# print(deltaP)

# N = len(rho_im)

# result = np.zeros(N)

# result2 = np.zeros(N)

# rho_s = 0

# rho_st = 0

# lambd = 0.49405

# P_s = 1.0

# #t = np.linspace(0, 0.3, num=N)

# #rho = np.zeros(N)

# rho = rho_im

# for i in range(N):

#     #print('call2')

#     result2[i] = analytic_PJA_2part(P_s, rho_st, lambd, 5, 0.15, -5, t[i])

#     #print('resulst')

#     #print(i,t[i],result[i], result2[i])

#     #rho[i] = rho[i-1] + drho_dt*(t[1]-t[0])



# store results

with open("results.out",'w') as fresult:

    # write title

    fresult.write(" time(second)  reactivity power(analytical) power(numerical) ")

    for k in range(groups):

        fresult.write("precursor(g="+str(k+1)+") ")

    fresult.write("\n")



    # write data

    for i in range(N):

        # write reactivity, analytical solution and numerical solution of power

        fresult.write("%12.8f %12.8f  %12.8f     %12.8f  " % (t[i],rho[i],result_analytical[i], P[i]))

        # write precursor concentration for each group

        for k in range(groups):

            fresult.write("   %12.8f" % zeta[i,k])

        fresult.write("\n")



# plot reactivity

reactivity_plot = plt.figure(1)

plt.plot(t, rho)

plt.xlabel('Time (s)')

plt.ylabel('Reactivity ($)')

plt.title('Reactivity Insertion vs Time')



# plot power

power_plot = plt.figure(2)

plt.plot(t, P,'+', label='numerical')

plt.plot(t, result_analytical, label='analytical')

plt.xlabel('Time (s)')

plt.ylabel('Power')

plt.title('Power vs Time')

plt.legend()



# plot precursor concentration

precursor_plot = plt.figure(3)

for k in range(groups):

    plt.plot(t, zeta[:,k], label='group '+str(k+1))

plt.xlabel('Time (s)')

plt.ylabel('Precursor concentration')

plt.title('Precursor concentration vs Time')



plt.legend()

plt.show()

