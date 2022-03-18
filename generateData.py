fout = open('symmetric_ramp_data_gen.txt','w')

fout.write("Time (s), Reactivity ($)\n")

t_max = 6
dt = 1.0E-3
n = t_max/dt

for i in range(int(n)):
    t = i/1000
    if i<=1000:
        rho_im = 0.5*t
    else:
        # rho_im = 0.5*beta_eff-0.5*beta_eff*(i-1000)/5000
        rho_im = 0.5-0.1*(t-1)
    
    fout.write("%.8E , %.8E \n\n" % (i/1000, rho_im))

fout.close()
