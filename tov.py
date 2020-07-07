import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import units as nu
from math import pi
# from polytropes import monotrope, polytrope
# from crust import SLyCrust
# from eoslib import get_eos, glue_crust_and_core, eosLib
from scipy.integrate import odeint, solve_ivp
# from label_line import label_line

import math

# from matplotlib import cm
import palettable as pal

cmap = pal.colorbrewer.qualitative.Set1_6.mpl_colormap


# cmap = pal.cmocean.sequential.Matter_8.mpl_colormap #best so far
# cmap = pal.wesanderson.Zissou_5.mpl_colormap

# --------------------------------------------------

class eos:
    # monotropic EoS
    # transition continuity constant
    a = 0.0
    c2 = nu.c**2

    def __init__(self, K, G):
        self.K = K / self.c2
        self.G = G
        self.n = 1.0 / (G - 1)

    # pressure P(density)
    def pressure_from_density(self, density):
        return self.c2 * self.K * density ** self.G

    # energy density_adm = density * (1 + energy)
    def rho_from_density(self, density):
        return (1.0 + self.a) * density + (self.K / (self.G - 1)) * density ** self.G

    # for inverse functions lets define  density (P)
    def density_from_pressure(self, pressure):
        if pressure < 0.0:
            return 0.0
        return (pressure / self.c2 / self.K) ** (1 / self.G)

    def densities_from_pressure(self, pressure):
        mask = np.array(pressure < 0.0, dtype=bool)
        out = (pressure / self.c2 / self.K) ** (1 / self.G)
        out[mask] = 0.0
        return out


class tov:

    def __init__(self, peos, r0=1e-8, rf=1e10):
        self.physical_eos = peos
        self.r0 = r0
        self.rf = rf

    def tov(self, y, r):
        P, mass, lna0 = y
        rho = self.physical_eos.density_from_pressure(P)

        dPdr = -nu.G * (rho + P / nu.c ** 2) * (mass + 4.0 * pi * r ** 3 * P / nu.c ** 2)
        dPdr = dPdr / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
        dmdr = 4.0 * pi * r ** 2 * rho

        dlnadr = (mass + 4*pi*r**3*P)/ (r * (r - 2.0 * nu.G * mass / nu.c ** 2))

        return [dPdr, dmdr, dlnadr]

    def tov_ivp(self, r, y):
        P, mass, lna0 = y
        rho = self.physical_eos.density_from_pressure(P)

        dPdr = -nu.G * (rho + P / nu.c ** 2) * (mass + 4.0 * pi * r ** 3 * P / nu.c ** 2)
        dPdr = dPdr / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
        dmdr = 4.0 * pi * r ** 2 * rho

        dlnadr = (mass + 4*pi*r**3*P)/ (r * (r - 2.0 * nu.G * mass / nu.c ** 2))

        return [dPdr, dmdr, dlnadr]

    def tovsolve(self, density_central):

        N = 800
        r0, rf = [self.r0, self.rf]
        r = np.linspace(r0, rf, N)
        P_0 = self.physical_eos.pressure_from_density(density_central)
        rho_0 = self.physical_eos.density_from_pressure(P_0)
        mass_0 = 4.0 * pi * r[0] ** 3 * rho_0
        lna_0 = 1

        # intmethod = "odeint"
        intmethod = "ivp"  # does not work for sys of diff.eqs ???

        if intmethod == "odeint":
            psol = odeint(self.tov, [P_0, mass_0, lna_0], r)  # , rtol=1.0e-4, atol=1.0e-4)
            press = psol[:, 0]
            mass = psol[:, 1]
            lna = psol[:, 2]
        elif intmethod == "ivp":
            sol = solve_ivp(self.tov_ivp, [r0, rf], [P_0, mass_0, lna_0])
            r = sol.t
            press = sol.y[0]
            mass = sol.y[1]
            lna = sol.y[2]
        else:
            return "error"

        return r, press, mass, lna

    def mass_radius(self): #TODO: not useful
        N = 10
        mcurve = np.zeros(N)
        rcurve = np.zeros(N)
        rhocs = np.logspace(14.0, 16.0, N)

        print(rhocs)

        mass_max = 0.0
        j = 0
        for rhoc in rhocs:
            rad, press, mass, lna = self.tovsolve(rhoc)

            # rad /= 1.0e5  # cm to km
            # mass /= nu.Msun

            mstar = mass[-1]
            rstar = rad[-1]
            atm_indx = []
            for i, p in enumerate(press):
                if p > 0.0:
                    mstar = mass[i]
                    rstar = rad[i]
                else:
                    atm_indx.append(i)
            mcurve[j] = mstar
            rcurve[j] = rstar

            j += 1
            if mass_max < mstar:
                mass_max = mstar
            else:
                pass
                # break

        return mcurve[:j], rcurve[:j], rhocs[:j]

    def solve_with_atm(self, density_central):

        rad, press, mass, lna = self.tovsolve(density_central)

        atm_indx = []
        p_min = np.max(press)
        for i, p in enumerate(press):
            if p > 0.0:
                p_min = p if p < p_min else p_min
            else:
                atm_indx.append(i)

        press[atm_indx] = p_min * 0.1
        density = physical_eos.densities_from_pressure(press)
        rho = physical_eos.rho_from_density(density)
        energy = (K * density ** (G - 1)) / (G - 1)

        chi = (1 - 2*mass/rad)

        out = dict()
        out['pressure'] = press
        out['density'] = density
        out['energy'] = energy
        out['lapse'] = np.exp(lna)
        out['rho'] = rho
        out['chi'] = chi
        out['mass'] = mass
        out['rad'] = rad

        return out

###########################################################

if __name__ == "__main__":
    # main(sys.argv)
    # plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    # plt.savefig('mr.pdf')

    plt.rcParams.update({'font.size': 22})

    K = 1e0
    G = 4./3

    physical_eos = eos(K, G)
    # eos = glue_crust_and_core(SLyCrust, dense_eos)
    t = tov(physical_eos)

    # mass, rad, rho_c = t.mass_radius()

    density_central = 1e-1
    # rad, press, mass, lna = t.tovsolve(density_central)
    # density = physical_eos.densities_from_pressure(press)
    # rho = physical_eos.rho_from_density(density)
    #
    # energy = (K * density**(G-1)) / (G-1)
    # omega = press / (density * energy)
    #
    # print("rho_c", density_central)
    # print("mass", mass)
    # print("rad", rad)
    #
    # fig, axs = plt.subplots(5, 1, figsize=(20,15))
    #
    # axs[0].plot(rad, press, label="P")
    # axs[1].plot(rad, mass, label="mass")
    # axs[2].plot(rad, density, label="density")
    # axs[2].plot(rad, rho, label="rho ADM")
    # axs[3].plot(rad, rho, label="rho ADM")
    # axs[4].plot(rad, omega, label="omega")
    #
    # for ax in axs[:4]:
    #     ax.set_yscale('log')
    #     ax.set_xscale('log')
    #     ax.legend()

    out = t.solve_with_atm(density_central)

    nplots = 7
    fig, axs = plt.subplots(nplots, 1, figsize=(20,15))

    keys = list(out.keys())
    keys.remove('rad')

    for i in range(nplots):
        var = keys[i]

        init = np.where( 0.01 <=  out['rad'])[0][0]
        end = np.where(out['rad'] >= 1e8)[0][0]

        axs[i].plot(out['rad'][init:end], out[var][init:end], label=var, lw =4 )
        # axs[i].plot(out['rad'], out[var], label=var)


    for ax in axs[:]:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.01, 1e8)
        ax.legend()

    plt.tight_layout()
    plt.savefig("idata.png")





