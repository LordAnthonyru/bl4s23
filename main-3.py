import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class Constants():
  m_e = 9.11 * 1e-31
  q_e = 1.6 * 1e-19
  c = 3 * 1e8
  eps_0 = 8.85 * 1e-12
  N_A = 6.02 * 1e23
  M_u = 1 * 1e-3
  h_bar = 1.05 * 1e-34


constants = Constants()


class Particle():

  def __init__(self, z, m_p, name):
    self.z = z
    self.m_p = m_p
    self.name = name


class Material():

  def __init__(self, rho, Z, A, I, name):
    self.rho = rho
    self.Z = Z
    self.A = A
    self.I = I
    self.n = (constants.N_A * Z * rho) / (A * constants.M_u)
    self.name = name


class Simulation():

  def __init__(self, particle: Particle, material: Material, E_0, x_max):
    self.particle = particle
    self.material = material
    self.E_0 = E_0
    self.x_max = x_max

  def solve(self):
    z = self.particle.z
    m_p = self.particle.m_p
    n = self.material.n
    I = self.material.I

    C1 = (4 * np.pi * n) / (constants.m_e * constants.c**2)
    C2 = (constants.q_e**2 / (4 * np.pi * constants.eps_0))**2
    C3 = (2 * constants.m_e * constants.c**2) / (I**2)

    omega_p = constants.q_e * \
        np.sqrt(n / (constants.eps_0 * constants.m_e))

    def model(x, E):
      beta = np.sqrt(1 - 1 / (1 + E / (m_p * constants.c**2))**2)
      gamma = np.power(1 - beta**2, -1 / 2)

      T_max = (2 * constants.m_e * constants.c**2 * beta**2 * gamma**2) / \
          (1 + (2 * gamma * constants.m_e / m_p) + (constants.m_e / m_p)**2)
      delta = 2 * np.log(constants.h_bar * omega_p / I) + 2 * np.log(
        beta * gamma) - 1

      dEdx = - C1 * C2 * (z**2 / beta**2) * \
          (1/2 * np.log(C3 * T_max * beta**2 /
                        (1 - beta**2)) - beta**2 - delta / 2)

      return dEdx

    solution = integrate.solve_ivp(model, [0, self.x_max], [self.E_0],
                                   rtol=1e-10,
                                   atol=1e-20)

    self.solved_x = solution.t
    self.solved_E = solution.y[0]
    self.distance = self.solved_x[-1]

  def plot_solution(self):
    plt.grid('on')

    energy_unit = ""
    energy = self.E_0 / constants.q_e

    if int(energy / 1e12) >= 1:
      energy_unit = "TeV"
      energy /= 1e12
    elif int(energy / 1e9) >= 1:
      energy_unit = "GeV"
      energy /= 1e9
    elif int(energy / 1e6) >= 1:
      energy_unit = "MeV"
      energy /= 1e6
    else:
      energy_unit = "keV"
      energy /= 1e3

    title_name = f"{self.particle.name}, {round(energy, 2)} {energy_unit}, {self.material.name}"
    plt.title(title_name, fontsize=24)
    plt.xlabel(r"$x [cm]$", fontsize=20)
    plt.ylabel(r"$E [eV]$", fontsize=20)

    plt.plot(100 * self.solved_x, self.solved_E / constants.q_e)
    plt.show()


proton = Particle(z=1, m_p=1.67 * 1e-27, name="Proton")

polyethylene = Material(rho=970,
                        Z=16,
                        A=28,
                        I=57.4 * constants.q_e,
                        name="Polyethylene")
aluminum = Material(rho=2700,
                    Z=13,
                    A=27,
                    I=166 * constants.q_e,
                    name="Aluminum")
water = Material(rho=1000, Z=10, A=18, I=75 * constants.q_e, name="Water")

simulation1 = Simulation(particle=proton,
                         material=aluminum,
                         E_0=10 * 1e9 * constants.q_e,
                         x_max=100)

simulation1.solve()
print("main.py")
print(f"Max distance: {round(100 * simulation1.distance, 2)} cm")
simulation1.plot_solution()
