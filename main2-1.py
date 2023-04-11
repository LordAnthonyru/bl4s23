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

  def __init__(self, rho, Z, A, I, a, m, X0, X1, C, name):
    self.rho = rho
    self.Z = Z
    self.A = A
    self.I = I
    self.n = (constants.N_A * Z * rho) / (A * constants.M_u)
    self.name = name
    self.a = a
    self.m = m
    self.X0 = X0
    self.X1 = X1
    self.C = C


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
    a = self.material.a
    m = self.material.m
    X0 = self.material.X0
    X1 = self.material.X1
    C = self.material.C

    C1 = (4 * np.pi * n) / (constants.m_e * constants.c**2)
    C2 = (constants.q_e**2 / (4 * np.pi * constants.eps_0))**2
    C3 = (2 * constants.m_e * constants.c**2) / (I**2)

    def delta(E):
      X = np.log10(np.sqrt(1 - 1 / (1 + E / (m_p * constants.c ** 2)) ** 2)) \
          + np.log10(1 + E / (m_p * constants.c ** 2))

      if X < X0:
        return 0
      if X > X1:
        return 4.605 * X - C
      else:
        return 4.605 * X - C + a * ((X1 - X)**m)

    def model(x, E):
      if E <= 0.5e-13:
        return -100
      beta = np.sqrt(1 - 1 / (1 + E / (m_p * constants.c**2))**2)
      gamma = E/(m_p * constants.c**2) + 1

      T_max = (2 * constants.m_e * constants.c ** 2 * beta ** 2 * gamma ** 2) / \
              (1 + (2 * gamma * constants.m_e / m_p) + (constants.m_e / m_p) ** 2)

      # dEdx = - C1 * C2 * (z ** 2 / beta ** 2) * \
      #        (1 / 2 * np.log(C3 * T_max * beta ** 2 /
      #                        (1 - beta ** 2)) - beta ** 2 - delta(E) / 2)
      dEdx = - C1 * C2 * (z ** 2 / beta ** 2) * \
             (1 / 2 * np.log(C3 * T_max * beta ** 2 * gamma ** 2)
              - beta ** 2 - delta(E) / 2)
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


proton = Particle(z=1, m_p=1.6726 * 1e-27, name="Proton")
positron = Particle(z=1, m_p=9.109e-31, name="Positron")
polyethylene = Material(rho=970,
                        Z=16,
                        A=28,
                        I=57.4 * constants.q_e,
                        a=0.4875,
                        m=2.544,
                        X0=0.1379,
                        X1=2.0,
                        C=3.002,
                        name="Polyethylene")
aluminum = Material(rho=2700,
                    Z=13,
                    A=27,
                    I=166 * constants.q_e,
                    a=0.3346,
                    m=2.795,
                    X0=0.0966,
                    X1=2.5,
                    C=4.239,
                    name="Aluminum")
water = Material(rho=1000,
                 Z=10,
                 A=18,
                 I=75 * constants.q_e,
                 a=0.2065,
                 m=3.007,
                 X0=0.2400,
                 X1=2.5,
                 C=3.502,
                 name="Water")

print("main2.py")

simulation1 = Simulation(
  particle=proton,
  material=polyethylene,  # change the material
  E_0=15 * 1e9 * constants.q_e,  # change the energy
  x_max=100)

# simulation1.solve()
# print(f"Max distance: {round(100 * simulation1.distance, 2)} cm")
# simulation1.plot_solution()


def plot_distance_vs_energy(particle: Particle, material: Material, E_0_list,
                            x_max):

  distances = []
  for E_0 in E_0_list:
    simulation = Simulation(particle=particle,
                            material=material,
                            E_0=E_0,
                            x_max=x_max)
    simulation.solve()
    distance = simulation.distance
    distances.append(distance)

  plt.grid('on')
  plt.title(f"{particle.name}, {material.name}", fontsize=24)
  plt.xlabel("$E [eV]$")
  plt.ylabel("$d_{max} [cm]$")
  plt.plot(np.array(E_0_list) / constants.q_e, 100 * np.array(distances))
  plt.show()


def plot_distance_vs_momentum(particle: Particle, material: Material, p_0_list,
                            x_max):

  distances = []
  for p_0 in p_0_list:
    E_0 = np.sqrt(particle.m_p**2 * constants.c**4 + p_0** 2 * constants.c**2) - particle.m_p * constants.c**2
    simulation = Simulation(particle=particle,
                            material=material,
                            E_0=E_0,
                            x_max=x_max)
    simulation.solve()
    distance = simulation.distance
    distances.append(distance)

  plt.grid('on')
  plt.title(f"{particle.name}, {material.name}", fontsize=24)
  plt.xlabel("$p [GeV/c]$")
  plt.ylabel("$d_{max} [cm]$")

  plt.plot(np.array(p_0_list) / constants.q_e * constants.c / 1e9, 100 * np.array(distances))
  plt.show()


def plot_distance_vs_multiple_momentum(particle: Particle, materials, p_0_list,
                            x_max):

  distances = {}
  for material in materials:
    distances[material] = []

  for p_0 in p_0_list:
        E_0 = np.sqrt(particle.m_p**2 * constants.c**4 + p_0** 2 * constants.c**2) - particle.m_p * constants.c**2
        for material in materials:
              simulation = Simulation(particle=particle,
                                        material=material,
                                        E_0=E_0,
                                        x_max=x_max)
              simulation.solve()
              distance = simulation.distance
              distances[material].append(distance)

  plt.grid('on')
  plt.title(f"{particle.name}", fontsize=24)
  plt.xlabel("$p [GeV/c]$")
  plt.ylabel("$d_{max} [cm]$")
  for material in materials:
    plt.plot(np.array(p_0_list) / constants.q_e * constants.c / 1e9, 100 * np.array(distances[material]), label=material.name)
  plt.legend(loc="upper left")
  plt.show()

def plot_mass_distance_vs_multiple_momentum(particle: Particle, materials, p_0_list,
                            x_max):

  distances = {}
  for material in materials:
    distances[material] = []

  for p_0 in p_0_list:
        E_0 = np.sqrt(particle.m_p**2 * constants.c**4 + p_0** 2 * constants.c**2) - particle.m_p * constants.c**2
        for material in materials:
              simulation = Simulation(particle=particle,
                                        material=material,
                                        E_0=E_0,
                                        x_max=x_max)
              simulation.solve()
              distance = simulation.distance
              distances[material].append(distance)

  plt.grid('on')
  #plt.title(f"{particle.name}", fontsize=24)
  plt.xlabel("$p [GeV/c]$")
  plt.ylabel("$d_{max} [g/cm^{2}]$")
  for material in materials:
    plt.plot(np.array(p_0_list) / constants.q_e * constants.c / 1e9, 1e-1 * np.array(distances[material]) * material.rho , label=material.name)
  plt.legend(loc="upper left")
  plt.show()

#
# plot_distance_vs_energy(proton, polyethylene,
#                         [i * 1e8 * constants.q_e for i in range(1, 151)], 100)
# plot_distance_vs_momentum(proton, polyethylene,
#                         [i * 1e8 * constants.q_e / constants.c for i in range(1, 151)], 100)
# plot_distance_vs_momentum(proton, water,
#                         [i * 1e8 * constants.q_e / constants.c for i in range(1, 151)], 100)
# plot_distance_vs_momentum(proton, aluminum,
#                         [i * 1e8 * constants.q_e / constants.c for i in range(1, 151)], 100)
# plot_distance_vs_multiple_momentum(proton, [polyethylene, water, aluminum],
#                         [i * 0.2 * 1e8 * constants.q_e / constants.c for i in range(3, 80)], 100)
plot_distance_vs_multiple_momentum(positron, [polyethylene, water, aluminum],
                        [i * 0.2 * 1e8 * constants.q_e / constants.c for i in range(3, 80)], 100)
# plot_mass_distance_vs_multiple_momentum(proton, [polyethylene, water, aluminum],
#                         [i * 1e8 * constants.q_e / constants.c for i in range(1, 151)], 100)