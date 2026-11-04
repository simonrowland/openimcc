"""Shared thermodynamic constants and molar masses."""

from types import MappingProxyType


# CODATA 2018: exact N_A times exact k_B, in J/(mol K).
R_J_MOL_K = 8.31446261815324

# Published gas tables and oxygen-balance digests use R rounded to 9 places.
# Retain that contract value where those shipped outputs are evaluated or built.
R_PUBLISHED_GAS_J_MOL_K = round(R_J_MOL_K, 9)

# Oxide rows keep the legacy project oxide table, which is built on O = 15.9994
# (e.g. Al2O3 = 2*26.9815385 + 3*15.9994 = 101.9613 to the stored digits).
# P2O5 is the unrounded sum on that same convention:
# 2*30.973761998 + 5*15.9994 = 141.944523996 g/mol (formerly rounded to
# 141.9445; relative change +1.7e-7, visible only in P2O5 weight-to-mole
# conversion). Element O stays 15.999 because the oxygen-balance formula masses
# are summed from the element rows and must not move. Sulfur uses the current
# CIAAW abridged value 32.06; 32.065 was a former recommendation.
MOLAR_MASS_G_MOL = MappingProxyType(
    {
        "SiO2": 60.0843,
        "MgO": 40.3044,
        "FeO": 71.844,
        "CaO": 56.0774,
        "Al2O3": 101.9613,
        "TiO2": 79.866,
        "Na2O": 61.9789,
        "K2O": 94.196,
        "S": 32.06,
        "P2O5": 141.944523996,
        "MnO": 70.9374,
        "Fe2O3": 159.6882,
        "Cr2O3": 151.9904,
        "NiO": 74.6928,
        "CoO": 74.9326,
        "O": 15.999,
        "Na": 22.989769,
        "K": 39.0983,
        "Si": 28.085,
        "Fe": 55.845,
        "Mg": 24.305,
        "Al": 26.9815385,
        "Ca": 40.078,
        "Ti": 47.867,
        "Cr": 51.9961,
        "V": 50.9415,
        "Nb": 92.90637,
        "Ta": 180.94788,
        "Be": 9.0121831,
        "Tl": 204.38,
        "Mn": 54.938044,
        "Ni": 58.6934,
        "Co": 58.933194,
        "Sb": 121.760,
        "Ag": 107.8682,
        "P": 30.973761998,
        "Li": 6.94,
        "Rb": 85.4678,
        "Pb": 207.2,
        "B": 10.81,
        "Ga": 69.723,
        "Ge": 72.630,
        "In": 114.818,
        "Cs": 132.90545196,
        "Cu": 63.546,
        "Sn": 118.710,
        "Zn": 65.38,
        "Cd": 112.411,
        "Bi": 208.9804,
        "As": 74.921595,
        "Se": 78.971,
        "Te": 127.60,
        "Mo": 95.95,
        "W": 183.84,
        "Ba": 137.327,
        "Sr": 87.62,
        "Zr": 91.224,
        "Sc": 44.955908,
        "Y": 88.90584,
        "U": 238.02891,
        "Th": 232.0377,
        "Hf": 178.49,
        "La": 138.90547,
        "Ce": 140.116,
        "Pr": 140.90766,
        "Nd": 144.242,
        "Sm": 150.36,
        "Eu": 151.964,
        "Gd": 157.25,
        "Tb": 158.92535,
        "Dy": 162.500,
        "Ho": 164.93033,
        "Er": 167.259,
        "Tm": 168.93422,
        "Yb": 173.045,
        "Lu": 174.9668,
        "Br": 79.904,
        "I": 126.90447,
        "Cl": 35.45,
        "F": 18.998403163,
        "Hg": 200.592,
        "Au": 196.96657,
        "Pd": 106.42,
        "Pt": 195.084,
        "Ru": 101.07,
        "Rh": 102.9055,
        "Ir": 192.217,
        "Re": 186.207,
        "Os": 190.23,
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
    }
)

# Atom-only view for gas-channel consumers; values still come from the table.
ATOMIC_MASS_G_MOL = MappingProxyType(
    {
        name: mass
        for name, mass in MOLAR_MASS_G_MOL.items()
        if name.isalpha() and len(name) <= 2
    }
)
