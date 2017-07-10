import fs._fs as c


def init(omega_m):
    """Initialise cosmology module

    Args:
        init(omega_m)
    """

    c._cosmology_init(omega_m)

def D_growth(a):
    """
    Linear growth factor
    Arg:
        a (float): scale factor

    Returns:
        D(a) (float)
    """
    return c._cosmology_D_growth(a)

def D2_growth(a):
    """
    2nd-order growth factor D2
    2LPT displacement is D*dx1 + D2*dx2

    Arg:
        a (float): scale factor

    Returns:
        D2(a) (float)
    """
    return c._cosmology_D2_growth(a)


