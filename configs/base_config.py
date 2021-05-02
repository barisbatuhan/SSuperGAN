import enum


# NOTE: Barış and Caghan you can also introduce your own local env
# as B_LOCAL or C_LOCAL
class Environment(enum.Enum):
    G_LOCAL = 1
    CLUSTER = 2
    B_LOCAL = 3
    C_LOCAL = 4


def determine_base_dir(env: Environment) -> str:
    if env is Environment.CLUSTER:
        return "/kuacc/users/gsoykan20/projects/AF-GAN/"
    elif env is Environment.G_LOCAL:
        return "/home/gsoykan20/Desktop/AF-GAN/"
    else:
        raise NotImplementedError


environment = Environment.G_LOCAL
base_dir = determine_base_dir(environment)
