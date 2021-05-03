import enum
import os


# NOTE: Barış and Caghan you can also introduce your own local env
# as B_LOCAL or C_LOCAL
class Environment(enum.Enum):
    G_LOCAL = 1
    B_LOCAL = 2
    C_LOCAL = 3
    G_CLUSTER = 4
    B_CLUSTER = 5
    C_CLUSTER = 5


def determine_env() -> Environment:
    cwd = os.getcwd()
    if "/home/gsoykan20/Desktop/AF-GAN/" in cwd:
        return Environment.G_LOCAL
    elif "/kuacc/users/gsoykan20/" in cwd:
        return Environment.G_CLUSTER
    else:
        raise NotImplementedError


def determine_base_dir(env: Environment) -> str:
    if env is Environment.G_CLUSTER:
        return "/kuacc/users/gsoykan20/projects/AF-GAN/"
    elif env is Environment.G_LOCAL:
        return "/home/gsoykan20/Desktop/AF-GAN/"
    else:
        raise NotImplementedError


environment = determine_env()
base_dir = determine_base_dir(environment)

if __name__ == '__main__':
    print(base_dir)