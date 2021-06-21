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
    C_CLUSTER = 6


def determine_env() -> Environment:
    cwd = os.getcwd()
    if "/home/gsoykan20/Desktop/AF-GAN/" in cwd:
        return Environment.G_LOCAL
    elif "/users/gsoykan20/" in cwd:
        return Environment.G_CLUSTER
    elif "/users/baristopal20/" in cwd:
        return Environment.B_CLUSTER
    elif "/users/ckoksal20/" in cwd: 
        return Environment.C_CLUSTER
    elif "/home/ckoksal20/Desktop/COMP547/SSuperGAN/" in cwd:
        return Environment.C_LOCAL
    else:
        raise NotImplementedError


def determine_base_dir(env: Environment) -> str:
    if env is Environment.G_CLUSTER:
        return "/kuacc/users/gsoykan20/projects/AF-GAN/"
    elif env is Environment.G_LOCAL:
        return "/home/gsoykan20/Desktop/AF-GAN/"
    elif env is Environment.B_CLUSTER:
        return "/kuacc/users/baristopal20/SSuperGAN/"
    elif env is Environment.C_CLUSTER:
        return "/kuacc/users/ckoksal20/COMP547Project/SSuperGAN/"
    elif env is Environment.C_LOCAL:
        return "/home/ckoksal20/Desktop/COMP547/SSuperGAN/"
    else:
        raise NotImplementedError


environment = determine_env()
base_dir = determine_base_dir(environment)

if __name__ == '__main__':
    print(base_dir)
