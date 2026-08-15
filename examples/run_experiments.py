import argparse
from aftab import Aftab


def main():
    parser = parser_factory()
    args = parser.parse_args()
    agent = Aftab(experiment_name=args.name, verbose=args.verbose)
    agent.train(environment=args.environment, seed=args.seed)
    agent.log(directory="results")


def parser_factory():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        type=int,
        required=True,
        default=True,
        help="Verbose makes the agent to flush out periodical output as feedback.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed controls the reproducibility of the experiments. It seeds Numpy, PyTorch, Python Random module, and Scikit Learn libraey with the same seed.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        help="EnvPool environment name, this can be any environment listed on their website under Atari benchmark or Procgen.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Required experiment name that will eventually be used to store results on the disk drive.",
    )
    return parser


if __name__ == "__main__":
    main()
