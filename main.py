import argparse
from config import set_config, get_config
import cProfile

from commands import (
    hooke_reeves_example,
    bisection_example,
    RMSProp_example,
    forward_diff,
    reverse_diff,
    metropolis_hasting_example,
    adam_example,
    simulated_annealing_example,
    rmse_adjusted
)
from ttictoc import TicToc
from config import set_config


# Main algorithm
def main():
    # print('Configuration:\n' + json.dumps(CONFIG, indent=4))

    # Possible commands
    command_selection = {
        'hooke_reeves_example':hooke_reeves_example,
        'bisection':bisection_example,
        'RMSProp':RMSProp_example,
        'diff':forward_diff,
        'diff2':reverse_diff,
        'metropolis_hasting_example':metropolis_hasting_example,
        'adam': adam_example,
        'annealing': simulated_annealing_example,
        'rmse': rmse_adjusted
    }

    # Select what to do based on input arguments
    parser = argparse.ArgumentParser(
        description='transaction exclusions.'
    )
    parser.add_argument(
        'command',
        choices=list(command_selection.keys()),
        help='command to execute'
    )
    parser.add_argument(
        '--config',
        nargs='?',
        default='config.json',
        help='path to configuration JSON'
    )
    parser.add_argument(
        '--profile',
        type=int,
        default=0,
        help='number of times to run the selected command while profiling'
    )
    args = parser.parse_args()

    # Load settings from json
    set_config(args.config)

    # Profile?
    if args.profile:
        print('\nStarting profiler.\n')
        profiler = cProfile.Profile(
            builtins=True
        )
        profiler.enable()

    # Execute command
    print('Executing command "{}".\n'.format(args.command))
    command = command_selection.get(
        args.command,
        error
    )
    for count in range(max(1, args.profile)):
        if args.profile > 1 and args.profile < 101:
            print(f'{1 + count} of {args.profile}')
        command()

    # Profile?
    if args.profile:
        profiler.disable()
        profiler.dump_stats(args.command + '.cprof')
        print('\nFinished profiling.')

    # Finished
    input("Press [enter] to exit script...")
    print('Done.')


def error(_):
    raise Exception('This command is not implemented.')


# Run main
if __name__ == '__main__':
    main()
