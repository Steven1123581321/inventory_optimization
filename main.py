import argparse

from commands import (
    example_message
)
from ttictoc import TicToc
from config import set_config


# Main algorithm
def main():
    # print('Configuration:\n' + json.dumps(CONFIG, indent=4))

    # Possible commands
    commands = dict()
    commands['example_command'] = example_message

    # Select what to do based on input arguments
    parser = argparse.ArgumentParser(
        description=
        'A module with an example package.'
    )
    parser.add_argument(
        'command',
        choices=list(commands.keys()),
        help='command to execute'
    )
    parser.add_argument(
        '--config',
        nargs='?',
        default='config.json',
        help='path to configuration JSON'
    )
    parser.add_argument(
        '--time',
        nargs='?',
        type=int,
        default=0,
        help='time running the selected command [TIME] times'
    )
    args = parser.parse_args()

    # Load settings from json
    set_config(args.config)

    # Time?
    if args.time:
        t = TicToc()
        t.tic()
        print('\nStarting timer.')

    # Execute command
    print('Executing command "{}".'.format(args.command))
    command = commands.get(
        args.command,
        error
    )
    for count in range(max(1, args.time)):
        if args.time > 1:
            print(f'{1 + count} of {args.time}', end='\r')
        command()

    # Time?
    if args.time:
        t.toc()
        print(
            f'Total time elasped: {t.elapsed:8.3f} seconds' +
            f'\nAverage time: {t.elapsed / args.time:8.3f}.'
        )

    # Finished
    input("\nPress [enter] to exit script...")
    print('Done.')


def error(_):
    raise Exception('This command is not implemented.')


# Run main
if __name__ == '__main__':
    main()
