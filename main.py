import argparse

from commands import (
    example_message
)
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
    args = parser.parse_args()

    # Load settings from json
    set_config(args.config)

    # Execute command
    print('Executing command "{}".\n'.format(args.command))
    command = commands.get(
        args.command,
        error
    )
    command()

    # Finished
    input("\nPress [enter] to exit script...")
    print('Done.')


def error(_):
    raise Exception('This command is not implemented.')


# Run main
if __name__ == '__main__':
    main()
