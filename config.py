import json

CONFIG = {}

def set_config(file_path):
    with open(file_path, 'r') as config_file:
        CONFIG.clear()
        CONFIG.update(json.load(config_file))


def get_config(*keys, config=None, **kwargs):
    if config is None:
        config = CONFIG
    give_default = False
    if 'default' in kwargs:
        give_default = True
        default = kwargs.pop('default')
    if kwargs:
        raise ValueError('Unrecognized keyword arguments:', list(kwargs.keys()))
    for key in keys:
        if not key in config:
            if give_default:
                return default
            raise Exception(
                f'The expected key "{key}" cannot be found ' +
                'in the configuration json (can be a nested key).'
            )
        if config[key] is None:
            if give_default:
                return default
            raise Exception(
                f'The key "{key}" is empty in the ' +
                'configuration json (can be a nested key).'
            )
        config = config[key]
    return config
