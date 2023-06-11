def basic_parse_argument(parse=True):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',action='store',default='config')
    if parse:
        return parser.parse_args()
    else:
        return parser

def import_configuration(path):
    from importlib import import_module
    config = import_module(path)
    return config
