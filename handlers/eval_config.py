import yaml

from importlib import import_module


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        settings = yaml.safe_load(f)
    return settings


def eval_config(config):

    def _eval_config(config):
        if isinstance(config, dict):
            for key, value in config.items():
                if key not in ['module', 'class']:
                    config[key] = _eval_config(value)

            if 'module' in config and 'class' in config:
                module = config['module']
                class_ = config['class']
                config_kwargs = config.get(class_, {})
                return getattr(import_module(module), class_)(**config_kwargs)

            return config
        elif isinstance(config, list):
            return [_eval_config(ele) for ele in config]
        elif isinstance(config, str):
            return eval(config, __extralibs__)
        else:
            return config

    __extralibs__ = {name: import_module(lib) for (name, lib) in config.pop('extralibs', {}).items()}
    __extralibs__['config'] = config

    return _eval_config(config)
