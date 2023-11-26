import keras_tuner as kt
from zephyrus.utils.standard_logger import build_logger


class HyperParameters_Extend (kt.HyperParameters):
    """
    This is a custom extension to keras tuner HyperParameter class
    It adds the ability to add a "Config" type that can be changed
    This was motivated as the Fixed type cannot be updated once set

    This class should work as a drop in replacement for the existing Hyperparm with any hyperparamters being selected over
    config values
    """
    def __init__(self, verbose=True):
        super(HyperParameters_Extend, self).__init__()
        self.configs = {}
        self.verbose = verbose
        self.logger = build_logger(__name__)

    def __dict__(self):
        return dict(**self.configs, **self.get_config())

    def get(self, name):
        try:
            return super(HyperParameters_Extend, self).get(name)
        except KeyError:
            if name in self.configs:
                return self.configs[name]
            else:
                raise KeyError("{} does not exist.".format(name))


    def Config(self, name, value):
        """
        Add a config value to the HyperParameter object
        """
        if self.verbose and name in self.configs:
            self.logger.warn(f"Updating existing config value '{name}' from '{self.configs[name]}' to '{value}'")
        self.configs[name] = value
        return value

    #########################################
    #########        HP utils      #########
    #########################################

    def combination(hp, name, values, parent_name=None, parent_values=None):
        res = []
        for i, v in enumerate(values):
            if hp.Boolean(f"{name}_c_{i}", default=True, parent_name=parent_name, parent_values=parent_values):
                res.append(v)
        return res

    def array_set(hp, name, arr):
        hp.Fixed(f"_{name}_len", len(arr))
        for i, el in enumerate(arr):
            hp.Fixed(f"_{name}_el_{i}", el)

    def array_get(hp, name):
        num_els = hp.get(f"_{name}_len")
        arr = []
        for i in range(num_els):
            arr.append(hp.get(f"_{name}_el_{i}"))
        return arr