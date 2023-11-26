import os.path
import platform

#TODO: move this to a config file
# host: {pp: hp}
_paths = {
    "GPU_CLUSTER" : {
        "data":  "/db/psytc3/data",
        "results": "/home/psytc3/results",
        "mlflow": "/home/psytc3/mlflow",
        "cache": "/db/psytc3/cache",
        "dl": "/home/psytc3/dl"
    },
    "nimloth": {
        "data":  "/media/tim/Data/riastore",
        "results": "/media/tim/Data/results",
        "cache": "/media/tim/Data/cache",
        "dl":  "/media/tim/Data/dl",
        "mlflow":  "/media/tim/Data/dl"
    },
    "grond": {
        "data":  "/home/psytc3/data",
        "results": "/home/psytc3/results",
        "cache": "/home/psytc3/cache",
    },
    "zeus": {
        "data": "/Users/Tim/ML/data",
        "results": "/Users/Tim/ML/results",
        "dl": "/Users/Tim/ML/dl",
        "mlflow": "/Users/Tim/ML/mlflow",
        "cache": "/Users/Tim/ML/cache",
    },
    "atlas": {
        "data": "/mnt/c/Users/Tim/Data",
        "results": "/mnt/d/exper-results",
        "mlflow": "/mnt/d/mlflow",
        "dl":  "D:\data\dl",
        "cache": " D:/cache",
    },
    }


def get_path(folder: str) -> str:
    """
    Gets the local path for the project folder
    if none exists an out of bounds error is thrown
    TODO: make this better
    :param folder:
    :return:
    """
    host = platform.node().lower()
    # Cluser check
    if "cs.nott.ac.uk" in host:
        host = "GPU_CLUSTER"
    if "zeus" in host:
        host = "zeus"
    # path_dict = [x for x in _paths if x["host"].lower() == host and x["project_path"] == folder]
    if host in _paths:
        return _paths[host][folder]

    return os.path.join(".", folder)
