import functools
import inspect


class FromDict:
    @classmethod
    def from_dict(cls, env):
        mro = inspect.getmro(cls)
        params = functools.reduce(lambda ps, c: ps | set(inspect.signature(c).parameters.keys()), list(mro), set())
        return cls(**{
            k: v for k, v in env.items()
            if k in params
        })



def set_gpu_growth(opt=True):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, opt)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def split_gpus(num_vgpus=1, size_mb=4096):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=size_mb) for _ in range(num_vgpus)])

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Invalid device or cannot modify logical devices once initialized.
            print(e)


def patch_threading_excepthook():
    """Installs our exception handler into the threading modules Thread object
    Inspired by https://bugs.python.org/issue1230540
    """
    import threading
    import sys
    old_init = threading.Thread.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        old_run = self.run

        def run_with_our_excepthook(*args, **kwargs):
            try:
                old_run(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print(f"Uncaught error in thread: {self.name}", file=sys.stderr)
                sys.excepthook(*sys.exc_info())

        self.run = run_with_our_excepthook

    threading.Thread.__init__ = new_init
    print(f"WARN (COMMON): Running thead excpet patch, all uncaght errors in threads will be thrown")
