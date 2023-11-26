from absl import flags

flags.DEFINE_integer("batch_size", default=128, help="Batch size to train for (eval is sometimes 8*bs)")
flags.DEFINE_integer("warmup_steps", default=12, help="Number of warmup steps, i.e past irrad steps")
flags.DEFINE_integer("output_steps", default=6, help="Runs from `WARMUP_STEPS` e.g [ws+0, ... ws + output_steps]")
flags.DEFINE_boolean("auto_restore", default=False, help="Restore checkpoints if found")
flags.DEFINE_boolean("pad_psb", default=False, help="Pad PSB with zeros to a fixed batch size")
flags.DEFINE_integer("num_epochs", default=20, help="The number of epochs to train for")
flags.DEFINE_integer("profile_step", default=None, help="Runs from `WARMUP_STEPS` e.g [ws+0, ... ws + output_steps]")
flags.DEFINE_integer("profile_step_count", default=5, help="Runs from `WARMUP_STEPS` e.g [ws+0, ... ws + output_steps]")

