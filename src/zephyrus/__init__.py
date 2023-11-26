from absl import flags

flags.DEFINE_bool("debug", default=False, short_name='d', help="If set will run in debug mode, some assertions won't be "
                                                          "performed etc")
