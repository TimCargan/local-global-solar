import os

from src.zephyrus.data_pipelines.parsers.MetOffice import MetOffice
import tensorflow as tf


def _ds_map(ds: tf.data.Dataset, f):
    return ds.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _ds_filter(ds: tf.data.Dataset, f):
    return ds.filter(f)


def _ds_inter(ds: tf.data.Dataset, f):
    return ds.interleave(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)


PIPELINE_STEPS = {
    "map": _ds_map,
    "filter": _ds_filter,
    "interleave": _ds_inter
}

class Data:

    @staticmethod
    def _run(parser, ds: tf.data.Dataset):
        run_book = parser.run
        for r in run_book:
            op, func = r["op"], r["fn"]
            ds = PIPELINE_STEPS[op](ds, func)
        return ds

    def load(self, paths, parser, compression_type="GZIP", shuffle=True, shuffle_buff=100, batch=True, batch_size=32, drop_remainder=False):
        at = tf.data.experimental.AUTOTUNE
        files = tf.data.Dataset.from_tensor_slices(paths)
        ds = tf.data.TFRecordDataset(files, buffer_size=int(500e6),
                                          compression_type=compression_type,
                                          num_parallel_reads=at)
        ds = self._run(parser, ds)
        ds = ds.prefetch(at)

        ds = ds.shuffle(buffer_size=shuffle_buff) if shuffle else ds
        ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder) if batch else ds
        return ds

    def save(self, paths, parser, name, data_folder="/mnt/d/data/", compression="GZIP"):
        dataset = self.load(paths, parser, shuffle=False, batch=False)
        save_path = os.path.join(data_folder, f"{name}.dataset")
        tf.data.experimental.save(dataset, save_path, compression=compression)
        return dataset


    def data_load(name, hp, data_folder="/mnt/d/data/", filter=None, cache=False):
        dataset = tf.data.experimental.load(f"{data_folder}/{name}.dataset", compression="GZIP")
        dataset = dataset.filter(filter) if filter else dataset
        # "out_ts": x["ts"][hp.get("OUTPUT_STEPS_start") + 24: hp.get("OUTPUT_STEPS_end") + 24]
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .map(
            lambda x, y: ({**x, "out_ts": x["ts"][hp.get("OUTPUT_STEPS_start") + 24: hp.get("OUTPUT_STEPS_end") + 24]},
                          y[hp.get("OUTPUT_STEPS_start") + 24: hp.get("OUTPUT_STEPS_end") + 24]),
            num_parallel_calls=8)

        dataset = dataset.cache() if cache else dataset
        return dataset

    @staticmethod
    def save_file(path, hp, name, train=True, shuffle=False):
        parser = MetOffice(hp, train)

        files = tf.data.Dataset.list_files(path, shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(files, buffer_size=int(100e6), compression_type="GZIP",
                                          num_parallel_reads=tf.data.experimental.AUTOTUNE) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .map(parser.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .filter(parser.validate)\
            .map(parser.parse_imges, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.interleave(parser.melt, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def shard(x):
            return tf.cast(x["shard"], tf.int64)

    @staticmethod
    def load_file(path, hp, comp="GZIP", drop_remainder=True, add_scale=False):
        parser = MetOffice(hp)
        at = tf.data.experimental.AUTOTUNE
        dataset = tf.data.experimental.load(f"{path}.dataset", compression=comp)
        dataset = dataset.interleave(parser.melt, num_parallel_calls=at)
        dataset = dataset.map(parser.shape, num_parallel_calls=at)
        dataset = dataset.batch(batch_size=hp.get("BATCH_SIZE"))

        return dataset

    @staticmethod
    def data_ex_read(path, hp, drop_remainder=True, shuffle_multiple=100, train=True, shuffle=True):
        parser = MetOffice(hp, train)
        at = tf.data.experimental.AUTOTUNE
        files = tf.data.Dataset.from_tensor_slices(path)
        # files = files.shuffle(buffer_size=hp.get("BATCH_SIZE") * 5)
        dataset = tf.data.TFRecordDataset(files, buffer_size=int(500e6),
                                          compression_type="GZIP",
                                          num_parallel_reads=at)
        dataset = dataset.map(parser.parse, num_parallel_calls=at)
        dataset = dataset.filter(parser.validate)
        dataset = dataset.map(parser.parse_imges, num_parallel_calls=at)
        dataset = dataset.interleave(parser.melt, num_parallel_calls=at)
        dataset = dataset.map(parser.shape, num_parallel_calls=at)
        dataset = dataset.prefetch(at)
        dataset = dataset.shuffle(buffer_size=hp.get("BATCH_SIZE")*5)
        dataset = dataset.batch(batch_size=hp.get("BATCH_SIZE"))

        return dataset


    @staticmethod
    def data_raw_read(path, hp, drop_remainder=True, shuffle_multiple=100, train=True, shuffle=True):
        parser = MetOffice(hp, train)

        files = tf.data.Dataset.list_files(path, shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(files, buffer_size=int(500e6),
                                          compression_type="GZIP",
                                          num_parallel_reads=tf.data.experimental.AUTOTUNE) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .map(parser.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .filter(parser.validate)\
            .interleave(parser.shape, block_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .batch(hp.get("BATCH_SIZE"), drop_remainder=drop_remainder)

    # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
    # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
    #
        return dataset
