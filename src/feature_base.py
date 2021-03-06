import inspect
# import feather
import pandas as pd
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from utils import timer


def get_features(namespace):  # 特徴量クラスのみを抽出する関数
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            yield v()


def generate_features(namespace):
    for f in get_features(namespace):
        # if f.df_path.exists() and not overwrite:
        if f.train_path.exists() and f.test_path.exists():
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                "([A-Z])",
                lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip('_')

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        # self.df = pd.DataFrame()
        self.train_path = Path(self.dir) / '{}_train.feather'.format(self.name)
        self.test_path = Path(self.dir) / '{}_test.feather'.format(self.name)
        # self.df_path = Path(self.dir) / f'{self.name}_df.feather'

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
            # self.df.columns = prefix + self.df.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))
        # self.df.to_feather(str(self.df_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))
        # self.df = feather.read_dataframe(str(self.df_path))
