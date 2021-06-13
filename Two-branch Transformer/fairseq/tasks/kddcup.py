from fairseq.data.lru_cache_dataset import LRUCacheDataset
from fairseq.data.append_token_dataset import AppendTokenDataset
from functools import lru_cache, reduce
import logging
import os
import numpy as np
from numpy.core.fromnumeric import sort
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumSamplesDataset,
    NumelDataset,
    data_utils,
    LeftPadDataset,
    BaseWrapperDataset,
    RawLabelDataset,
)
from fairseq.data.shorten_dataset import TruncateDataset, maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import II
from fairseq.data.indexed_dataset import (
    MMapIndexedDataset,
    get_available_dataset_impl,
    make_dataset,
    infer_dataset_impl,
)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from torch_geometric.data import Data, Batch
from fairseq.data.molecule.molecule import Tensor2Data
from fairseq.tasks.doublemodel import NoiseOrderedDataset, StripTokenDatasetSizes

logger = logging.getLogger(__name__)

gold_num = 0.0027211385049999842

@dataclass
class KDDCUPConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    num_classes: int = field(default=17144)
    scaler_label: bool = field(default=False)
    no_normalize: bool = field(default=False)
    no_shuffle: bool = field(default=False)
    shorten_method: ChoiceEnum(["none", "truncate", "random_crop"]) = field(default="truncate")
    shorten_data_split_list: str = field(default="")
    max_positions: int = II("model.max_positions")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    seed: int = II("common.seed")
    order_noise: int = field(default=5)
    data_type: str = II("model.datatype")
    data_mask: float = field(default=0.0)


@register_task("kddcup", dataclass=KDDCUPConfig)
class KddCup(FairseqTask):

    cfg: KDDCUPConfig

    def __init__(self, cfg: KDDCUPConfig, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self.label_dictionary = label_dictionary
        self._max_positions = cfg.max_positions
        self.seed = cfg.seed
        self.order_noise = cfg.order_noise
        self.no_normalize = cfg.no_normalize
        if self.cfg.scaler_label:
            self.prepare_scaler()
        else:
            self.label_scaler = None
        
        self.label_min = int(label_dictionary[label_dictionary.nspecial])
        self.label_max = int(label_dictionary[-1])

    def get_path(self, key, split):
        return os.path.join(self.cfg.data, key, split)

    def prepare_scaler(self):
        label_path = "{}.y".format(self.get_path("label_reg", "train"))
        assert os.path.exists(label_path)

        def parse_regression_target(i, line):
            values = line.split()
            return [float(x) for x in values]

        with open(label_path) as h:
            x = [parse_regression_target(i, line.strip()) for i, line in enumerate(h.readlines())]
        self.label_scaler = StandardScaler(x, self.no_normalize)

    def inverse_transform(self, x):
        if self.label_scaler is None:
            return x
        else:
            return self.label_scaler.inverse_transform(x)

    def transform_label(self, x):
        if self.label_scaler is None:
            return x
        else:
            return self.label_scaler.transform(x)

    def cls2reg(self, x_c):
        x_c = x_c + self.label_min
        x_c = x_c * gold_num
        return self.transform_label(x_c)

    def reg2cls(self, x_r):
        x_r = self.inverse_transform(x_r)
        x_r = (x_r + gold_num/2) // gold_num
        x_r = x_r.clamp(self.label_min, self.label_max)
        x_r = (x_r - self.label_min).long()
        return x_r

    @classmethod
    def setup_task(cls, cfg: KDDCUPConfig, **kwargs):
        assert cfg.num_classes > 0
        data_dict = cls.load_dictionary(os.path.join(cfg.data, "input0", "dict.txt"))
        logger.info(
            "[input] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "input0",), len(data_dict)
            )
        )
        label_dict = cls.load_dictionary(
            os.path.join(cfg.data, "label_cls_{}".format(cfg.num_classes), "dict.txt")
        )
        logger.info(
            "[label] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "label_cls_{}".format(cfg.num_classes)), len(label_dict)
            )
        )
        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split: str, **kwargs):
        prefix = self.get_path("input0", split)
        if not MMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("SMILES data {} not found.".format(prefix))
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("PyG data {} not found.".format(prefix))

        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_dataset(prefix, impl=dataset_impl)
        assert src_dataset is not None

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDatasetSizes(src_dataset, self.source_dictionary.eos()),
                self._max_positions - 1,
            ),
            self.source_dictionary.eos(),
        )

        src_dataset_graph = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset_graph is not None
        src_dataset_graph = Tensor2Data(src_dataset_graph)

        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset),
            },
            "net_input1": {"graph": src_dataset_graph,},
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }

        if split != "test":
            prefix = self.get_path("label_cls_{}".format(self.cfg.num_classes), split)
            cls_dataset = make_dataset(prefix, impl=dataset_impl)
            assert cls_dataset is not None
            dataset.update(
                target_cls=OffsetTokensDataset(
                    StripTokenDataset(cls_dataset, id_to_strip=self.label_dictionary.eos()),
                    offset=-self.label_dictionary.nspecial,
                )
            )

            reg_path = "{}.y".format(self.get_path("label_reg", split))
            assert os.path.exists(reg_path)

            def parse_regression_target(i, line):
                values = line.split()
                return [self.transform_label(float(x)) for x in values]

            with open(reg_path) as h:
                dataset.update(
                    target_reg=RawLabelDataset(
                        [
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ]
                    )
                )
                
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nested_dataset,
            sort_order=[shuffle, src_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )

        logger.info("Loaded {} with #samples: {}.".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.data_type == "tt":
            model.register_classification_head(
                "tc", num_classes=self.cfg.num_classes,
            )
            model.register_classification_head(
                "tr", num_classes=1,
            )
        elif self.cfg.data_type == "gg":
            model.register_classification_head(
                "tc", num_classes=self.cfg.num_classes, encoder="g"
            )
            model.register_classification_head(
                "tr", num_classes=1, encoder="g"
            )
        else:
            NotImplementedError()
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class StandardScaler:
    def __init__(self, x, no_normalize=False):
        if not no_normalize:
            x = np.array(x).astype(np.float)
            self.means = np.nanmean(x, axis=0)
            self.stds = np.nanstd(x, axis=0)
            self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
            self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
            self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
            self.means = float(self.means[0])
            self.stds = float(self.stds[0])
        else:
            self.means = 0 
            self.stds = 1

    def transform(self, x):
        return (x - self.means) / self.stds

    def inverse_transform(self, x):
        return x * self.stds + self.means
