# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CrowdPRW(ImageDataset):
    _junk_pids = [0, -1, -2]
    dataset_dir = ''
    dataset_name = "crowdprw"
    
    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Crowd-PRW-v16.04.20')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Crowd-PRW-v16.04.20".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)

        super(CrowdPRW, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, valid_id=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        sorted(img_paths)

        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for imgid, img_path in enumerate(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -2:
                continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, imgid))

        return data
