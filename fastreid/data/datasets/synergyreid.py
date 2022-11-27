import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SynergyReID(ImageDataset):

    dataset_dir = ''
    dataset_url = ''
    dataset_name = "synergyreid_data"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, self.dataset_name)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as  under '
                          '"data_reid".')

        self.train_dir = osp.join(self.data_dir, 'reid_training')
        self.query_dir = osp.join(self.data_dir, 'reid_test/query')
        self.gallery_dir = osp.join(self.data_dir, 'reid_test/gallery')
        self.challenge_query_dir = osp.join(self.data_dir, 'reid_challenge/query')
        self.challenge_gallery_dir = osp.join(self.data_dir, 'reid_challenge/gallery')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.challenge_query_dir,
            self.challenge_gallery_dir
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
  
        if "use_challenge" in kwargs and kwargs["use_challenge"]:
            query = lambda: self.process_dir(self.challenge_query_dir, is_train=False)
            gallery = lambda: self.process_dir(self.challenge_gallery_dir, is_train=False)    
        else:
            query = lambda: self.process_dir(self.query_dir, is_train=True)
            gallery = lambda: self.process_dir(self.gallery_dir, is_train=True)

        super(SynergyReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpeg'))
        
        data = []
        for img_path in img_paths:

            camid = 1 if 'gallery' in dir_path else 0
            pid = 0
            # assert 0 <= pid <= 486  # pid == 0 means background
            # assert 0 <= camid <= 1

            if is_train:
                pattern = re.compile(r'([-\d]+)_(\d)')
                pid, sequnenceid = map(int, pattern.search(img_path).groups())
                # pid = self.dataset_name + "_" + str(pid)
                # camid = self.dataset_name + "_" + str(camid)
            else:
                pattern = re.compile(r'([-\d]+)')
                pid = int(pattern.search(img_path).group())

            data.append((img_path, pid, camid))

        return data
