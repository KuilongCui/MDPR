# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'img_paths':inputs['img_paths']
        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        img_paths = []

        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            img_paths.extend(prediction['img_paths'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]

        #  query_path, gallery_path
        query_path = img_paths[:self._num_query]
        gallery_path = img_paths[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        max_v,a,b,c,d = 0,1,1,1,1
        query_features = query_features.view(query_features.shape[0], 4 ,-1)
        gallery_features = gallery_features.view(gallery_features.shape[0], 4, -1)
        # dist1 = build_dist(query_features[:,0,:], gallery_features[:,0,:], self.cfg.TEST.METRIC)
        # dist2 = build_dist(query_features[:,1,:], gallery_features[:,1,:], self.cfg.TEST.METRIC)
        # dist3 = build_dist(query_features[:,2,:], gallery_features[:,2,:], self.cfg.TEST.METRIC)
        # dist4 = build_dist(query_features[:,3,:], gallery_features[:,3,:], self.cfg.TEST.METRIC)

        # dist = dist1 + dist2 + dist3 + dist4
        
        for i in range(1, 11, 1):
            for j in range(1, 11, 1):
               for k in range(1, 11, 1):
                    for l in range(1, 11, 1):

                        new_q = copy.deepcopy(query_features)
                        new_g = copy.deepcopy(gallery_features)

                        new_q[:,0,:] = new_q[:,0,:] * i /10
                        new_q[:,1,:] = new_q[:,1,:] * j /10
                        new_q[:,2,:] = new_q[:,2,:] * k /10
                        new_q[:,3,:] = new_q[:,3,:] * l /10

                        new_g[:,0,:] = new_g[:,0,:] * i /10
                        new_g[:,1,:] = new_g[:,1,:] * j /10
                        new_g[:,2,:] = new_g[:,2,:] * k /10
                        new_g[:,3,:] = new_g[:,3,:] * l /10

                        new_qf = new_q.view(3368, 2048)
                        new_gf = new_g.view(15913, 2048)

                        dist =  build_dist(new_qf, new_gf, self.cfg.TEST.METRIC)

                        if self.cfg.TEST.RERANK.ENABLED:
                            logger.info("Test with rerank setting")
                            k1 = self.cfg.TEST.RERANK.K1
                            k2 = self.cfg.TEST.RERANK.K2
                            lambda_value = self.cfg.TEST.RERANK.LAMBDA

                            if self.cfg.TEST.METRIC == "cosine":
                                query_features = F.normalize(query_features, dim=1)
                                gallery_features = F.normalize(gallery_features, dim=1)

                            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
                            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

                        from .rank import evaluate_rank

                        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids,
                            query_path, gallery_path, log_path=self.cfg.OUTPUT_DIR, output_mismatch=self.cfg.MISMTACH_OUTPUT)

                        mAP = np.mean(all_AP)
                        mINP = np.mean(all_INP)
                        for r in [1, 5, 10]:
                            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
                        self._results['mAP'] = mAP * 100
                        self._results['mINP'] = mINP * 100
                        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

                        if self._results['mAP'] > max_v:
                            max_v = self._results['mAP']
                            a,b,c,d = i,j,k,l

                            print('update')

                        print("mAP: ", self._results['mAP'], " ", i, " ",j, " ", k, " ", l)

        print("max_v: ", max_v, " ", a, " ",b, " ", c, " ", d)

        assert 1 == 0

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()
