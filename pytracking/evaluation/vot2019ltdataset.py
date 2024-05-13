import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class VOT2019LTDataset(BaseDataset):
    """
    VOT2019-LT dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = '/home/xyl/pysot-master/testing_dataset/VOT2019-LT'
        self.base_path2 = '/home/xyl/pysot-master/testing_dataset/VOT2019-LT_annotations'
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path2, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_name, frames, 'vot2019lt', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['ballet',
                        'bicycle',
                        'bike1',
                        'bird1',
                        'boat',
                        'bull',
                        'car1',
                        'car16',
                        'car3',
                        'car6',
                        'car8',
                        'car9',
                        'carchase',
                        'cat1',
                        'cat2',
                        'deer',
                        'dog',
                        'dragon',
                        'f1',
                        'following',
                        'freesbiedog',
                        'freestyle',
                        'group1',
                        'group2',
                        'group3',
                        'helicopter',
                        'horseride',
                        'kitesurfing',
                        'liverRun',
                        'longboard',
                        'nissan',
                        'parachute',
                        'person14',
                        'person17',
                        'person19',
                        'person2',
                        'person20',
                        'person4',
                        'person5',
                        'person7',
                        'rollerman',
                        'sitcom',
                        'skiing',
                        'sup',
                        'tightrope',
                        'uav1',
                        'volkswagen',
                        'warmup',
                        'wingsuit',
                        'yamaha']


        return sequence_list
