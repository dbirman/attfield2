import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import unittest

from proc import detection_task as det
import proc.network_manager as nm
from proc import video_gen
from proc import cornet

from pprint import pprint
import numpy as np
import skvideo.io


class TestSuite(unittest.TestCase):

    def setUp(self):
        self.model, ckpt = cornet.load_cornet("Z")
        self.task = det.IsolatedObjectDetectionTask(
            Paths.data('imagenet.h5'),
            whitelist = ['bathtub', 'artichoke'])
        self.SIZE = 20

    '''def test_loadfuncs(self):
        iso = det.IsolatedObjectDetectionTask(
            Paths.data('imagenet.h5'))
        imgs, all_ys = self.task.val_set(None, self.SIZE, cache = None)
        imgs, ys = self.task.train_set(None, self.SIZE, shuffle = True)

        four = det.FourWayObjectDetectionTask(
            Paths.data('imagenet.h5'))
        c = self.task.cats[0]
        imgs, ys, locs = four.val_set(c, self.SIZE, cache = None, loc = -1)
        imgs, ys, locs = four.train_set(c, self.SIZE, loc = 2, shuffle = True)

        # I don't know what thing should come out to for this
        self.task.val_size(self.SIZE)

    def test_fit_and_score(self):
        encodings, skregs, regmods, all_imgs, all_ys = det.fit_logregs(
            self.model, [(0, 4, 2)], self.task, train_size = self.SIZE,
            shuffle = False)
        c = self.task.cats[0]

        # Check output predictions match on and off model
        c_imgs, ys = self.task.train_set(c, self.SIZE, shuffle = False)
        mgr = nm.NetworkManager.assemble(self.model,
            c_imgs, mods = {(0,): regmods[c], (0, 4, 2): det.LayerBypass()})
        assembled_preds = mgr.computed[(0,)].detach().numpy()

        self.assertTrue(np.allclose(
            np.squeeze(regmods[c].predict_on_fn(assembled_preds)),
            regmods[c].predict(encodings[c])))
        self.assertTrue(np.allclose(
            regmods[c].predict(encodings[c]),
            skregs[c].predict(encodings[c])))

        det.score_logregs(regmods, encodings, all_ys)
        det.model_encodings(self.model, [(0, 4, 2)], all_imgs)
        det.model_encodings(self.model, [(0, 4, 2)], c_imgs)

    def test_misc(self):
        encodings, skregs, regmods, all_imgs, all_ys = det.fit_logregs(
            self.model, [(0, 4, 2)], self.task, train_size = self.SIZE,
            shuffle = False)
        det.save_logregs(Paths.data('unittest/test_logregs.npz'), regmods)
        det.load_logregs(Paths.data('unittest/test_logregs.npz'))
        det.save_logregs(Paths.data('unittest/test_logregs.npz'), skregs)
        det.load_logregs(Paths.data('unittest/test_logregs.npz'))

        decision1 = det.multi_decision(regmods, encodings)
        decision2 = det.by_cat(regmods,
            lambda c: regmods[c].decision_function(encodings[c]))
        self.assertTrue(all(
            np.allclose(decision1[c], decision2[c])
            for c in regmods))'''

    def test_quad_att(self):
        four = det.FourWayObjectDetectionTask(
            Paths.data('imagenet.h5'))
        c = four.cats[0]
        imgs, ys, locs = four.train_set(
            c, self.SIZE, shuffle = False)

        rand_att = det.QuadAttention(2., det.RAND_LOC)
        static_att = det.QuadAttention(2., 0)
        (imgs_rand_att,), _, _ = rand_att.pre_layer(imgs)
        (imgs_static_att,), _, _ = static_att.pre_layer(imgs)
        skvideo.io.vwrite(
            Paths.data('unittest/rand_att_{}.mp4'.format(c)),
            np.repeat(video_gen.to_numpy_force(imgs_rand_att), 20, axis = 0))
        skvideo.io.vwrite(
            Paths.data('unittest/static_att_{}.mp4'.format(c)),
            np.repeat(video_gen.to_numpy_force(imgs_static_att), 20, axis = 0))
        
        placed_att = det.QuadAttention(2., locs, profile = 'gauss')
        (imgs_placed_att,), _, _ = placed_att.pre_layer(imgs)
        skvideo.io.vwrite(
            Paths.data('unittest/placed_att_{}.mp4'.format(c)),
            np.repeat(video_gen.to_numpy_force(imgs_placed_att), 20, axis = 0))


if __name__ == '__main__':
    # Reached 99% coverage, with only a few cache operations missed
    # (30 May 2019)
    unittest.main()