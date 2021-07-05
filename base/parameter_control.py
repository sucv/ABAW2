from operator import itemgetter

import numpy as np


class ParamControl(object):
    def __init__(self, trainer, gradual_release=1, release_count=8, backbone_mode="ir"):
        self.trainer = trainer
        self.gradual_release = gradual_release
        self.release_count = release_count
        self.backbone_mode = backbone_mode
        self.module_list = self.init_module_list()
        self.module_stack = self.init_param_group()
        self.early_stop = False

    def init_module_list(self):

        module_list = [[(4, 10)], [(163, 187)], [(142, 163)]]
        if self.backbone_mode == "ir_se":
            module_list = [[(4, 10)], [(205, 235)], [(187, 205)]]

        return module_list

    def init_param_group(self):
        # return {'0': slice(151, 160), '1': slice(160, 169), '2': slice(169, 178),
        #         '3': slice(178, 187), '4': slice(187, 196), '5': slice(196, 205),
        #         '6': slice(205, 235), '7': slice(4, 10)}
        module_stack = []
        for groups in self.module_list:
            slice_range = []
            if len(groups) > 1:
                for group in groups:
                    slice_range += list(np.arange(*group))
            else:
                slice_range = list(np.arange(*groups[0]))

            module_stack.append(slice_range)
        return module_stack

    def get_param_group(self):
        modules_to_release = self.module_stack.pop(0)
        return modules_to_release

    def get_current_lr(self):
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        return current_lr

    def release_param(self):
        if self.gradual_release:
            if self.release_count > 0:
                indices = self.get_param_group()

                for param in list(itemgetter(*indices)(list(self.trainer.model.parameters()))):
                    param.requires_grad = True

                self.trainer.init_optimizer_and_scheduler()
                self.release_count -= 1
            else:
                print("Early stopped since no further parameters to release!")
                self.early_stop = True

    def load_trainer(self, trainer):
        self.trainer = trainer