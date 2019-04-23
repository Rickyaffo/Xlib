"""Constants declared and assigned which are imported to main file.

.. moduleauthor:: Riccardo Affolter <riccardo.affolter@gmail.com>
"""
import os
class Constants(object):
    PATH = os.path.abspath(os.curdir)
    BLACKBOX = ["RandomForest","Neural_Network", "SVM","CNN"]
    TYPE = ["TAB", "IMG", "TXT"]
    EXPLAINERTABGLOBAL = ["Skater"]
    EXPLAINERTABLOCAL = ["Lime", "Lore", "Anchor", "Ext"]
    EXPLAINERIMG = ["Lime","DeepExplainer","Saliency"]
    EXPLAINERTXT = []

    def __setattr__(self, attr, value):
        if hasattr(self, attr):
            raise ValueError('Attribute %s already has a value and so cannot be written to' % attr)
        self.__dict__[attr] = value

    def __getattr__(self, item):
        if not hasattr(self, item):
            raise ValueError("The value %s doesn't exist" % item)
        return self.__dict__[item]

