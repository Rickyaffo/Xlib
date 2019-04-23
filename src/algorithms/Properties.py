import cv2
from matplotlib import pylab as P, pyplot
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage import feature, transform
from utility.Utility import ShowImage, ShowGrayScaleImage


class Properties:
    def __getattr__(self, attr):
    #    print("Value {} not implemented or not available for this explanation.".format(attr))
        return None

    def displayTab(self):
        self.__printRules()
        self.__printTime()
        self.__printLenRules()
        self.__printFidelity()

    def displayImg(self):
        self.__printTime()
        self.__printImagesRules()

    def __printFidelity(self):
        try:
            self.Fidelity
            print('Fidelity: %.2f' % (self.Fidelity))
        except TypeError:
            return None

    def __printCRules(self):
        for crule in self.CRules:
            print(crule)

    def __printLenRules(self):
        try:
            self.Length
            print('Length rules: {}'.format(self.Length))
        except TypeError:
            return None

    def __printRules(self):
        try:
            self.Rules
            print('r = %s' % (self.Rules))
        except TypeError:
            return None

    def __printCounterFactuals(self):
        try:
            self.CounterFactuals
            print('c = %s' % (self.CounterFactuals))
        except TypeError:
            return None

    def __printTime(self):
        try:
            millis = int(round(self.Time * 1000))
            print('Time spent = %s ms' % str(millis))
        except TypeError:
            return None

    def __ShowExplanationRules(self,imgs2show, masks, title,exp):
        ROWS = 1
        COLS = len(title)
        UPSCALE_FACTOR = 15
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        i=1
        for img2show, mask, title in zip(imgs2show, masks, title):
            if(title == "Original"):
                if (isinstance(self.Xi , np.ndarray)):
                    self.__plot(img2show, cmap='Greys',axis=P.subplot(ROWS, COLS, i)).set_title(title)
                else:
                    ShowImage(img2show, title=title,ax=P.subplot(ROWS, COLS, i))
                    xi = (img2show - np.min(img2show))
                    xi /= np.max(xi)
            else:
                if(exp == "Lime"):
                    ShowImage(mark_boundaries(img2show / 2 + 0.5, mask), title, ax=P.subplot(ROWS, COLS, i))
                elif(exp == "Saliency"):
                    ShowGrayScaleImage(img2show, title=title,
                                       ax=P.subplot(ROWS, COLS, i))
                elif (exp == "DeepExplainer"):
                    if(isinstance(self.Xi , np.ndarray)):
                        self.__plot(img2show.reshape( self.SHAPE1,  self.SHAPE2), xi=self.Xi , axis=P.subplot(ROWS, COLS, i)).set_title(title)
                    else:
                        self.__plot(img2show, xi=xi, axis=P.subplot(ROWS, COLS, i), dilation=.5, percentile=99, alpha=.2).set_title(title)
                elif(exp == "Lore"):
                    if(title == "Hide"):
                        ShowImage(img2show, title, ax=P.subplot(ROWS, COLS, i))
                    else:
                        ShowImage(mark_boundaries(cv2.cvtColor(img2show, cv2.CV_32S), mask), title, ax=P.subplot(ROWS, COLS, i))
            i +=1

    def __printImagesRules(self):
        imgs2show = self.rulesImages[0][0]
        masks = self.rulesImages[0][1]
        title = self.rulesImages[0][2]
        self.__ShowExplanationRules(imgs2show, masks, title,self.exp)



    def __plot(self,data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
        dx, dy = 0.05, 0.05
        xx = np.arange(0.0, data.shape[1], dx)
        yy = np.arange(0.0, data.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_xi = plt.get_cmap('Greys_r')
        cmap_xi.set_bad(alpha=0)
        overlay = None
        if xi is not None:
            # Compute edges (to overlay to heatmaps later)
            xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
            in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
            edges = feature.canny(in_image_upscaled).astype(float)
            edges[edges < 0.5] = np.nan
            edges[:5, :] = np.nan
            edges[-5:, :] = np.nan
            edges[:, :5] = np.nan
            edges[:, -5:] = np.nan
            overlay = edges

        abs_max = np.percentile(np.abs(data), percentile)
        abs_min = abs_max

        if len(data.shape) == 3:
            data = np.mean(data, 2)
        axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        if overlay is not None:
            axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
        axis.axis('off')
        return axis
