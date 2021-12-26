# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from __future__ import print_function
import os
import numpy as np
import skimage.io

class RandomNoise(object):
  """
  An image transform that adds noise to random pixels in the image.
  """

  def __init__(self,
               noiselevel=0.0,
               constValue=0.1307 + 2*0.3081,
               type=0,
               logDir=None, logProbability=0.01):
    """
    :param noiselevel:
      From 0 to 1. For each pixel, set its value to constValue with this
      probability. Can also be set to -1*constValue depending on type.
      Can also be uniformly distributed noise between +/- constValue.

    :param logDir:
      If set to a directory name, then will save a random sample of the images
      to this directory.

    :param logProbability:
      The percentage of samples to save to the log directory.

    """
    self.noiseLevel = noiselevel
    self.constVal = constValue
    self.iteration = 0
    self.logDir = logDir
    self.logProbability = logProbability
    self.type = type


  def __call__(self, image):
    self.iteration += 1
    a = image.view(-1)
    numNoiseBits = int(a.shape[0] * self.noiseLevel)
    noise = np.random.permutation(a.shape[0])[0:numNoiseBits]
    if type == 0:
      a[noise] = self.constValue
    elif type == 1:
      a[noise] = self.constValue
    elif type == 2:
      a[noise] = self.constValue
    elif type == 3:
      a[noise] = self.constValue
    elif:
      assert False, "illegal noise type"; exit()

    # Save a subset of the images for debugging
    if self.logDir is not None:
      if np.random.random() <= self.logProbability:
        outfile = os.path.join(self.logDir,
                               "im_noise_" + str(int(self.noiseLevel*100)) + "_"
                               + str(self.iteration).rjust(6,'0') + ".png")
        skimage.io.imsave(outfile,image.view(28,28))

    return image
