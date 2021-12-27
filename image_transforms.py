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

class Noise(object):
  """
  An image transform that adds noise to random pixels in the image.
  """

  def __init__(self,
               noiselevel=0.0,
               stdev=0.308, # mnist
               type=0,
               logDir=None, logProbability=0.01):
    """
    :param noiselevel:
      From 0 to 1. For each pixel, set its value to stdev with this
      probability. Can also be set to -2*stdev depending on type.
      Can also be uniformly distributed noise between +/- 2 stdev.

    :param logDir:
      If set to a directory name, then will save a random sample of the images
      to this directory.

    :param logProbability:
      The percentage of samples to save to the log directory.

    """
    self.noiseLevel = noiselevel
    self.stdev = stdev
    self.iteration = 0
    self.logDir = "/tmp/"
    self.logProbability = logProbability
    self.type = type


  def __call__(self, image):
    self.iteration += 1
    a = image.view(-1)
    numNoiseBits = int(a.shape[0] * self.noiseLevel)
    noise = np.random.permutation(a.shape[0])[0:numNoiseBits]
  
    if self.type == 0:
      # noise is +2 stdev
      a[noise] = 2 * self.stdev
    elif self.type == 1:
      # noise is -2 stdev
      a[noise] = -1 * 2 * self.stdev
    elif self.type == 2:
      # gaussian noise with mean=0 variation=1
      a[noise] = np.random.randn(a.shape[0])[noise]
    elif self.type == 3:
      # uniform noise sampled over -/+ 2 stdev
      shifted = np.random.rand(a.shape[0]) - 0.5
      a[noise] = (4 * self.stdev * shifted)[noise]
    elif self.type == None:
      pass
    else:
      assert False, "illegal noise type: " + str(self.type)
      exit()

    # Save a subset of the images for debugging
    if self.logDir is not None:
      if np.random.random() <= self.logProbability:
        outfile = os.path.join(self.logDir,
                               "im_noise_" + str(int(self.noiseLevel*100)) + "_"
                               + str(self.iteration).rjust(6,'0') + ".png")
        skimage.io.imsave(outfile,image.view(28,28))

    return image
