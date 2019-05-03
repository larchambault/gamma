'''
Created on 2018-12-06

@author: Louis Archambault
'''

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

import skfmm # [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm)

class gamma:
   """ Gamma calculation (global) based on Chen et al. 2009

       WARNING (for now): assume same sampling between ref and meas
       Compute one distance map then apply to multiple measurements
       see reading notes and jupyter_notebook/gamma
   """


   def __init__(self,dose_crit,dist_crit,vox_size,threshold = None,ref=None):
      """ initialization

          ref: reference dose distribution a 3D numpy array
          vox_size: voxel size [x,y,z] in mm
            - Note that image shape is infered from vox_size
          dose_crit: \Delta [percent of max]
          dist_crit: \delta [mm]
          threshold: minimum dose below which gamma value is not computed [percent of max]
      """

      self.dist_crit = dist_crit
      self.dose_crit = dose_crit
      self.min_threshold = threshold
      self.set_voxel_size(vox_size)
      self.ndim = len(self.vox_size)
      print(f'Image dimensions: {self.ndim}')
      self.max_dose_pct = 120 # [%], fraction of maximum to use as the upper dose bin

      # empty initialization, waiting for ref image
      self.ndbins = None # number of dose bins
      self.dbins = []
      self.ref_img = None
      self.dist_map = None
      self.delta = None # voxel sizes as fx of criterion

      if ref is not None:
         self.set_reference(ref)

   def set_voxel_size(self,vox_size):
      self.vox_size = np.array(vox_size) # calling np.array does nothing
                                         # if vox_size is already a ndarray

   def set_reference(self,ref):

      # dx -> x,y,z distance as fx of distance criterion
      dx = self.vox_size/self.dist_crit # voxel dimensions expressed as fx of criterion
      dd = np.mean(dx) # dimension along dose axis should be the same as space axis.
                       # Dose bins are set afterward
      print(dx,dd)

      # absolute dose criterion = % of max
      max_dose =  np.max(ref)
      abs_dose_crit = max_dose*self.dose_crit/100.

      # absolute threshold criterion
      if self.min_threshold:
         self.abs_min_threshold = self.min_threshold/100 * max_dose
      else:
         self.abs_min_threshold = None

      # Set the number of dose bins to match the voxel dose length (i.e. dd)
      # dd = dose_vox_size/abs_dose_crit
      # dose_vox_size = max_dose/ndbins
      # max_dose = (maximum of ref)*max_dose_pct
      # -> ndbins = max_dose/(dd * abs_dose_crit)
      ndbins = int(np.ceil(max_dose*self.max_dose_pct/100/dd/abs_dose_crit))
      dbins = np.linspace(0,max_dose*self.max_dose_pct/100,ndbins+1)

      # assign important data to self
      self.ref_img = ref
      self.delta = np.append(dx,dd) # voxel sizes as fx of criterion
      self.ndbins = ndbins
      self.dbins = dbins


      self.compute_dist_map()

   def compute_dist_map(self):
      """ Compute distance map (can be time consuming) """
      # Build an hyper surface with three spatial dimensions + 1 dose dimension
      hypSurfDim = self.ref_img.shape + (self.ndbins,)
      hypSurf = np.ones( hypSurfDim )

      # Fill each layer of the dose axis
      # Dose points are set to 0

      lookup = np.digitize(self.ref_img,self.dbins) - 1 # lookup contains the index of dose bins

      print(f'Dose bins = {self.ndbins}')

      for i in range(self.ndbins):
         dose_points = lookup == i
         if self.ndim == 3:
            hypSurf[:,:,:,i][dose_points] = 0
            # simple (naive) interpolation. See Fig. 2 au Chen 2009
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,0)
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,1)
            hypSurf = self._interp_dose_along_ax3(hypSurf,lookup,2)
         elif self.ndim == 2:
            hypSurf[:,:,i][dose_points] = 0
            # simple (naive) interpolation. See Fig. 2 au Chen 2009
            hypSurf = self._interp_dose_along_ax2(hypSurf,lookup,0)
            hypSurf = self._interp_dose_along_ax2(hypSurf,lookup,1)
         else:
            raise IndexError('Only 2 and 3 spatial dimension supported at this moment')

      # print(self.delta)
      dst = edt(hypSurf,sampling=self.delta)

      self.dist_map = dst

   def _interp_dose_along_ax3(self,hypsrf,lookup,ax):
      """ interpolate the dose axis along a given spatial axis (3D)

          lookup: a matrix of the shape of spatial dimensions where each elements
                  corresponds to the dose bin at that point (produced by np.digitize)
          hypsrfs: the hypersurface: [x,y,z,dose_bin]
          ax: the index along which the interpolation is done (x=0,y=1,z=2)
      """

      dims = self.ref_img.shape # the spatial dimensions triplet (x, y, z)
      hs = np.copy(hypsrf)

      ax_slc = slice(None) # take the whole ax (same as [:])
      ax_order = [ax] + [x for x in [0,1,2] if x != ax] # [ax, oax1, oax2]
      ax_sort = np.argsort(ax_order)

      # a function that take (ax, oax1, oax2) and return (x,y,z)
      xyz = lambda x,y,z : ([x,y,z][ax_sort[0]], [x,y,z][ax_sort[1]], [x,y,z][ax_sort[2]])

      for oax1 in range(dims[ax_order[1]]):
         for oax2 in range(dims[ax_order[2]]):
            v = lookup[xyz(ax_slc,oax1,oax2)]
            jumps = np.diff(v) # number of pixels in gap
            idx_jump = np.argwhere(np.abs(jumps) > 1).flatten() # next dose jumps by more than 1 bin

            for i in idx_jump:
               # starting point to pad on hyper surface: (i,v[i])
               pad = 0
               if jumps[i] < 0: # dose decrease
                  pad = int(np.floor(jumps[i]/2)) # divide the gap in 2
                  hs[xyz(i,oax1,oax2) + (slice((v[i] + pad),v[i]),)] = 0 # pad down
                  hs[xyz(i+1,oax1,oax2) + (slice(v[i+1],(v[i+1]-pad)),)] = 0 # pad up
               else: # dose increase
                  pad = int(np.ceil(jumps[i]/2))
                  hs[xyz(i,oax1,oax2) + (slice(v[i],(v[i]+pad)),)] = 0 # pad up
                  hs[xyz(i+1,oax1,oax2) + (slice((v[i+1] - pad),v[i+1]),)] = 0 # pad down
      return hs

   def _interp_dose_along_ax2(self,hypsrf,lookup,ax):
      """ interpolate the dose axis along a given spatial axis (2D)

          lookup: a matrix of the shape of spatial dimensions where each elements
                  corresponds to the dose bin at that point (produced by np.digitize)
          hypsrfs: the hypersurface: [x,y,z,dose_bin]
          ax: the index along which the interpolation is done (x=0,y=1,z=2)
      """

      dims = self.ref_img.shape # the spatial dimensions triplet (x, y, z)
      hs = np.copy(hypsrf)

      ax_slc = slice(None) # take the whole ax (same as [:])
      ax_order = [ax] + [x for x in [0,1] if x != ax] # [ax, oax1]
      ax_sort = np.argsort(ax_order)

      # a function that take (ax, oax1, oax2) and return (x,y,z)
      xy = lambda x,y : ([x,y][ax_sort[0]], [x,y][ax_sort[1]])

      for oax1 in range(dims[ax_order[1]]):
         #for oax2 in range(dims[ax_order[2]]):
         v = lookup[xy(ax_slc,oax1)]
         jumps = np.diff(v) # number of pixels in gap
         idx_jump = np.argwhere(np.abs(jumps) > 1).flatten() # next dose jumps by more than 1 bin

         for i in idx_jump:
            # starting point to pad on hyper surface: (i,v[i])
            pad = 0
            if jumps[i] < 0: # dose decrease
               pad = int(np.floor(jumps[i]/2)) # divide the gap in 2
               hs[xy(i,oax1) + (slice((v[i] + pad),v[i]),)] = 0 # pad down
               hs[xy(i+1,oax1) + (slice(v[i+1],(v[i+1]-pad)),)] = 0 # pad up
            else: # dose increase
               pad = int(np.ceil(jumps[i]/2))
               hs[xy(i,oax1) + (slice(v[i],(v[i]+pad)),)] = 0 # pad up
               hs[xy(i+1,oax1) + (slice((v[i+1] - pad),v[i+1]),)] = 0 # pad down
      return hs

   def compute_gamma(self,img):
      """ compute gamma between img and ref
          img must have the same number and size of pixels
      """

      assert self.ref_img.shape == img.shape, 'reference and test must have the same dimensions'

      lookup = np.digitize(img,self.dbins) - 1 # values to lookup in dist_map
      gamma_map = np.ones(img.shape)*999 # initialize
      print(gamma_map.shape)

      #gamma values corrresponds to the pixel values on dist_map at the location of img
      for i in range(self.ndbins):
         test_points = lookup == i
         if self.ndim == 3:
            gamma_map[test_points] = self.dist_map[:,:,:,i][test_points]
         elif self.ndim == 2:
            gamma_map[test_points] = self.dist_map[:,:,i][test_points]
         else:
            raise IndexError('Only 2 and 3 spatial dimension supported at this moment')
      return gamma_map

   def __call__(self,img):
      """ compute the gamma between ref and img
      """

      return self.compute_gamma(img)
