import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os
import sys
import time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
import logging
from math import pi
import pdb
import matplotlib.pyplot as plt
import gen_figures as gf

if (sys.version_info > (3,0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()

    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_ = str(dataset)

        if split_name.lower() not in src_dir:
            src_dir = src_dir + '/' + split_name

        print('\n------Reading Lidar and Joints (IMU)------')
        lidar_data_ = split_name + '_lidar' + self.dataset_
        self.lidar_ = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=lidar_data_)

        print('\n------Reading Joints Data------')
        joint_data_ = split_name + '_joint' + self.dataset_
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=joint_data_)

        self.num_data_ = len(self.lidar_.data_)

        # Position of indices
        self.odo_indices_ = np.empty((2, self.num_data_), dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # Height of the lidar from the ground
        self.h_lidar_ = 0.93 + 0.33 + 0.15

        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9

        #TODO set a threshold value of probability to consider a map's cell occupied
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh

        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        self.num_p_ = num_p
        # Initialize particles
        self.particles_ = np.zeros((3, self.num_p_), dtype=np.float64) if particles is None else particles
        # Weights for the particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.zeros((2, self.num_data_), dtype=np.int64)
        # Best particles
        self.best_p_ = np.zeros((3, self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ = np.zeros(self.num_data_)

        # Covariance matrix of motion model
        tiny_mov_cov = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0, 1e-8]])
        self.mov_cov_ = mov_cov if mov_cov is not None else tiny_mov_cov

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        """ Input: resolution of map """
        MAP = {}
        MAP['res'] = map_resolution
        MAP['xmin'] = -20
        MAP['ymin'] = -20
        MAP['xmax'] = 20
        MAP['ymax'] = 20
        MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))
        MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)

        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.uint64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'], self.MAP_['sizey']), dtype=np.uint64)

    def _build_first_map(self, t0=0, use_lidar_yaw=True):
        """ Build the first map using first lidar"""
        self.t0 = t0

        MAP = self.MAP_
        print('\n -------- Building the first map --------')

        # Extract data from first lidar scan
        first_lidar = self.lidar_.data_[0]
        # Extract corresponding joint data
        joint_idx = np.where(first_lidar['t'][0] <= self.joints_.data_['ts'][0])[0][0]
        first_head_angle = self.joints_.data_['head_angles'][1,joint_idx]
        neck_angle = self.joints_.data_['head_angles'][0,joint_idx]

        pose = first_lidar['pose'][0]
        scans = first_lidar['scan'][0]
        lidar_res = first_lidar['res'][0][0]
        ray_angles = np.linspace(-2.355, 2.355, len(scans))

        dmin = np.zeros_like(scans)
        dmax = np.zeros_like(scans)
        last_occu = np.zeros_like(scans)

#        for i in range(len(scans)):
#
#            [dmin[i], dmax[i], last_occu[i], _] = self.lidar_._remove_ground(self.h_lidar_, ray_angles[i], scans[i], first_head_angle)

        [dmin, dmax, last_occu, _] = self.lidar_._remove_ground(self.h_lidar_, ray_angles, scans, first_head_angle)

        dmin = dmin[last_occu==1]
        dmax = dmax[last_occu==1]
        last_occu = last_occu[last_occu == 1]
        ray_angles = ray_angles[last_occu == 1]

        [sX, sY, eX, eY] = self.lidar_._ray2worldPhysicsPos(pose, neck_angle, np.array([dmin, dmax, last_occu, ray_angles]))

        [siX, siY] = self.lidar_._physicPos2Pos(MAP, [sX, sY])
        [eiX, eiY] = self.lidar_._physicPos2Pos(MAP, [eX, eY])

        # Get distance, ray_angle and then remove ground
        for i in range(len(scans)):
                # Get all cells covered by lidar ray
            covered_cells = self.lidar_._cellsFrom2Points([siX[i], siY[i], eiX[i], eiY[i]])

            covered_cells = covered_cells.astype(int)

            self.log_odds_[covered_cells[0,:-1], covered_cells[1,:-1]] += np.log(self.p_false_)
            self.log_odds_[covered_cells[0,-1], covered_cells[1,-1]] += np.log(self.p_true_)

        MAP['map'] = (self.log_odds_ > self.logodd_thresh_).astype(int)

        # End code
        self.MAP_ = MAP


    def _predict(self, t, use_lidar_yaw=True):
        logging.debug('\n -------- Doing prediction at t = {0} --------'.format(t))
        #TODO Integrate odometry later

        odom_prev = self.lidar_.data_[t-1]['pose'][0]
        odom_curr = self.lidar_.data_[t]['pose'][0]

        noise_mean = np.array([0,0,0])
        noise_vectors = np.random.multivariate_normal(noise_mean, self.mov_cov_, self.num_p_).T

        new_particles = np.zeros(self.particles_.shape)
        for i in range(self.num_p_):
            new_particles[:,i] = tf.twoDSmartPlus(tf.twoDSmartPlus(self.particles_[:,i], tf.twoDSmartMinus(odom_curr, odom_prev)), noise_vectors[:,i])
            # new_particles[:,i] = tf.twoDSmartPlus(self.particles_[:,i], noise_vectors[:,i])
        self.particles_ = new_particles


#    def _update(self, t, t0=0, fig='on'):
#
#        if t == t0:
#            self._build_first_map(t0, use_lidar_yaw=True)
#            return
#
#        # Get the lidar data at t
#        lidar_t = self.lidar_.data_[t]
#        scans = lidar_t['scan'][0]
#        lidar_res = lidar_t['res'][0][0]
#        # Get corresponding joint index
#        joint_idx = np.where(lidar_t['t'][0] < self.joints_.data_['ts'][0])[0][0]
#        head_angle, neck_angle = self.joints_.data_['head_angles'][:,joint_idx]
#
#        ray_angle = np.linspace(-2.355, 2.355, len(scans))
#
#        MAP = self.MAP_
#        correlations = np.zeros(self.num_p_)
#
#        dmin = np.zeros_like(scans)
#        dmax = np.zeros_like(scans)
#        last_occu = np.zeros_like(scans)
#
#        for i in range(len(scans)):
#            [dmin[i], dmax[i], last_occu[i], _] = self.lidar_._remove_ground(self.h_lidar_, ray_angle[i], scans[i], head_angle)
#
#        for i in range(self.num_p_):
#            particle_pose = self.particles_[:,i]
#            occupied_cells = []
#
#            [sX,sY,eX,eY] = self.lidar_._ray2worldPhysicsPos(particle_pose, neck_angle, np.array([dmin, dmax, last_occu, ray_angle]))
#
#            [eiX, eiY] = self.lidar_._physicPos2Pos(MAP, [eX,eY])
#
#            correlations[i] = prob.mapCorrelation(MAP['map'], np.array([eiX, eiY]))
#
#        self.weights_ = prob.update_weights(self.weights_, correlations)
#
#        # Find best particle
#        best_particle_idx = np.argmax(self.weights_)
#
#        self.best_p_[:,t] = self.particles_[:,best_particle_idx]
#
#        self.best_p_indices_[:,t] = self.lidar_._physicPos2Pos(MAP, self.best_p_[0:2,t])
#
#        dmin = np.zeros_like(scans)
#        dmax = np.zeros_like(scans)
#        last_occu = np.zeros_like(scans)
#
#        for i in range(len(scans)):
#
#            [dmin[i], dmax[i], last_occu[i], _] = self.lidar_._remove_ground(self.h_lidar_, ray_angle[i], scans[i], head_angle)
#
#        dmin = dmin[last_occu==1]
#        dmax = dmax[last_occu==1]
#        last_occu = last_occu[last_occu == 1]
#        ray_angle = ray_angle[last_occu == 1]
#
#        [sX, sY, eX, eY] = self.lidar_._ray2worldPhysicsPos(self.best_p_[:,t], neck_angle, np.array([dmin, dmax, last_occu, ray_angle]))
#
#        [siX, siY] = self.lidar_._physicPos2Pos(MAP, [sX, sY])
#        [eiX, eiY] = self.lidar_._physicPos2Pos(MAP, [eX, eY])
#
#        for i in range(len(siX)):
#                # Get all cells covered by lidar ray
#            covered_cells = self.lidar_._cellsFrom2Points([siX[i], siY[i], eiX[i], eiY[i]])
#
#            covered_cells = covered_cells.astype(int)
#
#            self.log_odds_[covered_cells[0,:-1], covered_cells[1,:-1]] += np.log(self.p_false_)
#            self.log_odds_[covered_cells[0,-1], covered_cells[1,-1]] += np.log(self.p_true_)
#
#
#        MAP['map'] = (self.log_odds_ > self.logodd_thresh_).astype(int)
#
#        self.MAP_ = MAP
#
#        return MAP
    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return
        # for one particle

        MAP = self.MAP_
        scans = self.lidar_.data_[t]['scan'][0]
        joint_idx = np.where(self.lidar_.data_[t]['t'][0]<=self.joints_.data_['ts'][0])[0][0]
        first_head_angle = self.joints_.data_['head_angles'][1,joint_idx]

        self.lidar_angles = np.linspace(-2.356,2.356,len(scans))
        scan_clean = np.zeros((4,len(self.lidar_angles)))
        scan_world = np.zeros((4,len(self.lidar_angles)))
#        for i in range(len(self.lidar_angles)):
#            scan_clean[:,i] = self.lidar_._remove_ground(self.h_lidar_,ray_angle=self.lidar_angles[i],ray_l=scans[i],head_angle=first_head_angle)

        scan_clean = self.lidar_._remove_ground(self.h_lidar_, self.lidar_angles, scans, first_head_angle)
        corr = np.zeros(self.particles_.shape[1])

        for particle_num in range(self.particles_.shape[1]):
            pose = self.particles_[:,particle_num]

            scan_world = self.lidar_._ray2worldPhysicsPos(pose,first_head_angle,scan_clean)

            #[sx,sy] = self.lidar_._physicPos2Pos(MAP,scan_world[0:2,:])
            [ex,ey] = self.lidar_._physicPos2Pos(MAP,scan_world[2:4,:])
            #ex,ey = np.clip(ex,0,800),np.clip(ey,0,800)


            corr[particle_num] = prob.mapCorrelation(MAP['map'],np.array([ex,ey]))

        self.weights_ = prob.update_weights(self.weights_,corr)


        pose = self.particles_[:,np.where(np.max(self.weights_)==self.weights_)[0][0]]
        #pdb.set_trace()

        self.best_p_indices_[:,t] = self.lidar_._physicPos2Pos(MAP,pose[0:2])
        self.best_p_[:,t] = pose
        #print('\n--------Build the map--------')
        # extract first lidar scan
        lidar_read = self.lidar_.data_[t]

        # extract corresponding joint data
        joint_idx = np.where(lidar_read['t'][0]<=self.joints_.data_['ts'][0])[0][0]
        first_head_angle = self.joints_.data_['head_angles'][1,joint_idx]

        pose = lidar_read['pose'][0]
        scans = lidar_read['scan'][0]
        lidar_res = lidar_read['res'][0]

        self.lidar_angles = np.linspace(-2.356,2.356,len(scans))
        scan_clean = np.zeros((4,len(self.lidar_angles)))
        scan_world = np.zeros((4,len(self.lidar_angles)))
#        for i in range(len(self.lidar_angles)):
#            scan_clean[:,i] = self.lidar_._remove_ground(self.h_lidar_,ray_angle=self.lidar_angles[i],ray_l=scans[i],head_angle=first_head_angle)
#
        scan_clean = self.lidar_._remove_ground(self.h_lidar_, self.lidar_angles, scans, first_head_angle)

        scan_world = self.lidar_._ray2worldPhysicsPos(pose,first_head_angle,scan_clean)


        [sx,sy] = self.lidar_._physicPos2Pos(MAP,scan_world[0:2,:])
        [ex,ey] = self.lidar_._physicPos2Pos(MAP,scan_world[2:4,:])

        for i in range(len(sx)):
            [x_map,y_map] = self.lidar_._cellsFrom2Points([sx[i],sy[i],ex[i],ey[i]])

            self.num_m_per_cell_[x_map,y_map] += 1
            self.log_odds_[x_map[:-1],y_map[:-1]] += np.log(self.p_false_)
            self.log_odds_[x_map[-1],y_map[-1]] += np.log(self.p_true_)*scan_clean[2,i]

        if t == self.num_data_ - 1:
            MAP['map'] = self.log_odds_>self.logodd_thresh_

if __name__ == "__main__":
    slam_inc = SLAM()
    slam_inc._read_data('data/train', 0, 'train')
    slam_inc._init_particles(num_p=100)
    slam_inc._init_map()
    slam_inc._build_first_map()
    MAP = gf.genMap(slam_inc, end_t=1)
    plt.imshow(MAP)
    plt.show()


