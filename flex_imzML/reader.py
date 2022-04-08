"""
flex imzML & mis file reader class
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""
from dataclasses import dataclass, field

from pyimzml.ImzMLParser import ImzMLParser
import xml.etree.ElementTree as ET
import re
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import NamedTuple, List, Dict

RegImage = NamedTuple("RegImage", [('path', str), ('tf', np.array), ('coreg_mis', bool), ('coreg_to', int)])
FlexRegion = NamedTuple("FlexRegion", [('name', str), ('points', np.array)])


@dataclass
class MsiData:
    x: List[int] = field(default_factory=list)
    y: List[int] = field(default_factory=list)
    z: List[str] = field(default_factory=list)
    msi: np.array = None
    spectrum_mean: np.array = None
    spectrum_sum: np.array = None
    spectrum_mzs: np.array = None
    table: pd.DataFrame = None
    name: str = None
    meta: Dict = field(default_factory=dict)


class flexImzMLHandler():
    def __init__(self, base_name, base_path, full_affine=True, mscale=5):
        self._imzML_file = os.path.join(base_path, '{}.imzML'.format(base_name))
        self._mis_file = os.path.join(base_path, '{}.mis'.format(base_name))
        self.base_path = base_path
        if os.path.exists(self._mis_file):
            self._tree = ET.parse(self._mis_file)
            self._root = self._tree.getroot()
        else:
            raise ValueError("No mis file found!")
        if os.path.exists(self._imzML_file):
            self._p = ImzMLParser(self._imzML_file)
        else:
            raise ValueError("imzML file not found!")
        self.mscale = np.array([[mscale, 0, 0],
                                [0, mscale, 0],
                                [0, 0, 1]])
        self.imgs = self.extract_images(full_affine=full_affine)
        self.regions = self.extract_regions()
        self.mreg = self.get_mreg()
        # need to have this due to unknown imzML file origin
        # after finding the transformation matrix use extreme region points to cross check againts
        # extreme data points of imzML file to get these corrections (done in function check_translation)
        self.mreg_translation_x_correction = 0
        self.mreg_translation_y_correction = 0

    def extract_images(self, root_xml=None, full_affine=True):
        if root_xml is None:
            root_xml = self._root
        _d = {}
        _img = None
        _base_img = None
        for c in root_xml:
            if c.tag == "CoRegistration":
                for cc in c:
                    _img = self.proc_child(cc, _d, _img)
                _d[_img]['CoReg'] = True
            elif c.tag in ["TeachPoint", "ImageFile"]:
                _img = self.proc_child(c, _d, _img)
                if _img is not None:
                    _base_img = _img
                _d[_img]['CoReg'] = False

        _rl = []
        if _base_img is not None:
            _r0, _r1 = self.extract_raster(root_xml)
            # y axis is inverted in machine
            _min_x, _ = np.array(_d[_base_img]['tps']).min(axis=0)
            _, _min_y = -1 * (np.array(_d[_base_img]['tps']).max(axis=0))
            for i, _tps in enumerate(_d[_base_img]['tps']):
                _d[_base_img]['tps'][i] = [(_tps[0] - _min_x) / _r0, (-1 * _tps[1] - _min_y) / _r1]
        for _img, _dat in _d.items():
            if full_affine:
                _rl.append(RegImage(_img, cv2.getAffineTransform(np.array(_dat['ps']).astype(np.float32),
                                                                 np.array(_dat['tps']).astype(np.float32)),
                                    _dat['CoReg'], 0))
            else:
                _rl.append(RegImage(_img, cv2.estimateAffinePartial2D(np.array(_dat['ps']).astype(np.float32),
                                                                      np.array(_dat['tps']).astype(np.float32))[0],
                                    _dat['CoReg'], 0))
        return _rl

    def auto_register_img(self, moving_img_path, target_img_path=None, rigid=True, rotation=False, warn_angle_deg=1,
                          min_match_count=10, flann_index_kdtree=0, flann_trees=5, flann_checks=50):
        """
        co-registers two images and returns the moving image warped to fit target_img and the respective transform matrix
        Script is very close to OpenCV2 image co-registration tutorial:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
        only addition/change here: rigid transform & no rotation option
        :param moving_img: image supposed to move
        :param target_img: reference / target image, default None -> uses the existing coregistered image from mis file
        :param rigid: if true only tranlation, rotation, and uniform scale
        :param rotation: if false no rotation
        :param warn_angle_deg: cuttoff for warning check if supposed rotation angle bigger in case of rotation=False
        :param min_match_count: min good feature matches
        :param flann_index_kdtree: define algorithm for Fast Library for Approximate Nearest Neighbors - see FLANN doc
        :return: moved/transformed image in target image "space" & transformation matrix
        """
        target_img = None
        registered_to_internal_img = 0
        if target_img_path is None:
            for _idx, _img in enumerate(self.imgs):
                if _img.coreg_mis:
                    target_img = cv2.imread(os.path.join(self.base_path, _img.path), cv2.IMREAD_UNCHANGED)
                    registered_to_internal_img = _idx
            if target_img is None:
                # if this is still true we have a problem...
                raise RuntimeError('No coregistered image associated with this object - please provide a target image')
        else:
            target_img = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
        moving_img = cv2.imread(moving_img_path, cv2.IMREAD_UNCHANGED)
        if len(target_img.shape) > 2:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        if len(moving_img.shape) > 2:
            moving_img_gray = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
        else:
            moving_img_gray = moving_img
        height, width = target_img.shape

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(moving_img_gray, None)
        kp2, des2 = sift.detectAndCompute(target_img, None)

        index_params = dict(algorithm=flann_index_kdtree, trees=flann_trees)
        search_params = dict(checks=flann_checks)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            if rigid:
                transformation_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                                          ransacReprojThreshold=5.0)
                transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
                if not rotation:
                    angle = np.arcsin(transformation_matrix[0, 1])
                    print('Current rotation {} degrees'.format(np.rad2deg(angle)))
                    if abs(np.rad2deg(angle)) > warn_angle_deg:
                        print('Warning: calculated rotation > {} degrees!'.format(warn_angle_deg))
                    pure_scale = transformation_matrix[0, 0] / np.cos(angle)
                    transformation_matrix[0, 0] = pure_scale
                    transformation_matrix[0, 1] = 0
                    transformation_matrix[1, 0] = 0
                    transformation_matrix[1, 1] = pure_scale
            else:
                transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            transformed_img = cv2.warpPerspective(moving_img, transformation_matrix, (width, height))
            print('Transformation matrix: {}'.format(transformation_matrix))
            if registered_to_internal_img > 0:
                self.imgs.append(RegImage(moving_img_path, transformation_matrix, False, registered_to_internal_img))
        else:
            print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
            matchesMask = None
            transformed_img, transformation_matrix = None, None
        return transformed_img, transformation_matrix

    @staticmethod
    def proc_child(c, _d, _img):
        if c.tag == "ImageFile":
            _img = c.text
            _d[_img] = {}
        if c.tag == "TeachPoint":
            raw = re.split(';|,', c.text)
            if 'ps' not in _d[_img]:
                _d[_img]['ps'] = []
                _d[_img]['tps'] = []
            _d[_img]['ps'].append([float(raw[0]), float(raw[1])])
            _d[_img]['tps'].append([float(raw[2]), float(raw[3])])
        return _img

    def get_mreg(self):
        for _img in self.imgs:
            if not _img.coreg_mis:
                if self.check_translation(np.hstack([_img.tf[0:2, 0:2], np.array([[-1 * self.mscale[0,0]], [-1 * self.mscale[1,1]]])])):
                    return np.hstack([_img.tf[0:2, 0:2], np.array([[-1 * self.mscale[0,0]], [-1 * self.mscale[1,1]]])])
                else:
                    return _img.tf

    def check_translation(self, m, tolerance=1.5):
        # assumption 1: at least the max x / y value of the two extreme points should be the same in imzml and regions
        # assumption 2: imzML origin is the same as for the intially registered image - so translation should not be req.
        # imzml has a shift of 1 * scalefactor
        # it is not perfect but works ok
        imx, imy = self.get_imzML_max_xy()
        rmx, rmy = self.get_regions_max_xy()
        if (abs(imx - self.transform([rmx], m))[0,0] > tolerance * self.mscale[0,0]) or (
                abs(imy - self.transform([rmy], m))[0,1] > tolerance * self.mscale[1,1]):
            # in this case we do not need to correct the translation in m just drop it completely
            return False
        else:
            # here we need to correct the translation
            self.mreg_translation_x_correction = (self.transform([rmx], m) - imx + 1 * self.mscale[0,0])[0,0]
            self.mreg_translation_y_correction = (self.transform([rmy], m) - imy + 1 * self.mscale[1,1])[0,1]
            return True

    def extract_regions(self, root_xml=None):
        if root_xml is None:
            root_xml = self._root
        _ret = []
        max_x_pt = [0, 0]
        max_y_pt = [0, 0]
        for item in root_xml:
            if item.tag in ['Area', 'ROI']:
                x = []
                y = []

                for child in item:
                    if child.tag == 'Point':
                        raw_vals = child.text.split(',')
                        # print(raw_vals)
                        x.append(int(raw_vals[0]))
                        y.append(int(raw_vals[1]))
                        if int(raw_vals[0]) > max_x_pt[0]:
                            max_x_pt = [int(raw_vals[0]), int(raw_vals[1])]
                        if int(raw_vals[1]) > max_y_pt[1]:
                            max_y_pt = [int(raw_vals[0]), int(raw_vals[1])]
                _ret.append(FlexRegion(item.attrib['Name'], np.array((x, y)).T))
        return _ret

    @staticmethod
    def extract_raster(root_xml):
        for area in root_xml.iter('Area'):
            for child in area:
                if child.tag == 'Raster':
                    _r1, _r2 = child.text.split(',')
                    return int(_r1), int(_r2)

    def get_transformed_regions(self):
        _rd = {}
        for _reg in self.regions:
            _rd[_reg.name] = FlexRegion(_reg.name, self.transform(_reg.points, np.dot(self.mreg, self.mscale)))
        return _rd

    def get_transformed_images(self):
        _rd = {}
        for _img in self.imgs:
            _img_p = None
            _target_img = None
            if _img.coreg_mis:
                _tm = np.dot(np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]), self.mscale),
                             np.vstack([cv2.invertAffineTransform(_img.tf), np.array([0, 0, 1])]))[:2, :]
            elif _img.coreg_to > 0:
                _tm = np.dot(np.dot(np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]), self.mscale),
                                    np.vstack([cv2.invertAffineTransform(self.imgs[_img.coreg_to].tf),
                                               np.array([0, 0, 1])])), _img.tf)[:2, :]
                _img_p = _img.path
                _target_img = cv2.imread(os.path.join(self.base_path, self.imgs[_img.coreg_to].path),
                                         cv2.IMREAD_UNCHANGED)
            else:
                _tm = np.dot(self.mreg, self.mscale)
            if _img_p is None:
                _img_p = os.path.join(self.base_path, _img.path)
            _imgo = plt.imread(_img_p)
            if _target_img is None:
                _target_img = _imgo
            _w, _h = tuple(np.ceil(self.transform([(_target_img.shape[1], _target_img.shape[0])], _tm)).astype(int)[0])
            _rd['tf_{}'.format(_img.path)] = cv2.warpAffine(_imgo, _tm, (_w, _h))
        return _rd

    @staticmethod
    def is_inside_cnt(cnt, x, y):
        if cv2.pointPolygonTest(cnt.round().astype(np.int32), (x, y), True) >= 0:
            return True
        else:
            return False

    @staticmethod
    def _use_point(x, y, unique_x, unique_y, cnt):
        if cnt is None:
            return x in unique_x and y in unique_y
        else:
            return flexImzMLHandler.is_inside_cnt(cnt, x, y)

    def get_msi_data(self, mz_intervals, x_interval=None, y_interval=None, intensity_f=None, norm_f=None, name=None,
                     inside_cnt=None, gen_spec=True, baseline_f=None, smooth_f=None):
        _mz_int_bounds = {}
        _msi_data = MsiData()
        _unique_x = np.array(list(set(sorted(np.array(self._p.coordinates)[:, 0]))))
        _unique_y = np.array(list(set(sorted(np.array(self._p.coordinates)[:, 1]))))
        if x_interval is not None:
            _unique_x = _unique_x[np.where(_unique_x == x_interval[0])[0][0]:np.where(_unique_x == x_interval[1])[0][0] + 1]
        if y_interval is not None:
            _unique_y = _unique_y[np.where(_unique_y == y_interval[0])[0][0]:np.where(_unique_y == y_interval[1])[0][0] + 1]
        if intensity_f is None:
            intensity_f = np.sum
        if norm_f is None:
            norm_f = self._identity_norm
        if name is None:
            _data_name = 'msi data {}'.format(self._mis_file)
        else:
            _data_name = name
        _xy = []
        _index = []
        _int_cols = []
        _int_data = []
        _data_mtxs = []
        _mz_idx = []
        _int_sum = None
        _last_mzs = None
        _int_n = 0
        for idx, (x, y, z) in enumerate(self._p.coordinates):
            if flexImzMLHandler._use_point(x, y, _unique_x, _unique_y, inside_cnt):
                _xy.append([x, y])
                _index.append(idx)
                # this step is time intensive
                mzs, intensities = self._p.getspectrum(idx)
                # originals are read-only - so we make just one copy here to avoid issues with some functions
                mzs = mzs.copy()
                intensities = intensities.copy()
                if smooth_f is not None:
                    intensities = smooth_f(intensities)
                if baseline_f is not None:
                    intensities = baseline_f(intensities)
                if gen_spec:
                    if _int_sum is None:
                        _int_sum = intensities
                        _last_mzs = mzs
                    else:
                        _int_sum += intensities
                        if not (_last_mzs == mzs).all():
                            raise RuntimeError(
                                "mzs not aligned in imzML file - currently not supported for mean & sum spectra generation - run again with gen_spec=False")
                        _last_mzs = mzs.copy()
                    _int_n += 1
                _row = []
                for _mz, _interval in mz_intervals:
                    if _mz not in _mz_int_bounds:
                        _mz_int_bounds[_mz] = {}
                        _data_mtxs.append(np.empty((len(_unique_y), len(_unique_x))))
                        _data_mtxs[-1][:] = np.nan
                        _mz_idx.append(_mz)
                        # find_nearest returns tuple(mz_value, index)
                        _mz_int_bounds[_mz]['l'] = self.find_nearest(mzs, _mz * (1 - _interval))
                        _mz_int_bounds[_mz]['u'] = self.find_nearest(mzs, _mz * (1 + _interval))
                        _int_cols.append('{} Da +/- {}% Int (true range: {} - {} mz)'.format(_mz, 100 * _interval,
                                                                                             _mz_int_bounds[_mz]['l'][0],
                                                                                             _mz_int_bounds[_mz]['u'][0]))
                    _row.append(norm_f(intensity_f(intensities[_mz_int_bounds[_mz]['l'][1]:_mz_int_bounds[_mz]['u'][1]]),
                                          intensities))
                    _data_mtxs[_mz_idx.index(_mz)][np.where(_unique_y == y)[0][0], np.where(_unique_x == x)[0][0]] = _row[-1]
                _int_data.append(_row)
        a_xy = np.array(_xy)
        tf_a_xy = self.transform(a_xy, self.mscale)
        df = pd.DataFrame()
        df['index'] = _index
        df['x'] = a_xy[:, 0]
        df['y'] = a_xy[:, 1]
        df['x_scaled'] = tf_a_xy[:, 0]
        df['y_scaled'] = tf_a_xy[:, 1]
        df.set_index(keys='index', inplace=True)
        _msi_data.table = pd.concat([df, pd.DataFrame(index=_index, columns=_int_cols, data=_int_data)], axis=1, verify_integrity=True)
        _msi_data.x = (_unique_x * self.mscale[0,0]).tolist()
        _msi_data.y = (_unique_y * self.mscale[1,1]).tolist()
        _msi_data.z = _int_cols
        _msi_data.msi = np.array(_data_mtxs)
        _msi_data.meta['normalization function'] = norm_f
        _msi_data.meta['intensity calc function'] = intensity_f
        _msi_data.name = _data_name
        if gen_spec:
            _msi_data.spectrum_mean = _int_sum / _int_n
            _msi_data.spectrum_mzs = _last_mzs
            _msi_data.spectrum_sum = _int_sum
        return _msi_data

    def get_scaled_msi(self, mz, interval=0.00025, break_at=100000, normalize=None):
        _mz_l = None
        _mz_u = None
        _idx_l = None
        _idx_u = None
        _xy = []
        _int = []
        _unique_x = np.array(list(set(sorted(np.array(self._p.coordinates)[:, 0]))))
        _unique_y = np.array(list(set(sorted(np.array(self._p.coordinates)[:, 1]))))
        _data_mtx = np.empty((len(_unique_x), len(_unique_y)))
        _data_mtx[:] = np.nan
        if normalize is None:
            normalize = self._identity_norm
        for idx, (x, y, z) in enumerate(self._p.coordinates):
            mzs, intensities = self._p.getspectrum(idx)
            if _mz_l is None:
                _mz_l, _idx_l = self.find_nearest(mzs, mz * (1 - interval))
                _mz_u, _idx_u = self.find_nearest(mzs, mz * (1 + interval))
            _xy.append([x, y])
            _int.append(normalize(np.sum(intensities[_idx_l:_idx_u]), intensities))
            _data_mtx[np.where(_unique_x == x)[0][0], np.where(_unique_y == y)[0][0]] = _int[-1]
            if idx > break_at:
                print('processing aborted due to break_at={} parameter'.format(break_at))
                break
        a_xy = np.array(_xy)
        tf_a_xy = self.transform(a_xy, self.mscale)
        df = pd.DataFrame()
        df['x'] = a_xy[:, 0]
        df['y'] = a_xy[:, 1]
        df['x_scaled'] = tf_a_xy[:, 0]
        df['y_scaled'] = tf_a_xy[:, 1]
        intcol = '{} Da +/- {}% Int'.format(mz, 100 * interval)
        df[intcol] = _int
        return df, pd.DataFrame(_data_mtx.T,
                                columns=sorted(df['x_scaled'].unique()),
                                index=sorted(df['y_scaled'].unique()))

    def get_imzML_max_xy(self):
        imzml_max_x = [0, 0]
        imzml_max_y = [0, 0]
        for idx, (x, y, z) in enumerate(self._p.coordinates):
            # mzs, intensities = p.getspectrum(idx)
            if x > imzml_max_x[0]:
                imzml_max_x = [x, y]
            if y > imzml_max_y[1]:
                imzml_max_y = [x, y]
        return np.array(imzml_max_x), np.array(imzml_max_y)

    def get_regions_max_xy(self):
        m_x = np.array([0, 0])
        m_y = np.array([0, 0])
        for _reg in self.regions:
            for e in _reg.points:
                if e[0] > m_x[0]:
                    m_x = e
                if e[1] > m_y[1]:
                    m_y = e
        return m_x, m_y

    @staticmethod
    def transform(points, mtx):
        tmp = []
        for p in points:
            tmp.append(np.dot(mtx[0:2, 0:2], p) + mtx[0:2, 2])
        return np.array(tmp)

    @staticmethod
    def _identity_norm(x, y):
        return x
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    @staticmethod
    def simplify_contour(cnt, simplify_factor=0.001):
        cnt = np.round(cnt).astype(int)
        return cv2.approxPolyDP(cnt, simplify_factor * cv2.arcLength(cnt, True), True)[:, 0, :]

    def inject_contour_into_mis(self, cnt, name, color='#000000', mtf=None, simplify=True, simplify_factor=0.001, save=True):
        roi = ET.SubElement(self._root, 'ROI')
        roi.set('Type', '3')
        roi.set('Name', name)
        roi.set('Enabled', '0')
        roi.set('ShowSpectra', '0')
        roi.set('SpectrumColor', color)
        if mtf is not None:
            if mtf == 'auto':
                cnt = self.transform(cnt, np.linalg.inv(np.vstack([np.dot(self.mreg, self.mscale), np.array([0, 0, 1])])))
            else:
                cnt = self.transform(cnt, mtf)
        cnt = np.round(cnt).astype(int)
        if simplify:
            cnt = flexImzMLHandler.simplify_contour(cnt=cnt, simplify_factor=simplify_factor)
        for x, y in cnt:
            _p = ET.SubElement(roi, 'Point')
            _p.text = '{},{}'.format(x, y)
        if save:
            self.save_mis_file()
        return roi, cnt

    def save_mis_file(self, filename_mod='_mod'):
        self._tree.write('{}'.format(filename_mod).join(os.path.splitext(self._mis_file)))
        return None


