"""
flex imzML & mis file reader class
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

from pyimzml.ImzMLParser import ImzMLParser
import xml.etree.ElementTree as ET
import re
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import NamedTuple

RegImage = NamedTuple("RegImage", [('path', str), ('tf', np.array), ('coreg', bool)])
FlexRegion = NamedTuple("FlexRegion", [('name', str), ('points', np.array)])


class flex_imzML_reader():
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
        print(_d)
        for _img, _dat in _d.items():
            if full_affine:
                _rl.append(RegImage(_img, cv2.getAffineTransform(np.array(_dat['ps']).astype(np.float32),
                                                                 np.array(_dat['tps']).astype(np.float32)),
                                    _dat['CoReg']))
            else:
                _rl.append(RegImage(_img, cv2.estimateAffinePartial2D(np.array(_dat['ps']).astype(np.float32),
                                                                      np.array(_dat['tps']).astype(np.float32))[0],
                                    _dat['CoReg']))
        return _rl

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
            if not _img.coreg:
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
        for area in root_xml.iter('Area'):
            x = []
            y = []

            for child in area:
                if child.tag == 'Point':
                    raw_vals = child.text.split(',')
                    # print(raw_vals)
                    x.append(int(raw_vals[0]))
                    y.append(int(raw_vals[1]))
                    if int(raw_vals[0]) > max_x_pt[0]:
                        max_x_pt = [int(raw_vals[0]), int(raw_vals[1])]
                    if int(raw_vals[1]) > max_y_pt[1]:
                        max_y_pt = [int(raw_vals[0]), int(raw_vals[1])]
            _ret.append(FlexRegion(area.attrib['Name'], np.array((x, y)).T))
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
            if _img.coreg:
                _tm = np.dot(np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]), self.mscale),
                             np.vstack([cv2.invertAffineTransform(_img.tf), np.array([0, 0, 1])]))[:2, :]
            else:
                _tm = np.dot(self.mreg, self.mscale)
            _imgo = plt.imread(os.path.join(self.base_path, _img.path))
            _w, _h = tuple(np.ceil(self.transform([(_imgo.shape[1], _imgo.shape[0])], _tm)).astype(int)[0])
            _rd['tf_{}'.format(_img.path)] = cv2.warpAffine(_imgo, _tm, (_w, _h))
        return _rd

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
