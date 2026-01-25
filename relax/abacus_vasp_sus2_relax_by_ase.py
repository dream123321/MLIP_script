#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作者信息:
    ============================================================
    作者: 黄晶
    单位: 南方科技大学
    邮箱: 2760344463@qq.com
    开发时间: 2026.1.22
    更新: 2024.1.25 - 添加多种优化器支持
    更新: 2024.1.26 - 简化参数设置，只使用默认参数
"""

import os
import sys
import time
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE, MDMin, GPMin, LBFGSLineSearch, BFGSLineSearch
from ase.filters import ExpCellFilter
from ase.units import GPa
import pandas as pd
from typing import Optional, Dict, Any


class CalculatorFactory:
    """计算器工厂类，创建VASP、ABACUS或MTP计算器"""

    @staticmethod
    def create_vasp_calculator(vasp_pp, num_cores, vasp_std, out_path):
        """创建VASP计算器"""
        from ase.calculators.vasp import Vasp

        os.environ['VASP_PP_PATH'] = vasp_pp

        calc = Vasp(
            command=f'mpirun -np {int(num_cores)} {vasp_std}',
            xc='PBE',
            encut=520,
            kspacing=0.2,
            nsw=0,
            ibrion=-1,
            ismear=0,
            sigma=0.05,
            prec='Normal',
            lreal='Auto',
            nelm=100,
            ediff=1e-6,
            lwave=False,
            lcharg=False,
            directory=out_path
        )
        return calc

    @staticmethod
    def create_abacus_calculator(abacus, mpi_cores, out_dir):
        """创建ABACUS计算器"""
        from ase.calculators.abacus import Abacus, AbacusProfile
        from ase.parallel import world

        def abacus_set_calculator(omp, mpi, abacus_path, directory, parameters, parallel=True):
            os.environ['OMP_NUM_THREADS'] = f'{omp}'
            profile = AbacusProfile(command=f"mpirun -np {mpi} {abacus_path}")

            if parallel:
                out_directory = f"{directory}-rank{world.rank}"
            else:
                out_directory = directory

            calc = Abacus(profile=profile, directory=out_directory, **parameters)
            return calc

        # 赝势和轨道文件字典
        pp_dic = {
            "Ag": "Ag_ONCV_PBE-1.0.upf", "Al": "Al_ONCV_PBE-1.0.upf", "Ar": "Ar_ONCV_PBE-1.0.upf",
            "As": "As_ONCV_PBE-1.0.upf", "Au": "Au_ONCV_PBE-1.0.upf", "Ba": "Ba_ONCV_PBE-1.0.upf",
            "Be": "Be_ONCV_PBE-1.0.upf", "Bi": "Bi_ONCV_PBE-1.0.upf", "Br": "Br_ONCV_PBE-1.0.upf",
            "B": "B_ONCV_PBE-1.0.upf", "Ca": "Ca_ONCV_PBE-1.0.upf", "Cd": "Cd_ONCV_PBE-1.0.upf",
            "Cl": "Cl_ONCV_PBE-1.0.upf", "Co": "Co_ONCV_PBE-1.0.upf", "Cr": "Cr_ONCV_PBE-1.0.upf",
            "Cs": "Cs_ONCV_PBE-1.0.upf", "Cu": "Cu_ONCV_PBE-1.0.upf", "C": "C_ONCV_PBE-1.0.upf",
            "Fe": "Fe_ONCV_PBE-1.0.upf", "F": "F_ONCV_PBE-1.0.upf", "Ga": "Ga_ONCV_PBE-1.0.upf",
            "Ge": "Ge_ONCV_PBE-1.0.upf", "He": "He_ONCV_PBE-1.0.upf", "Hf": "Hf_ONCV_PBE-1.0.upf",
            "Hg": "Hg_ONCV_PBE-1.0.upf", "H": "H_ONCV_PBE-1.0.upf", "In": "In_ONCV_PBE-1.0.upf",
            "Ir": "Ir_ONCV_PBE-1.0.upf", "I": "I_ONCV_PBE-1.0.upf", "Kr": "Kr_ONCV_PBE-1.0.upf",
            "K": "K_ONCV_PBE-1.0.upf", "La": "La_ONCV_PBE-1.0.upf", "Li": "Li_ONCV_PBE-1.0.upf",
            "Mg": "Mg_ONCV_PBE-1.0.upf", "Mn": "Mn_ONCV_PBE-1.0.upf", "Mo": "Mo_ONCV_PBE-1.0.upf",
            "Na": "Na_ONCV_PBE-1.0.upf", "Nb": "Nb_ONCV_PBE-1.0.upf", "Ne": "Ne_ONCV_PBE-1.0.upf",
            "Ni": "Ni_ONCV_PBE-1.0.upf", "N": "N_ONCV_PBE-1.0.upf", "Os": "Os_ONCV_PBE-1.0.upf",
            "O": "O_ONCV_PBE-1.0.upf", "Pb": "Pb_ONCV_PBE-1.0.upf", "Pd": "Pd_ONCV_PBE-1.0.upf",
            "Pt": "Pt_ONCV_PBE-1.0.upf", "P": "P_ONCV_PBE-1.0.upf", "Rb": "Rb_ONCV_PBE-1.0.upf",
            "Re": "Re_ONCV_PBE-1.0.upf", "Rh": "Rh_ONCV_PBE-1.0.upf", "Ru": "Ru_ONCV_PBE-1.0.upf",
            "Sb": "Sb_ONCV_PBE-1.0.upf", "Sc": "Sc_ONCV_PBE-1.0.upf", "Se": "Se_ONCV_PBE-1.0.upf",
            "Si": "Si_ONCV_PBE-1.0.upf", "Sn": "Sn_ONCV_PBE-1.0.upf", "Sr": "Sr_ONCV_PBE-1.0.upf",
            "S": "S_ONCV_PBE-1.0.upf", "Ta": "Ta_ONCV_PBE-1.0.upf", "Tc": "Tc_ONCV_PBE-1.0.upf",
            "Te": "Te_ONCV_PBE-1.0.upf", "Ti": "Ti_ONCV_PBE-1.0.upf", "Tl": "Tl_ONCV_PBE-1.0.upf",
            "V": "V_ONCV_PBE-1.0.upf", "W": "W_ONCV_PBE-1.0.upf", "Xe": "Xe_ONCV_PBE-1.0.upf",
            "Y": "Y_ONCV_PBE-1.0.upf", "Zn": "Zn_ONCV_PBE-1.0.upf", "Zr": "Zr_ONCV_PBE-1.0.upf"
        }

        orb_dic = {
            "Ag": "Ag_gga_10au_100Ry_4s2p2d1f.orb", "Al": "Al_gga_10au_100Ry_4s4p1d.orb",
            "Ar": "Ar_gga_10au_100Ry_2s2p1d.orb", "As": "As_gga_10au_100Ry_2s2p1d.orb",
            "Au": "Au_gga_10au_100Ry_4s2p2d1f.orb", "Ba": "Ba_gga_10au_100Ry_4s2p2d1f.orb",
            "Be": "Be_gga_10au_100Ry_4s1p.orb", "Bi": "Bi_gga_10au_100Ry_2s2p2d1f.orb",
            "Br": "Br_gga_10au_100Ry_2s2p1d.orb", "B": "B_gga_10au_100Ry_2s2p1d.orb",
            "Ca": "Ca_gga_10au_100Ry_4s2p1d.orb", "Cd": "Cd_gga_10au_100Ry_4s2p2d1f.orb",
            "Cl": "Cl_gga_10au_100Ry_2s2p1d.orb", "Cr": "Cr_gga_10au_100Ry_4s2p2d1f.orb",
            "Cs": "Cs_gga_10au_100Ry_4s2p1d.orb", "Cu": "Cu_gga_10au_100Ry_4s2p2d1f.orb",
            "C": "C_gga_10au_100Ry_2s2p1d.orb", "Fe": "Fe_gga_10au_100Ry_4s2p2d1f.orb",
            "F": "F_gga_10au_100Ry_2s2p1d.orb", "Ga": "Ga_gga_10au_100Ry_2s2p2d1f.orb",
            "Ge": "Ge_gga_10au_100Ry_2s2p2d1f.orb", "He": "He_gga_10au_100Ry_2s1p.orb",
            "Hf": "Hf_gga_10au_100Ry_4s2p2d2f1g.orb", "Hg": "Hg_gga_10au_100Ry_4s2p2d1f.orb",
            "H": "H_gga_10au_100Ry_2s1p.orb", "In": "In_gga_10au_100Ry_2s2p2d1f.orb",
            "Ir": "Ir_gga_10au_100Ry_4s2p2d1f.orb", "I": "I_gga_10au_100Ry_2s2p2d1f.orb",
            "Kr": "Kr_gga_10au_100Ry_2s2p1d.orb", "K": "K_gga_10au_100Ry_4s2p1d.orb",
            "Li": "Li_gga_10au_100Ry_4s1p.orb", "Mg": "Mg_gga_10au_100Ry_4s2p1d.orb",
            "Mn": "Mn_gga_10au_100Ry_4s2p2d1f.orb", "Mo": "Mo_gga_10au_100Ry_4s2p2d1f.orb",
            "Na": "Na_gga_10au_100Ry_4s2p1d.orb", "Nb": "Nb_gga_10au_100Ry_4s2p2d1f.orb",
            "Ne": "Ne_gga_10au_100Ry_2s2p1d.orb", "Ni": "Ni_gga_10au_100Ry_4s2p2d1f.orb",
            "N": "N_gga_10au_100Ry_2s2p1d.orb", "Os": "Os_gga_10au_100Ry_4s2p2d1f.orb",
            "O": "O_gga_10au_100Ry_2s2p1d.orb", "Pb": "Pb_gga_10au_100Ry_2s2p2d1f.orb",
            "Pd": "Pd_gga_10au_100Ry_4s2p2d1f.orb", "Pt": "Pt_gga_10au_100Ry_4s2p2d1f.orb",
            "P": "P_gga_10au_100Ry_2s2p1d.orb", "Rb": "Rb_gga_10au_100Ry_4s2p1d.orb",
            "Re": "Re_gga_10au_100Ry_4s2p2d1f.orb", "Rh": "Rh_gga_10au_100Ry_4s2p2d1f.orb",
            "Ru": "Ru_gga_10au_100Ry_4s2p2d1f.orb", "Sb": "Sb_gga_10au_100Ry_2s2p2d1f.orb",
            "Sc": "Sc_gga_10au_100Ry_4s2p2d1f.orb", "Se": "Se_gga_10au_100Ry_2s2p1d.orb",
            "Si": "Si_gga_10au_100Ry_2s2p1d.orb", "Sn": "Sn_gga_10au_100Ry_2s2p2d1f.orb",
            "Sr": "Sr_gga_10au_100Ry_4s2p1d.orb", "S": "S_gga_10au_100Ry_2s2p1d.orb",
            "Ta": "Ta_gga_10au_100Ry_4s2p2d2f1g.orb", "Tc": "Tc_gga_10au_100Ry_4s2p2d1f.orb",
            "Te": "Te_gga_10au_100Ry_2s2p2d1f.orb", "Ti": "Ti_gga_10au_100Ry_4s2p2d1f.orb",
            "Tl": "Tl_gga_10au_100Ry_2s2p2d1f.orb", "V": "V_gga_10au_100Ry_4s2p2d1f.orb",
            "W": "W_gga_10au_100Ry_4s2p2d2f1g.orb", "Xe": "Xe_gga_10au_100Ry_2s2p2d1f.orb",
            "Y": "Y_gga_10au_100Ry_4s2p2d1f.orb", "Zn": "Zn_gga_10au_100Ry_4s2p2d1f.orb",
            "Zr": "Zr_gga_10au_100Ry_4s2p2d1f.orb", "Co": "Co_gga_10au_100Ry_4s2p2d1f.orb"
        }

        parameters = {
            'calculation': 'scf',
            'ecutwfc': 100,
            'dft_functional': 'PBE',
            'symmetry': 0,
            'basis_type': 'lcao',
            'scf_thr': 1e-6,
            'scf_nmax': 100,
            'gamma_only': 0,
            'kspacing': 0.2,
            'smearing_method': 'gauss',
            'smearing_sigma': 0.003,
            'mixing_type': 'pulay',
            'mixing_beta': 0.4,
            'mixing_ndim': 12,
            'pp': pp_dic,
            'basis': orb_dic,
            'pseudo_dir': "/share/home/xill/hyx/abacus_pp/SG15_v1.0_Pseudopotential/liushi_ONCV_v1.0_upf",
            'basis_dir': "/share/home/xill/hyx/abacus_pp/all_DZP_10au",
            'cal_force': 1,
            'cal_stress': 1,
            'out_stru': 1,
        }

        return abacus_set_calculator(
            omp=1,
            mpi=mpi_cores,
            abacus_path=abacus,
            directory=out_dir,
            parameters=parameters
        )

    @staticmethod
    def create_mtp_calculator(potential_file, ele_list=None, compute_stress=True, stress_weight=1.0):
        """
        创建MTP计算器

        Args:
            potential_file: MTP势函数文件路径
            ele_list: 元素列表，[H, C, O]
            compute_stress: 是否计算应力
            stress_weight: 应力权重
        """

        try:
            from pymlip.core import MTPCalactor
            from ase.data import atomic_numbers
            from ase.calculators.calculator import Calculator, all_changes
            from ase import Atoms
            import numpy as np
        except ImportError as e:
            print(f"导入MTP相关模块失败: {e}")
            raise

        class MTPCalculator(Calculator):
            """MTP calculator based on ASE Calculator"""

            implemented_properties = ["energy", "forces", "energies", "stress"]

            def __init__(self, potential: str = "p.mtp", mtpcalc: MTPCalactor = None,
                         ele_list: list = None, compute_stress: bool = True,
                         stress_weight: float = 1.0, print_EK: bool = True, **kwargs):
                super().__init__(**kwargs)
                self.potential = potential
                self.compute_stress = compute_stress
                self.print_EK = print_EK
                self.stress_weight = stress_weight
                self.mtpcalc = MTPCalactor(self.potential)
                self.unique_numbers = sorted([atomic_numbers[i] for i in ele_list])

            def calculate(
                    self,
                    atoms: Atoms = None,
                    properties: list = None,
                    system_changes: list = None,
                    unique_numbers: list = None
            ):
                properties = properties or ["energy"]
                system_changes = system_changes or all_changes
                super().calculate(atoms=atoms, properties=properties,
                                  system_changes=system_changes)

                if unique_numbers is not None:
                    self.unique_numbers = unique_numbers

                try:
                    from pymlip.core import PyConfiguration
                    cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=self.unique_numbers)
                except ImportError:
                    cfg = atoms
                    use_pyconfiguration = False
                else:
                    use_pyconfiguration = True

                V = atoms.cell.volume

                if use_pyconfiguration:
                    self.mtpcalc.calc(cfg)
                    self.results['energy'] = np.array(cfg.energy)
                    self.results['forces'] = cfg.force
                    self.results['energies'] = np.array(cfg.site_energys)
                else:
                    raise NotImplementedError("直接使用MTPCalactor的方法未实现，请安装pymlip.core")

                if self.compute_stress and use_pyconfiguration:
                    self.results['stress'] = -np.array([
                        cfg.stresses[0, 0], cfg.stresses[1, 1], cfg.stresses[2, 2],
                        cfg.stresses[1, 2], cfg.stresses[0, 2], cfg.stresses[0, 1]
                    ]) * self.stress_weight / V

        return MTPCalculator(
            potential=potential_file,
            ele_list=ele_list,
            compute_stress=compute_stress,
            stress_weight=stress_weight
        )


class StructureOptimizer:
    """结构优化器，支持多种优化算法"""

    # 优化器映射表
    OPTIMIZER_MAP = {
        'bfgs': BFGS,
        'lbfgs': LBFGS,
        'fire': FIRE,
        'mdmin': MDMin,
        'gpmin': GPMin,
        'lbfgs_linesearch': LBFGSLineSearch,
        'bfgs_linesearch': BFGSLineSearch
    }

    # 优化器描述
    OPTIMIZER_DESCRIPTIONS = {
        'bfgs': 'BFGS (拟牛顿法，适用于大多数情况)',
        'lbfgs': 'L-BFGS (内存受限的BFGS，适用于大体系)',
        'fire': 'FIRE (快速惯性弛豫引擎，收敛快)',
        'mdmin': 'MDMin (分子动力学最小化，适用于崎岖势能面)',
        'gpmin': 'GPMin (Gaussian过程优化器)',
        'lbfgs_linesearch': 'L-BFGS with line search (带线搜索的L-BFGS，通常更稳定)',
        'bfgs_linesearch': 'BFGS with line search (带线搜索的BFGS，更精确的线搜索)'
    }

    def __init__(self, calculator, fmax=0.02, max_steps=300, maxstep=0.1,
                 optimizer='bfgs', optimizer_params=None):
        """
        初始化结构优化器

        Parameters:
        -----------
        calculator : ASE计算器对象
        fmax : float, 力收敛标准 (eV/Å)
        max_steps : int, 最大优化步数
        maxstep : float, 最大步长
        optimizer : str, 优化器类型 ['bfgs', 'lbfgs', 'fire', 'mdmin', 'gpmin', 'lbfgs_linesearch', 'bfgs_linesearch']
        optimizer_params : dict, 优化器特定参数（简化版，不传递特殊参数）
        """
        self.calculator = calculator
        self.fmax = fmax
        self.max_steps = max_steps
        self.maxstep = maxstep
        self.optimizer_type = optimizer.lower()
        self.optimizer_params = optimizer_params or {}

        # 验证优化器类型
        if self.optimizer_type not in self.OPTIMIZER_MAP:
            available = list(self.OPTIMIZER_MAP.keys())
            raise ValueError(f"不支持的优化器类型: {optimizer}。可用选项: {available}")

        print(f"使用优化器: {self.OPTIMIZER_DESCRIPTIONS.get(self.optimizer_type, self.optimizer_type)}")

    def _create_optimizer(self, atoms, use_cell_filter=True,
                          pressure=0.0, mask=None, constant_volume=False,
                          logfile=None):
        """
        创建优化器实例，只使用默认参数

        Returns:
        --------
        optimizer : 优化器实例
        atoms_or_filter : 优化的对象（Atoms或CellFilter）
        """
        atoms.calc = self.calculator

        if use_cell_filter:
            ucf = ExpCellFilter(
                atoms,
                scalar_pressure=pressure * GPa,
                mask=mask,
                constant_volume=constant_volume
            )
            atoms_or_filter = ucf
        else:
            atoms_or_filter = atoms

        # 获取优化器类
        optimizer_class = self.OPTIMIZER_MAP[self.optimizer_type]

        # 只使用最基本的参数
        params = {}

        # 设置日志文件（如果有）
        if logfile is not None:
            params['logfile'] = logfile

        # 创建优化器实例，只传递必要的参数
        # 不传递任何额外的优化器特定参数，使用ASE默认值
        try:
            optimizer = optimizer_class(atoms_or_filter, **params)
        except Exception as e:
            print(f"创建优化器时出错: {e}")
            # 如果出错，尝试不传递任何参数
            try:
                if logfile is not None:
                    optimizer = optimizer_class(atoms_or_filter, logfile=logfile)
                else:
                    optimizer = optimizer_class(atoms_or_filter)
            except Exception as e2:
                print(f"再次尝试创建优化器失败: {e2}")
                raise

        return optimizer, atoms_or_filter

    def relax(self, atoms, pressure=0.0, dim=3, constant_volume=False,
              out_process=False, fix_cell=False):
        """
        结构弛豫

        Parameters:
        -----------
        atoms : ASE Atoms对象
        pressure : float, 外部压强 (GPa)
        dim : int, 维度 (2或3)
        constant_volume : bool, 是否保持体积不变
        out_process : bool, 是否输出过程信息
        fix_cell : bool, 是否固定晶胞

        Returns:
        --------
        relaxed_atoms : 优化后的Atoms对象
        optimization_info : 优化过程信息字典
        """
        # 设置cell filter的mask
        use_cell_filter = not fix_cell
        mask = None

        if use_cell_filter:
            if dim == 2:
                # 2D材料：只优化z方向
                mask = [False, False, True, True, True, True]
            else:
                # 3D材料：优化所有方向
                mask = [True, True, True, True, True, True]

        # 创建优化器
        logfile = 'optimization.log' if out_process else None
        optimizer, atoms_or_filter = self._create_optimizer(
            atoms, use_cell_filter=use_cell_filter,
            pressure=pressure, mask=mask, constant_volume=constant_volume,
            logfile=logfile
        )

        # 记录优化开始时间
        start_time = time.time()

        try:
            # 运行优化
            optimizer.run(fmax=self.fmax, steps=self.max_steps)
            converged = optimizer.converged()
        except Exception as e:
            print(f"优化过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            print("返回当前状态的结构...")
            converged = False

        # 获取最终的结构
        if use_cell_filter:
            relaxed_atoms = atoms_or_filter.atoms
        else:
            relaxed_atoms = atoms

        # 获取优化过程信息
        optimization_time = time.time() - start_time

        # 读取优化日志获取迭代信息
        if out_process and os.path.exists('optimization.log'):
            with open('optimization.log', 'r') as f:
                lines = f.readlines()
                # 计算实际迭代步数（跳过标题行）
                nsteps = len([line for line in lines if line.strip() and
                              not line.startswith('Step') and not line.startswith('---')])
        else:
            nsteps = None

        optimization_info = {
            'converged': converged,
            'nsteps': nsteps,
            'optimization_time': optimization_time,
            'fmax_target': self.fmax,
            'max_steps_target': self.max_steps,
            'optimizer_type': self.optimizer_type
        }

        return relaxed_atoms, optimization_info

    def optimize(self, atoms, pressure=0.0, dim=3, constant_volume=False,
                 out_process=False, fix_cell=False, output_file_base='optimized'):
        """
        执行结构优化并返回结果

        Parameters:
        -----------
        atoms : ASE Atoms对象
        pressure : float, 外部压强 (GPa)
        dim : int, 维度
        constant_volume : bool, 是否保持体积不变
        out_process : bool, 是否输出过程信息
        fix_cell : bool, 是否固定晶胞
        output_file_base : str, 输出文件的基础名（不含扩展名）

        Returns:
        --------
        dict : 包含优化结果的字典
        """
        start_time = time.time()

        # 执行优化
        relaxed_atoms, opt_info = self.relax(
            atoms, pressure, dim, constant_volume, out_process, fix_cell
        )

        # 获取结果
        forces = relaxed_atoms.get_forces()
        energy = relaxed_atoms.get_potential_energy()

        # 尝试获取应力
        try:
            stress = relaxed_atoms.get_stress()
        except (NotImplementedError, AttributeError, KeyError):
            stress = None

        # 检查收敛性
        max_force = np.max(np.linalg.norm(forces, axis=1))
        converged = opt_info['converged']

        # 计算总时间
        total_time = time.time() - start_time

        # 保存优化后的结构
        calculator_type = str(type(self.calculator)).split('.')[-1].strip("'>")
        output_suffix = f"_by_{calculator_type}_{self.optimizer_type}"

        # 1. 保存为extxyz格式
        extxyz_file = f"{output_file_base}{output_suffix}.xyz"
        write(extxyz_file, relaxed_atoms, format='extxyz')

        # 2. 保存为VASP格式
        vasp_file = f"{output_file_base}{output_suffix}.vasp"
        write(vasp_file, relaxed_atoms, format='vasp', vasp5=True, direct=True)

        # 3. 如果未收敛，保存一个特殊标记的文件
        if not converged:
            unconverged_file = f"{output_file_base}{output_suffix}_unconverged.xyz"
            write(unconverged_file, relaxed_atoms, format='extxyz')

        # 返回结果
        results = {
            'converged': converged,
            'energy': energy,
            'max_force': max_force,
            'forces': forces,
            'stress': stress,
            'atoms': relaxed_atoms,
            'total_time': total_time,
            'optimization_time': opt_info['optimization_time'],
            'nsteps': opt_info['nsteps'],
            'volume': relaxed_atoms.get_volume(),
            'cell': relaxed_atoms.cell,
            'positions': relaxed_atoms.positions,
            'optimizer_type': self.optimizer_type,
            'output_files': {
                'extxyz': extxyz_file,
                'vasp': vasp_file,
                'unconverged': f"{output_file_base}{output_suffix}_unconverged.xyz" if not converged else None
            }
        }

        return results


def main_relax():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='ASE结构优化脚本（支持VASP、ABACUS和sus2，多种优化器）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python abacus_vasp_sus2_relax_by_ase.py 407.vasp --calculator vasp
  python abacus_vasp_sus2_relax_by_ase.py 407.vasp --calculator sus2 --optimizer lbfgs_linesearch
  python abacus_vasp_sus2_relax_by_ase.py 407.vasp --calculator abacus --optimizer bfgs_linesearch

优化器说明:
  bfgs:           BFGS拟牛顿法，适用于大多数情况
  lbfgs:          内存受限的BFGS，适用于大体系
  fire:           快速惯性弛豫引擎，收敛快
  mdmin:          分子动力学最小化，适用于崎岖势能面
  gpmin:          Gaussian过程优化器
  lbfgs_linesearch: 带线搜索的L-BFGS，通常更稳定 
  bfgs_linesearch:  带线搜索的BFGS，更精确的线搜索 （默认）

注意: 所有优化器都使用ASE默认参数，简化配置。

        """
    )

    parser.add_argument('input_file', type=str, help='输入结构文件（如POSCAR, CONTCAR等）')
    parser.add_argument('--calculator', type=str, choices=['vasp', 'abacus', 'sus2'],
                        default='sus2', help='计算器类型(默认sus2)')
    parser.add_argument('--optimizer', type=str,
                        choices=['bfgs', 'lbfgs', 'fire', 'mdmin', 'gpmin', 'lbfgs_linesearch', 'bfgs_linesearch'],
                        default='bfgs_linesearch', help='优化器类型(默认bfgs_linesearch)')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3, help='维度（2D或3D）(默认值3)')
    parser.add_argument('--pressure', type=float, default=0.0, help='外部压强（GPa）(默认值0.0)')
    parser.add_argument('--fix_cell', action='store_true',
                        help='是否固定晶格弛豫(默认不固定)，加了--fix_cell表示固定')
    parser.add_argument('--constant_volume', action='store_true',
                        help='是否保持体积不变(默认不保持)，加了--constant_volume表示保持')
    parser.add_argument('--output', type=str, default='optimized',
                        help='输出文件的基础名（不含扩展名）')
    parser.add_argument('--fmax', type=float, default=0.03, help='力收敛标准（eV/Å）(默认值0.03)')
    parser.add_argument('--max_steps', type=int, default=500, help='最大优化步数(默认值500)')
    parser.add_argument('--maxstep', type=float, default=0.1, help='最大步长(默认值0.1)')
    parser.add_argument('--out_process', action='store_true',
                        help='输出csv和优化过程记录')

    # MTP特定参数（保持）
    parser.add_argument('--mtp_potential', type=str, default='current.mtp',
                        help='MTP势函数文件路径(默认./current.mtp)')
    parser.add_argument('--ele_list', type=str, nargs='+',
                        default=['Na', 'Al', 'Si', 'S', 'Ge', 'Sn'],
                        help='元素符号列表，如 Na Al Si S Ge Sn (默认: Na Al Si S Ge Sn)')
    parser.add_argument('--mtp_no_stress', action='store_false',
                        help='MTP计算器不计算应力（默认不启用）')
    parser.add_argument('--stress_weight', type=float, default=1.0,
                        help='MTP应力权重(默认值1.0)')

    args = parser.parse_args()

    # 读取结构
    print(f"读取结构文件: {args.input_file}")
    atoms = read(args.input_file)

    # 设置计算器
    print(f"使用 {args.calculator.upper()} 计算器")

    if args.calculator == 'vasp':
        os.environ['vasp_pp'] = '/share/home/xill/hj/app/vasp_pp'
        os.environ['num_cores'] = '64'
        os.environ['vasp_std'] = 'vasp_std'
        os.environ['vasp_out_path'] = 'vasp'

        calc = CalculatorFactory.create_vasp_calculator(
            vasp_pp=os.environ['vasp_pp'],
            num_cores=os.environ['num_cores'],
            vasp_std=os.environ['vasp_std'],
            out_path=os.environ['vasp_out_path']
        )
    elif args.calculator == 'abacus':
        os.environ['out_dir'] = 'abacus'
        os.environ['abacus'] = "/share/home/xill/hyx/abacus-3.6.5/bin/abacus"
        os.environ['mpi'] = '64'

        calc = CalculatorFactory.create_abacus_calculator(
            abacus=os.environ['abacus'],
            mpi_cores=int(os.environ['mpi']),
            out_dir=os.environ['out_dir']
        )
    elif args.calculator == 'sus2':
        print(f"使用MTP势函数: {args.mtp_potential}")
        if not os.path.exists(args.mtp_potential):
            raise ValueError(f'势函数文件({args.mtp_potential})不存在')
        if args.ele_list:
            print(f"元素: {args.ele_list}")

        calc = CalculatorFactory.create_mtp_calculator(
            potential_file=args.mtp_potential,
            ele_list=args.ele_list,
            compute_stress=args.mtp_no_stress,
            stress_weight=args.stress_weight
        )

    # 创建优化器，不传递任何特殊参数
    optimizer = StructureOptimizer(
        calculator=calc,
        fmax=args.fmax,
        max_steps=args.max_steps,
        maxstep=args.maxstep,
        optimizer=args.optimizer,
        optimizer_params={}  # 空字典，不使用任何特殊参数
    )

    # 执行优化
    print(f"开始结构优化...")
    print(f"维度: {args.dim}D")
    print(f"压强: {args.pressure} GPa")
    print(f"优化器: {args.optimizer}")
    print(f"收敛标准: {args.fmax} eV/Å")
    print(f"最大步数: {args.max_steps}")
    print(f"使用ASE默认参数")

    results = optimizer.optimize(
        atoms=atoms,
        pressure=args.pressure,
        dim=args.dim,
        constant_volume=args.constant_volume,
        out_process=args.out_process,
        fix_cell=args.fix_cell,
        output_file_base=f'{args.output}'
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("优化结果摘要:")
    print(f"  优化器: {results['optimizer_type']}")
    print(f"  是否收敛: {results['converged']}")
    print(f"  迭代步数: {results['nsteps']}")
    print(f"  fix_cell: {args.fix_cell}")
    print(f"  总能: {results['energy']:.6f} eV")
    print(f"  最大力: {results['max_force']:.6f} eV/Å (目标: {args.fmax} eV/Å)")
    print(f"  体积: {results['volume']:.3f} Å³")
    print(f"  总时间: {results['total_time']:.2f} 秒")
    print(f"  优化时间: {results['optimization_time']:.2f} 秒")

    print("\n输出文件:")
    for key, value in results['output_files'].items():
        if value:
            print(f"  {key}: {value}")

    # 如果未收敛，显示详细信息
    if not results['converged']:
        print("\n" + "!" * 60)
        print("警告: 结构优化未达到收敛标准!")
        print(f"  当前最大力: {results['max_force']:.6f} eV/Å")
        print(f"  收敛标准: {args.fmax} eV/Å")
        print(f"  已迭代步数: {results['nsteps']}")
        print(f"  最大允许步数: {args.max_steps}")
        print("  已保存未收敛结构，请检查结果或调整参数重新计算")
        print("!" * 60)

    if args.out_process:
        # 保存详细结果到CSV
        convergence_status = "converged" if results['converged'] else "unconverged"

        df = pd.DataFrame([{
            'structure': args.input_file,
            'calculator': args.calculator,
            'optimizer': args.optimizer,
            'convergence': convergence_status,
            'converged': results['converged'],
            'energy_eV': results['energy'],
            'max_force_eV_per_ang': results['max_force'],
            'target_fmax': args.fmax,
            'nsteps': results['nsteps'],
            'fix_cell': args.fix_cell,
            'max_steps': args.max_steps,
            'volume_ang3': results['volume'],
            'total_time_s': results['total_time'],
            'optimization_time_s': results['optimization_time'],
            'pressure_GPa': args.pressure,
            'dimension': args.dim,
            'output_extxyz': results['output_files']['extxyz'],
            'output_vasp': results['output_files']['vasp'],
            'output_unconverged': results['output_files']['unconverged'] if not results['converged'] else None,
            'mtp_potential': args.mtp_potential if args.calculator == 'sus2' else None,
            'ele_list': args.ele_list if args.calculator == 'sus2' else None,
            'optimizer_params': 'default'  # 标记为使用默认参数
        }])

        csv_filename = f'optimization_results_{convergence_status}_{args.optimizer}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n详细结果已保存到: {csv_filename}")


if __name__ == '__main__':
    main_relax()