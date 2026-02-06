#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from ase.optimize import QuasiNewton, BFGS, LBFGS, FIRE, MDMin, GPMin, LBFGSLineSearch, BFGSLineSearch
from ase.io import read, write,iread
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase import Atoms
from ase.calculators.calculator import Calculator
from typing import Optional, List, Dict, Any, Union
import glob
from tqdm import tqdm
from ase.data import atomic_numbers

# 尝试导入pymlip，如果失败则给出提示
try:
    from pymlip.core import MTPCalactor, PyConfiguration

    PYMLIP_AVAILABLE = True
except ImportError:
    PYMLIP_AVAILABLE = False
    print("警告: pymlip 未安装。请安装 pymlip 以使用此脚本。")
    print("安装方法: pip install pymlip")


class MTPCalculator(Calculator):
    """
    MTP calculator based on ase Calculator
    """
    implemented_properties = ["energy", "forces", "energies", "stress"]

    def __init__(self,
                 potential: str = "p.mtp",
                 ele_list: Optional[List[str]] = None,
                 compute_stress: bool = True,
                 stress_weight: float = 1.0,
                 print_EK: bool = True,
                 **kwargs):
        """
        Args:
            potential (str): xxx.mtp
            ele_list (List[str]): 元素符号列表，例如 ["Al", "O"]
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip 未安装。请先安装 pymlip")

        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = compute_stress
        self.print_EK = print_EK
        self.stress_weight = stress_weight
        self.mtpcalc = MTPCalactor(self.potential)
        self.unique_numbers = [atomic_numbers[ele] for ele in ele_list]

    def calculate(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[list] = None,
            system_changes: Optional[list] = None,
            unique_numbers: Optional[list] = None
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """
        properties = properties or ["energy"]
        system_changes = system_changes or self.all_changes
        super().calculate(atoms=atoms, properties=properties,
                          system_changes=system_changes)

        cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=unique_numbers)
        V = atoms.cell.volume

        self.mtpcalc.calc(cfg)

        self.results['energy'] = np.array(cfg.energy)
        self.results['forces'] = cfg.force
        self.results['energies'] = np.array(cfg.site_energys)

        if self.compute_stress:
            self.results['stress'] = -np.array([
                cfg.stresses[0, 0],
                cfg.stresses[1, 1],
                cfg.stresses[2, 2],
                cfg.stresses[1, 2],
                cfg.stresses[0, 2],
                cfg.stresses[0, 1]
            ]) * self.stress_weight / V


class MTPRelaxer:
    """使用MTP势进行结构弛豫的类"""

    def __init__(self,
                 potential: str,
                 ele_list: Optional[List[str]] = None,
                 optimizer: str = "BFGS",
                 fmax: float = 0.05,
                 steps: int = 500,
                 relax_cell: bool = True,  # 修改为默认True，弛豫晶格
                 cell_filter: str = "exp",
                 pressure: float = 0.0,
                 stress_weight: float = 1.0,
                 logfile: str = "-",
                 trajectory: str = None):

        """
        Args:
            potential: MTP势文件路径
            ele_list: 元素符号列表，例如 ["Al", "O", "H"]
            optimizer: 优化器类型 (BFGS, LBFGS, FIRE, MDMin, GPMin, LBFGSLineSearch, BFGSLineSearch)
            fmax: 最大力收敛标准 (eV/Å)
            steps: 最大优化步数
            relax_cell: 是否弛豫晶胞（默认True）
            cell_filter: 晶胞滤波器类型 (exp, frechet)
            pressure: 外部压力 (GPa)
            stress_weight: 应力权重
            logfile: 日志文件路径 ("-"表示输出到stdout)
            trajectory: 轨迹文件路径
        """
        self.potential = potential
        self.optimizer_type = optimizer
        self.fmax = fmax
        self.max_steps = steps
        self.relax_cell = relax_cell
        self.cell_filter = cell_filter
        self.pressure = pressure
        self.stress_weight = stress_weight
        self.logfile = logfile
        self.trajectory = trajectory
        self.ele_list = ele_list

        # 创建计算器
        self.calculator = MTPCalculator(
            potential=potential,
            ele_list=ele_list,
        )

    def relax_structure(self, atoms: Atoms) -> Atoms:
        """
        弛豫单个结构

        Args:
            atoms: ASE原子对象

        Returns:
            relaxed_atoms: 弛豫后的原子对象
        """
        # 设置计算器
        atoms.calc = self.calculator

        # 选择优化器
        optimizers = {
            "BFGS": BFGS,
            "LBFGS": LBFGS,
            "FIRE": FIRE,
            "MDMin": MDMin,
            "GPMin": GPMin,
            "LBFGSLineSearch": LBFGSLineSearch,
            "BFGSLineSearch": BFGSLineSearch,
            "QuasiNewton": QuasiNewton
        }

        optimizer_class = optimizers.get(self.optimizer_type, BFGSLineSearch)

        # 是否弛豫晶胞
        if self.relax_cell:
            if self.cell_filter.lower() == "frechet":
                atoms = FrechetCellFilter(atoms,scalar_pressure=self.pressure)
            else:
                # 使用ExpCellFilter，可以设置外部压力
                atoms = ExpCellFilter(atoms, scalar_pressure=self.pressure)
        else:
            print("注意: 晶格弛豫已禁用，仅弛豫原子位置")

        # 创建优化器
        dyn = optimizer_class(
            atoms,
            logfile=self.logfile,
            trajectory=self.trajectory
        )

        # 运行优化
        ase_convergence_label = dyn.run(fmax=self.fmax, steps=self.max_steps)
        #print(atoms)
        atoms = atoms.atoms
        s = atoms.get_stress()
        six2nine = np.array([s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]])
        virial = -six2nine * atoms.get_volume()
        atoms.info['virial'] = virial
        atoms.info['ase_convergence_label'] = ase_convergence_label

        return atoms

    def compute_efs(self, stru: Atoms) -> Atoms:
        """
        计算结构的能量、力、应力

        Args:
            stru: ASE原子对象

        Returns:
            atoms: ASE原子对象
        """

        # 设置计算器（计算应力）
        stru.calc = self.calculator
        e = stru.get_potential_energy()
        f = stru.get_forces()
        s = stru.get_stress()

        atoms = Atoms(stru.get_chemical_symbols(), positions=stru.get_positions(), cell=stru.get_cell())

        six2nine = np.array([s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]])

        atoms.info['energy'] = e
        atoms.info['stress'] = six2nine
        virial = -1 * six2nine * stru.get_volume()
        atoms.info['virial'] = virial

        atoms.info['pbc'] = "T T T"
        atoms.arrays['forces'] = f
        atoms.pbc = [True, True, True]

        return atoms


def detect_file_format(filename: str) -> None:
    """
    检测文件格式

    Args:
        filename: 文件名

    Returns:
        format_str: 文件格式字符串
    """
    # 获取文件名（不包含路径）
    basename = os.path.basename(filename)
    ext = os.path.splitext(basename)[1].lower()

    # ASE支持的文件格式映射
    format_map = {
        '.cif': 'cif',
        '.vasp': 'vasp',
        'poscar': 'vasp',
        'contcar': 'vasp',
        '.xyz': 'extxyz',
        '.extxyz': 'extxyz',
        '.pdb': 'proteindatabank',
        '.in': 'espresso-in',
        '.out': 'espresso-out',
        '.json': 'json',
        '.traj': 'traj',
        '.xsf': 'xsf',
        '.cfg': 'cfg',
        '.gen': 'gen',
        '.fdf': 'fdf',
        '.struct': 'struct'
    }

    # 先检查扩展名
    if ext in format_map:
        return format_map[ext]

    # 如果扩展名为空（如 POSCAR、CONTCAR），检查完整文件名的小写形式
    basename_lower = basename.lower()
    if basename_lower in format_map:
        return format_map[basename_lower]

    # 都没有找到，返回 None
    return None


def read_structures(input_file: str, index: str = ":") -> List[Atoms]:
    """
    读取不同格式的结构文件

    Args:
        input_file: 输入文件路径
        index: 索引，用于读取多个结构

    Returns:
        structures: ASE原子对象列表
    """
    # 检测文件格式
    file_format = detect_file_format(input_file)

    if file_format is None:
        # 尝试让ASE自动检测格式
        print(f"警告: 无法自动检测文件格式，尝试使用ASE自动检测: {input_file}")
        file_format = None

    try:
        # 尝试读取文件
        if index == ":":
            # 读取所有结构
            structures = read(input_file, index=index, format=file_format)
        else:
            # 读取单个结构
            structures = [read(input_file, format=file_format)]

        # 确保返回的是列表
        if isinstance(structures, Atoms):
            structures = [structures]

        return structures

    except Exception as e:
        print(f"读取文件时出错: {e}")

        # 尝试其他方法
        try:
            # 尝试使用ase的iread
            structures = list(read(input_file, index=index, format=file_format))
            return structures
        except Exception as e2:
            print(f"尝试其他读取方法也失败: {e2}")
            raise


def save_structures(structures: List[Atoms],
                    output_file: str,
                    format_override: Optional[str] = None) -> None:
    """
    保存结构到文件

    Args:
        structures: ASE原子对象列表
        output_file: 输出文件路径
        format_override: 强制指定输出格式
    """
    # 如果指定了格式，使用指定格式
    if format_override:
        file_format = format_override
        if format_override == 'xyz':
            file_format = 'extxyz'
    else:
        # 根据扩展名自动检测格式
        file_format = detect_file_format(output_file)

    if len(structures) == 1:
        # 单个结构
        write(output_file, structures[0], format=file_format)
    else:
        # 多个结构
        write(output_file, structures, format=file_format)


def process_file(input_file: str,
                 potential: str,
                 ele_list: Optional[List[str]] = None,
                 keep_order: bool = False,
                 mode: str = "predict",
                 optimizer: str = "BFGS",
                 fmax: float = 0.05,
                 steps: int = 500,
                 relax_cell: bool = True,  # 修改为默认True
                 cell_filter: str = "exp",
                 pressure: float = 0.0,
                 stress_weight: float = 1.0,
                 output_file: str = None,
                 output_format: str = None,
                 log_file: str = None):
    """
    处理结构文件

    Args:
        input_file: 输入文件路径
        potential: MTP势文件路径
        ele_list: 元素符号列表，例如 ["Al", "O", "H"]
        keep_order: 是否保持ele_list的顺序（默认False，自动排序）
        mode: 模式 ("predict"或"relax")
        optimizer: 优化器类型
        fmax: 最大力收敛标准
        steps: 最大优化步数
        relax_cell: 是否弛豫晶胞（默认True）
        cell_filter: 晶胞滤波器类型
        pressure: 外部压力 (GPa)
        stress_weight: 应力权重
        output_file: 输出文件路径
        output_format: 输出格式 (如'cif', 'vasp', 'xyz','extxyz'等)
        log_file: 日志文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    if not os.path.exists(potential):
        print(f"错误: 势文件不存在: {potential}")
        sys.exit(1)

    # 检测输入文件格式
    input_format = detect_file_format(input_file)
    if input_format:
        print(f"检测到输入文件格式: {input_format}")

    # 显示元素列表信息
    if ele_list:
        if keep_order:
            print(f"使用指定的元素列表（保持顺序）: {ele_list}")
        else:
            print(f"使用指定的元素列表（自动排序后）: {sorted(ele_list, key=lambda x: atomic_numbers[x])}")
            ele_list = sorted(ele_list, key=lambda x: atomic_numbers[x])
    else:
        temp = set()
        for atoms in iread(input_file,format=input_format):
            symbol = set(atoms.get_chemical_symbols())
            temp = temp | symbol
        ele_list = sorted(temp, key=lambda x: atomic_numbers[x])
        print(f"未指定元素列表，将从文件中自动检测，（按原子序数自动排序后，请检查是否正确!!!）: {ele_list}")

    # 显示弛豫模式信息
    if mode == "relax":
        if relax_cell:
            print("弛豫模式: 原子位置 + 晶格 (默认)")
        else:
            print("弛豫模式: 仅原子位置")

        # 显示优化器信息
        print(f"使用优化器: {optimizer}")


    # 设置输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        if mode == "efs":
            suffix = "_efs"
        else:
            suffix = "_relaxed"

        # 如果没有指定输出格式，使用与输入相同的格式
        if output_format:
            output_file = f"{base_name}{suffix}.{output_format}"
        elif input_format:
            # 使用输入文件的扩展名
            ext_map = {
                'cif': '.cif',
                'vasp': '.vasp',
                'xyz': '.extxyz',
                'extxyz': '.extxyz'
            }
            ext = ext_map.get(input_format, '.xyz')
            if ext == '.extxyz':
                output_file = f"{base_name}{suffix}.xyz"
            else:
                output_file = f"{base_name}{suffix}{ext}"
        else:
            output_file = f"{base_name}{suffix}.xyz"

    # 设置日志
    if log_file is None:
        log_file = os.path.splitext(input_file)[0] + ".log"

    # 读取输入文件
    print(f"读取文件: {input_file}")
    try:
        structures = read_structures(input_file, index=":")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)

    print(f"找到 {len(structures)} 个结构")

    # 创建弛豫器
    relaxer = MTPRelaxer(
        potential=potential,
        ele_list=ele_list,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        relax_cell=relax_cell,
        cell_filter=cell_filter,
        pressure=pressure,
        stress_weight=stress_weight,
        logfile=log_file,
        trajectory=None
    )

    # 处理每个结构
    processed_structures = []

    for i, atoms in enumerate(tqdm(structures, desc="处理结构")):
        if mode == "relax":
            # 弛豫结构
            relaxed_atoms = relaxer.relax_structure(atoms.copy())
            processed_structures.append(relaxed_atoms)
        elif mode == "efs":
            cal_atoms = relaxer.compute_efs(atoms.copy())
            processed_structures.append(cal_atoms)
        else:
            raise ValueError(f'{mode} is not exist!')

    # 写入输出文件
    print(f"\n写入输出文件: {output_file}")
    try:
        save_structures(processed_structures, output_file, output_format)
    except Exception as e:
        print(f"写入文件时出错: {e}")
        print("尝试使用extxyz格式...")
        # 尝试使用extxyz格式
        write(output_file, processed_structures, format='extxyz')

    # 生成统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理的构型数: {len(processed_structures)}")
    if mode == "relax":
        print(f"弛豫类型: {'原子位置 + 晶格' if relax_cell else '仅原子位置'}")
        print(f"优化器: {optimizer}")
        print(f"日志文件: {log_file}")
    print("=" * 60)


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(
        description="使用SUS2势进行结构弛豫和efs预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的文件格式:
  .cif       - CIF格式
  .vasp      - VASP POSCAR/CONTCAR格式
  .extxyz    - 扩展XYZ格式 (这里定义.xyz 等于.extxyz)

注意:
  1. 在relax模式下，默认弛豫原子位置和晶格（relax-cell=True）
  2. 使用--no-relax-cell禁用晶格弛豫，仅弛豫原子位置
  3. 支持多种优化器，包括带线搜索的优化器（LBFGSLineSearch, BFGSLineSearch）
  4. 如果自动检测文件的ele_list，请确保文件内的元素与训练的势函数训练的元素相等，否则，势函数处理文件的时候，识别有误!!!

优化器说明:
  BFGS           - 标准的BFGS优化器（默认）
  LBFGS          - 限制内存的BFGS，适合大体系
  FIRE           - 快速惯性弛豫引擎
  MDMin          - 基于分子动力学的优化器
  GPMin          - 基于高斯过程的优化器
  LBFGSLineSearch - 带线搜索的LBFGS，通常更稳定
  BFGSLineSearch  - 带线搜索的BFGS，通常更稳定
  QuasiNewton    - 准牛顿法

示例:
  %(prog)s structure.cif p.mtp --mode efs
  %(prog)s POSCAR p.mtp --mode relax --optimizer LBFGS --fmax 0.01
  %(prog)s POSCAR p.mtp --mode relax --optimizer LBFGSLineSearch  # 使用线搜索
  %(prog)s structure.vasp p.mtp --mode relax --no-relax-cell  # 仅弛豫原子位置
  %(prog)s structure.vasp p.mtp --mode relax --pressure 5.0  # 在5 GPa压力下弛豫
  %(prog)s "*.cif" p.mtp --mode relax --batch --output-format vasp
  %(prog)s POSCAR p.mtp --ele-list Al O H  # 指定元素顺序
  %(prog)s POSCAR p.mtp --ele-list Al O H --keep-order  # 保持指定顺序
        """
    )

    # 必需参数 - 使用 nargs='*' 来接收所有参数，然后手动分离
    parser.add_argument("args", nargs='*', help="输入文件和势文件")
    parser.add_argument("--ele-list", nargs="+", type=str,
                        help="指定元素符号列表，例如: --ele-list Al O H")
    parser.add_argument("--keep-order", action="store_true",
                        help="保持ele-list中指定的顺序，不自动排序（默认自动排序）")

    # 模式选择
    parser.add_argument("--mode", choices=["efs", "relax"],
                        default="efs", help="运行模式: efs(预测能量、力、应力) 或 relax(弛豫结构)")

    # 弛豫参数
    parser.add_argument("--optimizer",
                        choices=["BFGS", "LBFGS", "FIRE", "MDMin", "GPMin",
                                 "LBFGSLineSearch", "BFGSLineSearch", "QuasiNewton"],
                        default="BFGSLineSearch", help="优化器类型 (默认: BFGSLineSearch)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="最大力收敛标准 (eV/Å, 默认: 0.05)")
    parser.add_argument("--steps", type=int, default=500,
                        help="最大优化步数 (默认: 500)")
    parser.add_argument("--no-relax-cell", dest="relax_cell", action="store_false",
                        help="禁用晶格弛豫，仅弛豫原子位置（默认弛豫原子位置+晶格）")
    parser.add_argument("--cell-filter", choices=["exp", "frechet"],
                        default="exp", help="晶胞滤波器类型 (默认: exp)")
    parser.add_argument("--pressure", type=float, default=0.0,
                        help="外部压力 (GPa, 默认: 0.0)")
    parser.add_argument("--stress-weight", type=float, default=1.0,
                        help="应力权重 (默认: 1.0)")

    # 输出选项
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("--output-format",
                        choices=["cif", "vasp", "xyz", "extxyz", "pdb", "json", "xsf"],
                        help="输出文件格式 (默认根据扩展名自动检测)")
    parser.add_argument("--log-file", help="日志文件路径")

    # 其他选项
    parser.add_argument("--batch", action="store_true",
                        help="批量处理多个文件 (使用通配符)")
    parser.add_argument("--single", action="store_true",
                        help="只读取第一个结构 (对于包含多个结构的文件)")

    # 设置默认值
    parser.set_defaults(relax_cell=True)  # 设置默认值为True

    args = parser.parse_args()

    # 手动分离输入文件和势文件
    if len(args.args) < 2:
        print("错误: 需要至少两个参数: 输入文件和势文件")
        sys.exit(1)

    # 最后一个参数是势文件
    potential = args.args[-1]
    # 其他参数都是输入文件
    input_files = args.args[:-1]

    # 检查pymlip是否可用
    if not PYMLIP_AVAILABLE:
        print("错误: pymlip 未安装。请安装 pymlip: pip install pymlip")
        sys.exit(1)

    # 处理单个或多个文件
    if args.batch or len(input_files) > 1:
        # 批量处理
        print(f"找到 {len(input_files)} 个文件:")
        for f in input_files:
            print(f"  {f}")

        for input_file in input_files:
            print(f"\n{'=' * 60}")
            print(f"处理文件: {input_file}")
            print('=' * 60)

            # 修改读取方式，如果指定了single，只读取第一个结构
            index = 0 if args.single else ":"

            process_file(
                input_file=input_file,
                potential=potential,
                ele_list=args.ele_list,
                keep_order=args.keep_order,
                mode=args.mode,
                optimizer=args.optimizer,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=args.relax_cell,
                cell_filter=args.cell_filter,
                pressure=args.pressure,
                stress_weight=args.stress_weight,
                output_file=args.output,
                output_format=args.output_format,
                log_file=args.log_file
            )
    else:
        # 处理单个文件
        process_file(
            input_file=input_files[0],
            potential=potential,
            ele_list=args.ele_list,
            keep_order=args.keep_order,
            mode=args.mode,
            optimizer=args.optimizer,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=args.relax_cell,
            cell_filter=args.cell_filter,
            pressure=args.pressure,
            stress_weight=args.stress_weight,
            output_file=args.output,
            output_format=args.output_format,
            log_file=args.log_file
        )


if __name__ == "__main__":
    main()