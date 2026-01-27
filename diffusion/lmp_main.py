#!/usr/bin/env python3
"""
sus2扩散系数工作流管理工具
功能：生成、提交和分析LAMMPS分子动力学模拟
作者信息:
    ============================================================
    作者: 黄晶
    单位: 南方科技大学
    邮箱: 2760344463@qq.com
    开发时间: 2026.1.27
"""

import os
import sys
import re
import shutil
import json
import argparse
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

from utils.diffusion_analysics import main_analysics as main_diffusion
from utils.diffusion_gen import main_lmp
from ase.io import iread,write,read


def mkdir_vasp(stru_path,work_path):
    os.makedirs(work_path,exist_ok=True)
    t = [f for f in os.listdir(stru_path) if f.endswith('.vasp')]
    cif_files = [f for f in os.listdir(stru_path) if f.endswith('.cif')]
    xyz_files = [f for f in os.listdir(stru_path) if f.endswith('.xyz')]
    for a in t:
        file = os.path.join(stru_path,a)
        dir = os.path.join(work_path,a.replace('.vasp',''))
        os.makedirs(dir,exist_ok=True)
        shutil.copy(file,dir)
    for a in xyz_files:
        for index,b in enumerate(iread(os.path.join(stru_path,a),format='extxyz')):
            dir = os.path.join(work_path, f"{index}_{a.replace('.xyz', '')}")
            os.makedirs(dir,exist_ok=True)
            write(os.path.join(dir,f'{index}.vasp'),b,format='vasp')
    for cif_file in cif_files:
        file = os.path.join(stru_path, cif_file)
        dir = os.path.join(work_path, cif_file.replace('.cif', ''))
        dir_list.append(dir)
        os.makedirs(dir,exist_ok=True)
        write(os.path.join(dir,f"{cif_file.replace('.cif', '')}.vasp"),read(file),format='vasp')


class MDWorkflowManager:
    """
    分子动力学工作流管理器

    负责管理LAMMPS模拟的整个工作流，包括：
    1. 准备输入文件
    2. 提交计算任务
    3. 分析模拟结果
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化工作流管理器

        Args:
            config: 配置参数字典，如果为None则使用默认配置
        """
        # 完整默认配置 - 只包含实际需要的参数
        self.default_config = {
            # 基础配置
            'work_dir': 'work',
            'stru_dir': 'stru', #结构目录支持，.xyz,.cif,.vasp文件
            'mtp_file': 'current.mtp',
            'lmp_exe': '/share/home/xill/hj/hj_app/interface-lammps-mlip-v1.2/lmp_intel_cpu_intelmpi',  # LAMMPS可执行文件

            # 服务器配置
            'server': 'slurm',  # 'lsf', 'slurm'
            'queue': '256G56c',
            'core': 56,
            'ptile': 56,

            # 元素和体系配置
            'analysics_specie': 'Na',
            'elements': ['Na', 'Zn', 'S', 'P','As','Sb','Bi'],  #势函数对应的元素
            'supercell_size': (3, 3, 3),
            # MSD元素配置（可选）
            'msd_elements': None,  # None,  # 计算MSD的元素列表，默认None表示使用analysics_specie, 如果选择['Li', 'S']，MSD文件会输出两列，一列Li,一列S
            'ele_mode': 1,  # 1: 排序(按原子序数的顺序对elements重排序，从小到大), 2: 不排序


            # 模拟参数
            'relax_bool': True, #模拟之前，是否弛豫结构
            'time_step': 0.003,  # 时间步长 (ps)
            'npt_steps': 0,  # NPT步数
            'npt_mode': 'tri',  # NPT模式: 'tri', 'iso', 'aniso'
            'pressure': 1.0,  # 压力 (bar)

            'nvt_steps': 50000,  # NVT步数
            'dump_frequency': 10000,  # 输出轨迹结构频率
            # 模拟统计参数（可选）
            'ave_params': None,  # 统计平均参数，默认None

            # 温度设置
            'temperatures': [1000,800,600,500],  # 温度列表 (K)
            'repeat_num': 3,  # 重复模拟次数，结果求平均


            # 分析参数
            'start_proportion': 0.3,  # 开始比例 丢弃头部的一些数据
            'end_proportion': 0.00,  # 结束比例   丢弃尾部的一些数据
            'extrapolated_temperature': 300,
            'dic_ele_valence': {'Li':1,'Na':1,'S':-2} #计算离子电导率时候，不同元素对应的z值，abs(z)防止负数

        }

        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        # 设置默认的msd_elements
        if self.config['msd_elements'] is None:
            self.config['msd_elements'] = [self.config['analysics_specie']]

        # 设置默认的ave_params
        if self.config['ave_params'] is None:
            self.config['ave_params'] = {
                'Nevery': 100,  # 每100步采样
                'Nrepeat': 1,  # 累积1次平均
                'Nfreq': 100  # 每100步输出
            }

        self.work_path = os.path.join(self.config['work_dir'])

    def list_folders(self, directory: str) -> List[str]:
        """
        列出目录下所有文件夹

        Args:
            directory: 目录路径

        Returns:
            List[str]: 文件夹路径列表
        """
        folders = []
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    folders.append(item_path)
        return folders


    def modify_sus2mlip(self, file_path: str) -> None:
        """
        修改MTP势函数文件中的径向基类型

        Args:
            file_path: MTP文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否需要修改
            old_pattern = 'radial_basis_type = RBChebyshev_sss'
            new_pattern = 'radial_basis_type = RBChebyshev_sss_lmp'

            # 如果已经包含新的格式，不需要修改
            if new_pattern in content:
                # print(f"MTP文件已符合要求，无需修改: {file_path}")
                return

            # 如果包含旧的格式，才进行修改
            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                #print(f"已修改MTP文件: {file_path}")
            else:
                # 两种格式都没有找到，可能是其他格式
                # print(f"MTP文件格式不同，未修改: {file_path}")
                pass

        except Exception as e:
            print(f"修改MTP文件时出错 {file_path}: {e}")

    def prepare_simulation(self) -> None:
        """
        准备模拟：生成输入文件

        功能：
        1. 创建工作目录
        2. 复制MTP势函数文件
        3. 修改MTP文件
        4. 生成LAMMPS输入文件
        """
        print("=" * 60)
        print("准备模拟输入文件...")
        print("=" * 60)

        # 创建工作目录
        mkdir_vasp(self.config['stru_dir'],self.config['work_dir'])

        # 获取工作目录下的文件夹
        try:
            work_folders = os.listdir(self.work_path)
        except FileNotFoundError:
            print(f"工作目录不存在: {self.work_path}")
            return

        for folder in work_folders:
            folder_path = os.path.join(self.work_path, folder)

            # 检查是否已经有完成标记文件
            restart_file = os.path.join(self.work_path, str(folder), 'npt.restart')
            done_file = os.path.join(self.work_path, str(folder), '__ok__')

            # 如果已经有完成标记，跳过
            if os.path.exists(done_file):
                print(f"任务已完成: {folder}，不生成文件")
                continue

            # 检查是否已经有重启文件
            if os.path.exists(restart_file):
                print(f"任务已有重启文件，可能正在运行: {folder}，不生成文件")
                continue


            if not os.path.isdir(folder_path):
                continue

            print(f"处理文件夹: {folder}")

            # 复制MTP文件
            mtp_filename = os.path.basename(self.config['mtp_file'])
            dst_mtp_path = os.path.join(folder_path, mtp_filename)
            try:
                shutil.copy(self.config['mtp_file'], dst_mtp_path)
            except Exception as e:
                print(f"复制MTP文件失败: {e}")
                continue

            # 修改MTP文件
            self.modify_sus2mlip(dst_mtp_path)

            # 切换到工作目录
            original_dir = os.getcwd()
            os.chdir(folder_path)

            # 生成LAMMPS输入文件
            job_name = folder

            # 准备传递给main_lmp的参数
            lmp_params = {
                'ele_': self.config['elements'],
                'size': self.config['supercell_size'],
                'ele_model': self.config['ele_mode'],
                'server': self.config['server'],
                'queue': self.config['queue'],
                'core': self.config['core'],
                'ptile': self.config['ptile'],
                'lmp_exe': self.config['lmp_exe'],
                'job_name': job_name,
                'relax_bool': self.config['relax_bool'],
                't_step': self.config['time_step'],
                'npt_steps': self.config['npt_steps'],
                'npt_mode': self.config['npt_mode'],
                'pressure': self.config['pressure'],
                'nvt_steps': self.config['nvt_steps'],
                'dump_frequency': self.config['dump_frequency'],
                'temp_list': self.config['temperatures'],
                'repeat_num': self.config['repeat_num'],
                'msd_elements': self.config['msd_elements'],
                'ave_params': self.config['ave_params']
            }

            # 调用LAMMPS输入文件生成函数
            try:
                main_lmp(**lmp_params)
                #print(f"成功生成LAMMPS输入文件: {job_name}")
            except Exception as e:
                print(f"生成LAMMPS输入文件失败: {e}")

            # 返回原始目录
            os.chdir(original_dir)

        print("=" * 60)
        print("模拟文件准备完成。")
        print("=" * 60)

    def submit_simulation(self) -> None:
        """
        提交模拟任务到计算集群

        功能：
        检查并提交未完成的计算任务
        """
        print("=" * 60)
        print("提交模拟任务...")
        print("=" * 60)

        if not os.path.exists(self.work_path):
            print(f"工作目录不存在: {self.work_path}")
            return

        submitted_count = 0
        for folder in self.list_folders(self.work_path):
            folder_name = os.path.basename(folder)

            # 检查是否已经有完成标记文件
            restart_file = os.path.join(folder, 'npt.restart')
            done_file = os.path.join(folder, '__ok__')

            # 如果已经有完成标记，跳过
            if os.path.exists(done_file):
                print(f"任务已完成: {folder_name}，不提交任务")
                continue

            # 检查是否已经有重启文件
            if os.path.exists(restart_file):
                print(f"任务已有重启文件，可能正在运行: {folder_name}，不提交任务")
                continue

            # 检查是否有提交脚本
            submit_script = os.path.join(folder, 'bsub.lsf')
            if not os.path.exists(submit_script):
                print(f"没有找到提交脚本: {submit_script}")
                continue

            # 提交任务
            print(f"提交任务: {folder_name}")
            original_dir = os.getcwd()
            os.chdir(folder)

            # 根据服务器类型提交任务
            if self.config['server'] == 'lsf':
                os.system('bsub<bsub.lsf')
                print(f"已提交LSF任务: {folder_name}")
            elif self.config['server'] == 'slurm':
                os.system('sbatch bsub.lsf')
                print(f"已提交Slurm任务: {folder_name}")
            else:
                print(f"未知服务器类型: {self.config['server']}")
                os.chdir(original_dir)
                continue

            submitted_count += 1
            os.chdir(original_dir)

        print(f"\n总计提交了 {submitted_count} 个任务")

    def analyze_results(self) -> None:
        """
        分析模拟结果

        功能：
        计算扩散系数等物理量
        """

        if not os.path.exists(self.work_path):
            print(f"工作目录不存在: {self.work_path}")
            return

        analyzed_count = 0
        for folder in self.list_folders(self.work_path):
            folder_name = os.path.basename(folder)
            print(f"----------{folder_name}----------")

            try:
                # 调用分析函数
                ok_file = os.path.join(folder,'__ok__')
                if not os.path.exists(ok_file):
                    print('还未全部模拟完成！')
                main_diffusion(
                    folder,
                    room_temp=self.config['extrapolated_temperature'],
                    specie=self.config['analysics_specie'],
                    time_step=self.config['time_step'] * 1e-12,
                    start_proportion=self.config['start_proportion'],
                    end_proportion=self.config['end_proportion'],
                    dic_ele_valence=self.config['dic_ele_valence']
                )
                analyzed_count += 1
            except Exception as e:
                print(f"分析出错 {folder_name}: {e}")

        print(f"\n总计分析了 {analyzed_count} 个文件夹")

    def run(self, mode: str) -> None:
        """
        运行指定模式的工作流

        Args:
            mode: 运行模式，可选 'gen', 'cal', 'ans'
        """
        if mode == 'gen':
            self.prepare_simulation()
        elif mode == 'cal':
            self.submit_simulation()
        elif mode == 'ans':
            self.analyze_results()
        else:
            print(f"错误: 未知模式 '{mode}'")
            print("可用模式: gen, cal, ans")
            sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description='sus2扩散系数工作流管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python md_workflow.py gen          # 生成输入文件
  python md_workflow.py cal          # 提交计算任务
  python md_workflow.py ans          # 分析计算结果
  python md_workflow.py all          # 执行工作流(gen,cal)
  
  python md_workflow.py gen -v              # 执行命令的同时，显示当前详细的默认设置
  python md_workflow.py gen -c xxx.json     # 通过json文件，修改默认设置

配置示例 (JSON格式):
  {
    "work_dir": "lammps_work",
    "mtp_file": "potential.mtp", # 合适的sus2势函数 自己提供
    "server": "slurm",
    "queue": "4T64c",
    "core": 64,
    "lmp_exe": "/path_exe/lmp"   # lammps可执行程序位置 自己提供
    "elements": ["Li", "Ti", "O"], # 势函数对应的元素
    "supercell_size": [2, 2, 2],
    "temperatures": [300, 400, 500],
    "nvt_steps": 10000,
    "repeat_num": 3,
    "analysics_specie": 'Li',
    "stru_dir": 'stru', # 结构目录支持，.xyz,.cif,.vasp结构文件 自己提供
  }
或者进入lmp_main.py，然后修改默认设置
        """
    )

    parser.add_argument(
        'mode',
        choices=['gen', 'cal', 'ans', 'all'],
        help='运行模式: gen(生成文件), cal(提交计算), ans(分析结果), all(完整流程)'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        help='配置文件路径（JSON格式）'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    return parser


def load_config(config_file: str) -> Dict:
    """
    从JSON文件加载配置

    Args:
        config_file: 配置文件路径

    Returns:
        Dict: 配置字典
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件不存在: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return {}
    except Exception as e:
        print(f"加载配置文件失败 {config_file}: {e}")
        return {}


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 加载配置
    config = {}
    if args.config:
        config = load_config(args.config)
        if config:
            print(f"已加载配置文件: {args.config}")

    # 创建管理器
    manager = MDWorkflowManager(config)

    # 显示当前配置
    if args.verbose:
        print("\n当前配置:")
        for key, value in manager.config.items():
            print(f"  {key}: {value}")

    # 运行指定模式
    if args.mode == 'all':
        for mode in ['gen', 'cal', 'ans']:
            print(f"\n{'=' * 60}")
            print(f"执行模式: {mode}")
            print(f"{'=' * 60}")
            manager.run(mode)
    else:
        manager.run(args.mode)


if __name__ == '__main__':
    main()