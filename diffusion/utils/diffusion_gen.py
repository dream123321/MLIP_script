import os
from ase.data import atomic_numbers
from ase.io import read, write
from ase.data import chemical_symbols
import sys


def main_lmp(ele_, size, ele_model, server, queue, core, ptile, lmp_exe, job_name,
                 relax_bool, t_step=0.001, npt_steps=0, npt_mode='tri', pressure=1,
                 nvt_steps=30000, dump_frequency=30, temp_list=None, repeat_num=1,
                 msd_elements=None, ave_params=None):
    """
    Args:
        msd_elements: List of element symbols to calculate MSD for.
                     If None, defaults to ['Li'] if Li exists.
        ave_params: Dictionary with Nevery, Nrepeat, Nfreq values for fix ave/time
                   e.g., {'Nevery': 100, 'Nrepeat': 1, 'Nfreq': 100}
                   If None, uses default values.
    """
    if temp_list is None:
        temp_list = [400.0, 300.0]

    # 默认MSD计算元素：如果指定了msd_elements就用指定的，否则如果存在Li就用Li，否则为空
    if msd_elements is None:
        msd_elements = []

    # 默认ave/time参数
    if ave_params is None:
        ave_params = {'Nevery': 100, 'Nrepeat': 1, 'Nfreq': 100}

    # 提取参数
    nevery = ave_params.get('Nevery', 100)
    nrepeat = ave_params.get('Nrepeat', 1)
    nfreq = ave_params.get('Nfreq', 100)

    #print(f"Using ave/time parameters: Nevery={nevery}, Nrepeat={nrepeat}, Nfreq={nfreq}")

    if os.path.exists('POSCAR'):
        POSCAR = 'POSCAR'
    else:
        vasp_files = [f for f in os.listdir() if f.endswith('.vasp')]
        if vasp_files:
            POSCAR = vasp_files[0]
        else:
            raise FileNotFoundError("No POSCAR or .vasp file found!")

    a = read(POSCAR)
    a = a.repeat(size)

    if ele_model == 1:
        ele = [atomic_numbers[i] for i in ele_]
        ele = sorted(ele)
        specorder = [chemical_symbols[i] for i in ele]
    elif ele_model == 2:
        specorder = ele_
    else:
        raise ValueError('ele_model must be 1 or 2!')

    write('data.in', a, format='lammps-data', masses=True, specorder=specorder, force_skew=True)

    if relax_bool:
        relax = """
min_style cg
minimize 1.0e-8 1.0e-10 5000 100000
write_data relaxed_structure.data   
reset_timestep  0
"""
    else:
        relax = ""

    # 创建元素到类型ID的映射
    dic = {}
    if ele_model == 1:
        ele = [atomic_numbers[i] for i in ele_]
        ele = sorted(ele)
        #print("Sorted atomic numbers:", ele)

        ele_order = {}
        for i in ele_:
            ele_order.update({i: atomic_numbers[i]})
        ele_ = sorted(ele_, key=lambda x: ele_order[x])

        for i in range(len(ele)):
            dic.update({ele[i]: i + 1})
    elif ele_model == 2:
        ele = [atomic_numbers[i] for i in ele_]
        #print("Atomic numbers:", ele)

        for i in range(len(ele)):
            dic.update({ele[i]: i + 1})

    # 获取实际存在的元素
    eles = a.get_chemical_symbols()
    eles = list(set(eles))

    # 创建元素到类型ID的映射（按类型ID排序）
    dic_2 = {}
    nums = []
    for i in eles:
        nums.append(dic[atomic_numbers[i]])

    sorted_indexes = sorted(range(len(nums)), key=lambda k: nums[k])
    for i in sorted_indexes:
        dic_2[eles[i]] = nums[i]

    #print("Element to type ID mapping:", dic_2)

    # 自动确定要计算MSD的元素
    if not msd_elements:
        # 默认包含Li如果存在
        if 'Li' in dic_2.keys():
            msd_elements = ['Li']
            print("No MSD elements specified, defaulting to Li")
        else:
            msd_elements = []
            print("No MSD elements specified and Li not found, no MSD calculation")

    # 只计算实际存在于体系中的元素
    valid_msd_elements = [elem for elem in msd_elements if elem in dic_2.keys()]
    if len(valid_msd_elements) != len(msd_elements):
        missing = set(msd_elements) - set(valid_msd_elements)
        raise ValueError(f"Warning: Some MSD elements not found in system: {missing}")

    #print(f"Calculating MSD for elements: {valid_msd_elements}")

    # 生成LAMMPS输入文件内容
    group = ''
    compute = ''
    c_msd = ''

    # 创建MSD变量到元素符号的映射
    msd_var_to_element = {}
    msd_columns = []  # 存储列信息用于文件头

    # 生成所有元素的group定义
    for key, value in dic_2.items():
        group = group + f'group {key} type {str(value)}\n'

    # 为选定的元素生成MSD计算
    for elem in valid_msd_elements:
        type_id = dic_2[elem]
        compute = compute + f'compute msd{type_id} {elem} msd com yes\n'
        c_msd = c_msd + f'c_msd{type_id}[4] '

        # 记录映射关系
        msd_var_to_element[f'c_msd{type_id}[4]'] = elem
        msd_columns.append(elem)

    # 添加电荷扩散系数计算（如果计算MSD的元素中包含碱金属）
    alkali_metals = ['Li', 'Na', 'K', 'Rb', 'Cs']
    charge_diff_elements = [elem for elem in valid_msd_elements if elem in alkali_metals]

    charge_diff = ''
    for elem in charge_diff_elements:
        type_id = dic_2[elem]
        charge_diff += f"""
# Charge Diffusion Coefficient for {elem}
#variable    N_{elem}     equal   count({elem})           # {elem}原子数量
#compute     disp_{elem}  {elem}  displace/atom          # 计算{elem}原子的位移
#compute     sum_d_{elem} {elem}  reduce sum c_disp_{elem}[1] c_disp_{elem}[2] c_disp_{elem}[3]
#variable    msd_{elem}   equal   "(c_sum_d_{elem}[1]^2 + c_sum_d_{elem}[2]^2 + c_sum_d_{elem}[3]^2)/v_N_{elem}"
#fix         charge_{elem} all ave/time 100 1 100 v_msd_{elem} file msd_charge_{elem}_${{T}}.data
"""

    # 生成thermo_style中要输出的MSD项
    thermo_msd = ''
    for elem in valid_msd_elements:
        type_id = dic_2[elem]
        thermo_msd += f' c_msd{type_id}[4]'

    content_1 = f'''
## 
units       metal
dimension   3
boundary    p p p
atom_style  atomic
box         tilt large
read_data   data.in

neigh_modify    every 1 delay 0 check no

pair_style      mlip mlip.ini
pair_coeff      * *

##################################################################################

#variable        T               equal 900
#variable        eql_step        equal 10000000
#variable        rd_seed         equal 666666
variable        P           equal {pressure}    # 目标压力 (atm)

variable        temp_damp   equal {t_step * 100}  # tau_NVT = timestep * 100  温度松弛时间
variable        press_damp  equal {t_step * 1000} # press_damp = timestep * 1000  压力松弛时间

variable        t_step      equal {t_step}
variable        rlx_step    equal {npt_steps}

variable        thermo_freq equal 20000
variable        dump_freq   equal {dump_frequency}

{group}

thermo_style    custom step pe ke etotal press lx ly lz vol density
thermo          5000

{relax}

velocity        all create ${{T}} ${{rd_seed}} dist gaussian

fix             1 all npt temp ${{T}} ${{T}} ${{temp_damp}} {npt_mode} ${{P}} ${{P}} ${{press_damp}}
timestep        ${{t_step}}
run             ${{rlx_step}}
unfix           1
write_restart   npt.restart

reset_timestep  0

##################################################################################

## nvt-equal
fix             equil all nvt temp ${{T}} ${{T}} ${{temp_damp}}
timestep        ${{t_step}}

# rdf calculation (optional)
#compute         rdf all rdf 100 6 2 2 2 2 3 2 4
#fix             rdf_fix all ave/time 100 1 100 c_rdf[*] file target.rdf mode vector

# msd calculation
{compute}

# Combined MSD for all selected elements
# 使用title命令设置列标题
fix             all_msd all ave/time {nevery} {nrepeat} {nfreq} {c_msd} file msd_tracer_${{T}}_${{file_num}}.data

{charge_diff}

## thermo_style
thermo_style    custom step etotal temp vol pxx pyy pzz lx ly lz {thermo_msd}
thermo          ${{thermo_freq}}
thermo_modify   flush yes
dump            3 all custom ${{dump_freq}} mtp_${{T}}_nvt_${{file_num}}.traj id type element x y z
dump_modify     3 element {' '.join(specorder)}
run             ${{steps}}
write_restart   msd.restart
'''

    with open('lmp.in', 'w') as f:
        f.write(content_1)

    # 写入MLIP配置文件
    content_3 = """mtp-filename           current.mtp
select                  FALSE"""

    def run_temp(temp_list, nvt_steps):
        str_list = [f"  $COMMAND_std -in lmp.in -var T {t} -var steps {nvt_steps} "
                    f"-var rd_seed $RANDOM -var file_num $i > lmp_{t}_$i_$RANDOM.out\n"
                    for t in temp_list]
        return 'do\n' + ''.join(str_list) + 'done\n'

    run_temp_str = run_temp(temp_list, nvt_steps)

    with open('mlip.ini', 'w') as f:
        f.write(content_3)

    # 生成作业提交脚本
    if server == 'lsf':
        content_4 = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {core}
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -R "span[ptile={ptile}]"
hostfile=`echo $LSB_DJOB_HOSTFILE`
NP=`cat $hostfile | wc -l`
#cd $LS_SUBCWD

COMMAND_std="mpirun -n {core} {lmp_exe}"
touch __start__
num={repeat_num}
for i in $(seq 1 $num)
{run_temp_str}
touch __ok__
'''
    elif server == 'slurm':
        content_4 = f'''#!/bin/bash
#SBATCH --job-name {job_name}
#SBATCH --partition {queue}
##SBATCH --nodelist c0[01-40]
#SBATCH --ntasks {core}  #number of core
##SBATCH --qos=840cpu

COMMAND_std="mpirun -n {core} {lmp_exe}"
num={repeat_num}
touch __start__
for i in $(seq 1 $num)
{run_temp_str}
touch __ok__
'''
    else:
        raise ValueError('server no exist!')

    with open('bsub.lsf', 'w') as f:
        f.write(content_4)

    # print(f"\nLAMMPS input file generated successfully.")
    # print(f"MSD will be calculated for: {valid_msd_elements}")
    # print(f"ave/time parameters: Nevery={nevery}, Nrepeat={nrepeat}, Nfreq={nfreq}")
    # print(f"Output file columns: TimeStep {' '.join(valid_msd_elements)}")
    # print(f"Output file: msd_tracer_T_file_num.data")


if __name__ == '__main__':
    # 示例用法
    ele_ = ['Li', 'O', 'Al', 'Si', 'P', 'S', 'Cl', 'Ca', 'Ti', 'V', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Zr', 'Nb', 'Mo', 'In', 'Sn',
            'Sb', 'I', 'Hf', 'Ta', 'W', 'Bi']
    size = (3, 3, 3)
    ele_model = 1
    server = 'slurm'

    # 计算Li和Al的MSD
    msd_elements = ['Li', 'Al']

    # 定义ave/time参数
    ave_params = {
        'Nevery': 100,  # 每100步采样
        'Nrepeat': 1,  # 累积1次平均
        'Nfreq': 100  # 每100步输出
    }

    main_lmp(
        ele_=ele_,
        size=size,
        ele_model=ele_model,
        server=server,
        queue='4T64c',
        core=64,
        ptile=64,
        lmp_exe='/share/home/xill/hj/hj_app/interface-lammps-mlip-11.26/lmp_intel_cpu_intelmpi',
        job_name='test_job',
        relax_bool=True,
        t_step=0.001,
        npt_steps=0,
        nvt_steps=6000,
        msd_elements=msd_elements,
        ave_params=ave_params
    )