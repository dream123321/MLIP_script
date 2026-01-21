import glob
import sys
import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings
from ase.io import read
from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd
from collections import defaultdict
import json
import re

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.titleweight': 'bold',
    'axes.labelsize': 30,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 18,
    'legend.title_fontsize': 18,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 4,
    'xtick.major.width': 4,
    'ytick.major.width': 4,
    'xtick.major.size': 14,
    'ytick.major.size': 14,
    'xtick.minor.width': 4,
    'ytick.minor.width': 4,
    'xtick.minor.size': 7,
    'ytick.minor.size': 7,
})


def calculate_conductivity(traj, T, D, specie):
    """Calculate conductivity from diffusion coefficient"""
    atoms = read(traj)

    V = atoms.get_volume()
    N = sum(atom.symbol == f'{specie}' for atom in atoms)

    e = 1.60e-19  # Electron charge (C)
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    unit_conversion = 10  # Unit conversion from SI to mS/cm (S/m)
    A3_to_m3 = 1e-30
    cm2tom2 = 1e-4

    sigma = N * 1 ** 2 * e ** 2 / (V * A3_to_m3 * kB * T) * (D * cm2tom2) * unit_conversion
    return sigma


def parse_msd_commands_from_lmpin(lmpin_file, specie):
    """
    从lmp.in文件中解析MSD计算命令，生成元素到列的映射字典

    Parameters:
    -----------
    lmpin_file : str
        lmp.in文件路径

    Returns:
    --------
    dict
        元素到MSD列索引的映射字典，格式如 {'Li': 1, 'S': 2}
        (列索引从1开始，与MSD数据文件列号对应)
    """
    msd_commands = {}
    column_index = 1  # 从1开始计数

    if not os.path.exists(lmpin_file):
        print(f"Warning: lmp.in文件不存在: {lmpin_file}，使用默认元素'{specie}'，列1")
        return {specie: 1}

    try:
        with open(lmpin_file, 'r') as f:
            lines = f.readlines()

        # 查找包含"compute"和"msd"的行
        for line in lines:
            line = line.strip()
            # 跳过注释行
            if line.startswith('#') or not line:
                continue

            # 查找compute msd命令
            if line.startswith('compute') and 'msd' in line:
                # 使用正则表达式提取元素
                pattern = r'compute\s+\w+\s+(\w+)\s+msd'
                match = re.search(pattern, line)
                if match:
                    element = match.group(1)

                    # 检查是否已经记录了该元素
                    if element not in msd_commands:
                        msd_commands[element] = column_index
                        column_index += 1
                        # print(f"Found MSD for element {element} at column {msd_commands[element]}")

    except Exception as e:
        print(f"Warning: 解析lmp.in文件失败: {str(e)}，使用默认元素'{specie}'，列1")
        return {specie: 1}, 1

    if not msd_commands:
        print(f"Warning: 在{lmpin_file}中未找到MSD计算命令，使用默认元素'{specie}'，列1")
        return {specie: 1}, 1

    if specie not in msd_commands:
        raise ValueError(f'Warning: specie({specie}) is not in {msd_commands}')
        #return {specie: 1}, 1
    index = msd_commands[specie]
    #print(f"从{lmpin_file}解析的MSD元素映射: {msd_commands} use:{index}")
    return msd_commands, index


def diffusion(input_file, start_proportion=0.0, end_proportion=0.0, time_step=1e-15, dimension=3, column_index=1):
    """
    Calculate diffusion coefficient from MSD data

    Parameters:
    -----------
    input_file : str
        Path to MSD data file
    start_proportion : float
        Proportion of data to discard from the beginning (0.0 to 1.0)
    end_proportion : float
        Proportion of data to discard from the end (0.0 to 1.0)
    time_step : float
        Time step in seconds
    dimension : int
        Dimensionality (1, 2, or 3)

    Returns:
    --------
    D : float
        Diffusion coefficient
    r_value : float
        Correlation coefficient
    timesteps : array
        Time steps
    msd : array
        Mean squared displacement
    start_idx : int
        Start index for fitting
    end_idx : int
        End index for fitting
    """
    try:
        data = np.loadtxt(input_file, skiprows=2)
    except Exception as e:
        raise ValueError(f"Error reading file {input_file}: {str(e)}")

    if data.shape[1] < 2:
        raise ValueError(f"Input file {input_file} must have at least 2 columns")

    # Convert units
    timesteps = data[:, 0] * time_step  # Convert to seconds
    msd = data[:, column_index] * 1e-16  # Convert Å² to cm²

    # Validate inputs
    if not 0 <= start_proportion < 1:
        raise ValueError("start_proportion must be in range [0, 1)")
    if not 0 <= end_proportion < 1:
        raise ValueError("end_proportion must be in range [0, 1)")
    if start_proportion + end_proportion >= 1:
        raise ValueError("start_proportion + end_proportion must be less than 1")
    if dimension not in {1, 2, 3}:
        raise ValueError("Dimension must be 1, 2, or 3")

    n_points = len(timesteps)

    # Calculate start and end indices for fitting
    start_idx = int(start_proportion * n_points)
    end_idx = n_points - int(end_proportion * n_points)

    # Ensure we have enough points for fitting
    if end_idx - start_idx < 3:
        raise ValueError(f"Not enough points for fitting after truncation. "
                         f"Total points: {n_points}, After truncation: {end_idx - start_idx}")

    # Perform linear regression on the truncated data
    t = timesteps[start_idx:end_idx]
    y = msd[start_idx:end_idx]

    slope, intercept, r_value, p_value, std_err = linregress(t, y)

    # Calculate diffusion coefficient
    dimension_factor = {1: 2, 2: 4, 3: 6}[dimension]
    D = slope / dimension_factor

    return D, r_value, timesteps, msd, start_idx, end_idx


def parse_filename(filename):
    """Parse filename to extract temperature and replicate number"""
    basename = os.path.basename(filename)

    if '_' in basename and '.' in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            try:
                temp_str = parts[2]  # 温度值字符串，如 "800" 或 "800.5"
                replicate = int(parts[3].split('.')[0])
                return temp_str, replicate  # 直接返回字符串形式的温度
            except (ValueError, IndexError):
                pass

    try:
        temp_str = basename.split('_')[2].split('.data')[0]  # 直接获取字符串
        return temp_str, 1  # 返回温度字符串
    except (ValueError, IndexError):
        return None, None


def find_corresponding_traj(msd_file):
    """Find corresponding trajectory file for conductivity calculation"""
    basename = os.path.basename(msd_file)
    dirname = os.path.dirname(msd_file)

    if '_' in basename and '.' in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            temp_str = parts[2]
            replicate = parts[3].split('.')[0]

            patterns = [
                f"mtp_{temp_str}_nvt_{replicate}.traj",
                f"mtp_{temp_str}.0_nvt_{replicate}.traj",
                f"mtp_{temp_str}_nvt_{replicate}.0.traj"
            ]

            for pattern in patterns:
                traj_files = glob.glob(os.path.join(dirname, pattern))
                if traj_files:
                    return traj_files[0]

    traj_files = glob.glob(os.path.join(dirname, "*.traj"))
    if traj_files:
        return traj_files[0]
    # print(traj_files)
    return None


def get_volume_and_N_from_traj(traj_file, specie):
    """Get volume and number of specified atoms from trajectory file"""
    atoms = read(traj_file)
    V = atoms.get_volume()
    N = sum(atom.symbol == f'{specie}' for atom in atoms)
    return V, N


def group_files_by_temperature(files):
    """Group MSD files by temperature"""
    grouped = defaultdict(list)

    for file in files:
        temp, replicate = parse_filename(file)
        if temp is not None:
            grouped[temp].append((replicate, file))

    for temp in grouped:
        grouped[temp].sort(key=lambda x: x[0])
    #print(grouped)
    return grouped


def process_all_replicates(grouped_files, start_proportion, end_proportion, time_step, dimension, specie):
    """Process all replicates and return comprehensive data"""
    all_results = []
    lmpin_file = os.path.join(os.path.dirname(list(grouped_files.values())[0][0][1]), 'lmp.in')
    msd_map, column_index = parse_msd_commands_from_lmpin(lmpin_file, specie)

    for temp_str, files in sorted(grouped_files.items(), key=lambda x: float(x[0]), reverse=True):  # 修改：按温度数值排序
        D_list = []
        sigma_list = []
        R2_list = []
        replicate_numbers = []
        volumes = []
        N_counts = []
        # Store detailed fit information for each replicate
        fit_info_list = []

        # 将温度字符串转换为浮点数
        try:
            T = float(temp_str)
        except ValueError:
            print(f"Warning: 无法将温度 '{temp_str}' 转换为浮点数，跳过该温度点")
            continue

        for replicate, msd_file in files:
            try:
                # Calculate diffusion with truncation
                D, r_value, times, msd, start_idx, end_idx = diffusion(
                    msd_file, start_proportion, end_proportion, time_step, dimension, column_index
                )

                n_total = len(times)
                n_fit = end_idx - start_idx
                start_pct = start_idx / n_total * 100
                end_pct = (n_total - end_idx) / n_total * 100

                # Store fit information
                fit_info = {
                    'replicate': replicate,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'n_points_total': n_total,
                    'n_points_fit': n_fit,
                    'start_percentage': start_pct,
                    'end_percentage': end_pct,
                    'fit_range_used': f"{start_pct:.1f}%-{100 - end_pct:.1f}%"
                }
                fit_info_list.append(fit_info)

                # Calculate conductivity
                traj_file = find_corresponding_traj(msd_file)
                if traj_file:
                    # 传递浮点数温度T而不是字符串temp_str
                    sigma = calculate_conductivity(traj_file, T, D, specie)
                    V, N = get_volume_and_N_from_traj(traj_file, specie)
                    volumes.append(V)
                    N_counts.append(N)
                else:
                    sigma = np.nan
                    print(f"Warning: No trajectory file found for {msd_file}")
                    volumes.append(np.nan)
                    N_counts.append(np.nan)

                D_list.append(D)
                sigma_list.append(sigma)
                R2_list.append(r_value ** 2)
                replicate_numbers.append(replicate)

            except Exception as e:
                print(f"Error processing {msd_file}: {str(e)}")
                continue

        if D_list:
            D_avg = np.mean(D_list)
            D_std = np.std(D_list, ddof=1) if len(D_list) > 1 else 0

            sigma_avg = np.mean(sigma_list)
            sigma_std = np.std(sigma_list, ddof=1) if len(sigma_list) > 1 else 0

            R2_avg = np.mean(R2_list)
            R2_std = np.std(R2_list, ddof=1) if len(R2_list) > 1 else 0

            V_avg = np.nanmean(volumes) if volumes else np.nan
            N_avg = np.nanmean(N_counts) if N_counts else np.nan

            # Calculate average fit information
            if fit_info_list:
                avg_fit_info = {
                    'avg_start_percentage': np.mean([info['start_percentage'] for info in fit_info_list]),
                    'avg_end_percentage': np.mean([info['end_percentage'] for info in fit_info_list]),
                    'avg_n_points_fit': np.mean([info['n_points_fit'] for info in fit_info_list]),
                    'total_replicates': len(fit_info_list)
                }
            else:
                avg_fit_info = {}

            all_results.append({
                'temperature(K)': T,  # 存储浮点数温度
                'temperature_str': temp_str,  # 保留字符串形式用于其他用途
                'n_replicates': len(D_list),
                'D_avg(cm2/s)': D_avg,
                'D_std(cm2/s)': D_std,
                'sigma_avg(mS/cm)': sigma_avg,
                'sigma_std(mS/cm)': sigma_std,
                'R2_avg': R2_avg,
                'R2_std': R2_std,
                'D_list': D_list,
                'sigma_list': sigma_list,
                'R2_list': R2_list,
                'replicate_numbers': replicate_numbers,
                'invT': 1000 / T,
                'volume_avg(A3)': V_avg,
                'N_avg': N_avg,
                'fit_info': avg_fit_info,
                'detailed_fit_info': fit_info_list
            })

    return all_results


def create_dataframe_for_csv(all_results):
    """Create DataFrame for CSV output with all data"""
    rows = []

    for result in all_results:
        # Prepare fit range information string
        fit_range_str = ""
        if result['fit_info']:
            info = result['fit_info']
            fit_range_str = f"Discard: {info.get('avg_start_percentage', 0):.1f}% start, {info.get('avg_end_percentage', 0):.1f}% end"

        # Prepare detailed fit info for each replicate
        detailed_fit_info = []
        if 'detailed_fit_info' in result and result['detailed_fit_info']:
            for fit_info in result['detailed_fit_info']:
                detailed_info = (f"Rep{fit_info['replicate']}: {fit_info['n_points_fit']} points "
                                 f"({fit_info['start_percentage']:.1f}%-{100 - fit_info['end_percentage']:.1f}%)")
                detailed_fit_info.append(detailed_info)

        row = {
            'temperature(K)': result['temperature(K)'],
            'temperature_str': result.get('temperature_str', str(result['temperature(K)'])),
            'n_replicates': result['n_replicates'],
            'D_avg(cm2/s)': result['D_avg(cm2/s)'],
            'D_std(cm2/s)': result['D_std(cm2/s)'],
            'sigma_avg(mS/cm)': result['sigma_avg(mS/cm)'],
            'sigma_std(mS/cm)': result['sigma_std(mS/cm)'],
            'R2_avg': result['R2_avg'],
            'R2_std': result['R2_std'],
            'D_list': json.dumps([x for x in result['D_list']]),
            'sigma_list': json.dumps([x for x in result['sigma_list']]),
            'R2_list': json.dumps([x for x in result['R2_list']]),
            'replicate_numbers': json.dumps(result['replicate_numbers']),
            'invT': result['invT'],
            'volume_avg(A3)': result['volume_avg(A3)'],
            'N_avg': result['N_avg'],
            'fit_range_summary': fit_range_str,
            'fit_range_details': "; ".join(detailed_fit_info) if detailed_fit_info else ""
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def fit_arrhenius_diffusion_with_uncertainty(temperatures, D_values, D_std=None):
    """
    拟合扩散系数的Arrhenius方程（考虑1/T因子）
    """
    T = np.array(temperatures, dtype=float)
    D = np.array(D_values, dtype=float)

    if D_std is not None:
        D_std = np.array(D_std, dtype=float)

    if len(T) != len(D):
        raise ValueError("温度列表和扩散系数列表长度必须相同")
    if len(T) < 2:
        raise ValueError("至少需要2个温度点的数据才能拟合")

    x = 1 / T
    y = np.log(D)

    weights = None
    if D_std is not None and len(D_std) == len(D):
        valid_indices = np.where((D_std > 0) & (D > 0))[0]
        if len(valid_indices) >= 2:
            weights_array = 1.0 / (D_std[valid_indices] / D[valid_indices]) ** 2
            weights = np.ones(len(D))
            weights[valid_indices] = weights_array / np.sum(weights_array) * len(valid_indices)
        else:
            weights = None

    if weights is not None and len(weights[weights > 0]) >= 2:
        mask = weights > 0
        x_valid = x[mask]
        y_valid = y[mask]
        weights_valid = weights[mask]

        x_weighted_mean = np.average(x_valid, weights=weights_valid)
        y_weighted_mean = np.average(y_valid, weights=weights_valid)

        cov_xy = np.average((x_valid - x_weighted_mean) * (y_valid - y_weighted_mean),
                            weights=weights_valid)
        var_x = np.average((x_valid - x_weighted_mean) ** 2, weights=weights_valid)

        slope = cov_xy / var_x if var_x > 0 else 0
        intercept = y_weighted_mean - slope * x_weighted_mean

        y_pred = intercept + slope * x_valid
        r_squared = r2_score(y_valid, y_pred, sample_weight=weights_valid)

        residuals = y_valid - y_pred
        std_err = np.sqrt(np.sum(weights_valid * residuals ** 2) / np.sum(weights_valid))
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

    kB = 8.617333262145e-5
    Ea_D = -slope * kB
    D0 = np.exp(intercept)

    D0_std = D0 * std_err
    Ea_D_std = kB * std_err * np.sqrt(len(x))

    return {
        'activation_energy_eV': Ea_D,
        'activation_energy_std_eV': Ea_D_std,
        'pre_exponential_factor': D0,
        'pre_exponential_factor_std': D0_std,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'standard_error': std_err,
        'temperatures': T.tolist(),
        'D_values': D.tolist(),
        'ln_D_values': y.tolist(),
        'invT': x.tolist()
    }


def fit_arrhenius_conductivity_with_uncertainty(temperatures, sigma_values, sigma_std=None):
    """
    拟合离子电导率的Arrhenius方程（包含1/T因子）
    """
    T = np.array(temperatures, dtype=float)
    sigma = np.array(sigma_values, dtype=float)

    if sigma_std is not None:
        sigma_std = np.array(sigma_std, dtype=float)

    if len(T) != len(sigma):
        raise ValueError("温度列表和电导率列表长度必须相同")
    if len(T) < 2:
        raise ValueError("至少需要2个温度点的数据才能拟合")

    sigmaT = sigma * T
    y = np.log(sigmaT)
    x = 1 / T

    weights = None
    if sigma_std is not None and len(sigma_std) == len(sigma):
        valid_indices = np.where((sigma_std > 0) & (sigma > 0))[0]
        if len(valid_indices) >= 2:
            weights_array = 1.0 / (sigma_std[valid_indices] / sigma[valid_indices]) ** 2
            weights = np.ones(len(sigma))
            weights[valid_indices] = weights_array / np.sum(weights_array) * len(valid_indices)
        else:
            weights = None

    if weights is not None and len(weights[weights > 0]) >= 2:
        mask = weights > 0
        x_valid = x[mask]
        y_valid = y[mask]
        weights_valid = weights[mask]

        x_weighted_mean = np.average(x_valid, weights=weights_valid)
        y_weighted_mean = np.average(y_valid, weights=weights_valid)

        cov_xy = np.average((x_valid - x_weighted_mean) * (y_valid - y_weighted_mean),
                            weights=weights_valid)
        var_x = np.average((x_valid - x_weighted_mean) ** 2, weights=weights_valid)

        slope = cov_xy / var_x if var_x > 0 else 0
        intercept = y_weighted_mean - slope * x_weighted_mean

        y_pred = intercept + slope * x_valid
        r_squared = r2_score(y_valid, y_pred, sample_weight=weights_valid)

        residuals = y_valid - y_pred
        std_err = np.sqrt(np.sum(weights_valid * residuals ** 2) / np.sum(weights_valid))
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

    kB = 8.617333262145e-5
    Ea_sigma = -slope * kB
    sigma0_prime = np.exp(intercept)

    sigma0_prime_std = sigma0_prime * std_err if std_err > 0 else 0
    Ea_sigma_std = kB * std_err * np.sqrt(len(x)) if std_err > 0 else 0

    return {
        'activation_energy_eV': Ea_sigma,
        'activation_energy_std_eV': Ea_sigma_std,
        'sigma0_prime': sigma0_prime,
        'sigma0_prime_std': sigma0_prime_std,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'standard_error': std_err,
        'temperatures': T.tolist(),
        'sigma_values': sigma.tolist(),
        'sigmaT_values': sigmaT.tolist(),
        'ln_sigmaT_values': y.tolist(),
        'invT': x.tolist()
    }


def calculate_room_temp_diffusivity_with_uncertainty(D_fit_result, room_temp=298):
    """
    计算室温扩散系数及其不确定度
    """
    D0 = D_fit_result['pre_exponential_factor']
    D0_std = D_fit_result['pre_exponential_factor_std']
    Ea_D = D_fit_result['activation_energy_eV']
    Ea_D_std = D_fit_result['activation_energy_std_eV']

    kB = 8.617333262145e-5

    exponent = -Ea_D / (kB * room_temp)
    D_room = D0 * np.exp(exponent)

    D_room_std = D_room * np.sqrt((D0_std / D0) ** 2 + (Ea_D_std / (kB * room_temp)) ** 2)

    return D_room, D_room_std


def calculate_room_temp_conductivity_from_D(D_room, D_room_std, V, N, room_temp=298, z=1):
    """
    从室温扩散系数计算室温离子电导率（使用Nernst-Einstein方程）
    """
    e = 1.602176634e-19
    kB = 1.380649e-23
    A3_to_m3 = 1e-30
    cm2_to_m2 = 1e-4
    unit_conversion = 10

    D_room_SI = D_room * cm2_to_m2
    V_SI = V * A3_to_m3

    sigma_SI = (N * z ** 2 * e ** 2 / (V_SI * kB * room_temp)) * D_room_SI

    sigma_room = sigma_SI * unit_conversion
    sigma_room_std = sigma_room * (D_room_std / D_room)

    return sigma_room, sigma_room_std


def calculate_room_temp_conductivity_from_fit(sigma_fit_result, room_temp=298):
    """
    从电导率Arrhenius拟合计算室温离子电导率
    """
    sigma0_prime = sigma_fit_result['sigma0_prime']
    sigma0_prime_std = sigma_fit_result['sigma0_prime_std']
    Ea_sigma = sigma_fit_result['activation_energy_eV']
    Ea_sigma_std = sigma_fit_result['activation_energy_std_eV']

    kB = 8.617333262145e-5

    sigma_room = (sigma0_prime / room_temp) * np.exp(-Ea_sigma / (kB * room_temp))

    sigma_room_std = sigma_room * np.sqrt(
        (sigma0_prime_std / sigma0_prime) ** 2 +
        (Ea_sigma_std / (kB * room_temp)) ** 2
    )

    return sigma_room, sigma_room_std


def get_lowest_temperature_volume(all_results, input_dir, specie):
    """获取最低温度的结构体积和原子数"""
    min_result = min(all_results, key=lambda x: float(x['temperature(K)']))
    min_temp_float = min_result['temperature(K)']  # 浮点数
    min_temp_str = min_result.get('temperature_str', str(min_temp_float))  # 字符串

    for result in all_results:
        if float(result['temperature(K)']) == min_temp_float and result['n_replicates'] > 0:
            pattern = os.path.join(input_dir, f'*{min_temp_str}*nvt*traj')
            traj_files = glob.glob(pattern)
            if traj_files:
                traj_file = traj_files[0]
                V, N = get_volume_and_N_from_traj(traj_file, specie)
                return V, N, min_temp_float

    print("Warning: Could not find trajectory file for lowest temperature, using averages")
    V_avg = np.nanmean([r['volume_avg(A3)'] for r in all_results])
    N_avg = np.nanmean([r['N_avg'] for r in all_results])
    return V_avg, N_avg, min_temp_float


def create_single_replicate_msd_plot(all_results, replicate_num, out_path, start_proportion, end_proportion,
                                     time_step=1e-15, plot_save=True, plot_show=False, figsize=(20, 10)):
    """
    创建单个重复编号的MSD图
    包含该重复编号在所有温度下的MSD曲线
    """
    plt.figure(figsize=figsize)

    temp_data = []
    sorted_results = sorted(all_results, key=lambda x: x['temperature(K)'], reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_results)))

    for idx, result in enumerate(sorted_results):
        temp = result['temperature(K)']
        temp_str = result.get('temperature_str', str(temp))

        replicate_idx = -1
        if 'replicate_numbers' in result and result['replicate_numbers']:
            try:
                replicate_idx = result['replicate_numbers'].index(replicate_num)
            except ValueError:
                continue

        if replicate_idx >= 0:
            msd_file = find_msd_file_by_replicate(out_path, temp_str, replicate_num)  # 使用字符串温度查找文件

            if msd_file and os.path.exists(msd_file):
                try:
                    data = np.loadtxt(msd_file, skiprows=2)
                    timesteps = data[:, 0] * time_step * 1e12
                    msd = data[:, 1]

                    # 计算扩散系数（使用相同的截断参数）
                    D, r_value, times, msd_calc, start_idx, end_idx = diffusion(
                        msd_file, start_proportion, end_proportion, time_step, dimension=3)

                    # 绘制完整曲线
                    plt.plot(timesteps, msd, 'o-',
                             color=colors[idx],
                             markersize=3,
                             linewidth=1,
                             alpha=0.3,
                             label=f'{temp} K (full data)')

                    # 高亮显示用于拟合的部分
                    timesteps_fit = timesteps[start_idx:end_idx]
                    msd_fit = msd[start_idx:end_idx]
                    plt.plot(timesteps_fit, msd_fit, 's-',
                             color=colors[idx],
                             markersize=5,
                             linewidth=2,
                             alpha=0.8,
                             label=f'{temp} K (fit region): D={D:.2e} cm²/s, R²={r_value ** 2:.3f}')

                    # 保存数据
                    temp_data.append({
                        'temperature': temp,
                        'timesteps': timesteps,
                        'msd': msd,
                        'timesteps_fit': timesteps_fit,
                        'msd_fit': msd_fit,
                        'D': D,
                        'R2': r_value ** 2,
                        'color_idx': idx,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })

                except Exception as e:
                    print(f"Error processing {msd_file} for replicate {replicate_num}: {str(e)}")
                    continue

    if not temp_data:
        plt.close()
        return

    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title(
        f'MSD vs Time - Replicate {replicate_num}\n(Fit range: discard {start_proportion * 100:.1f}% start, {end_proportion * 100:.1f}% end)')
    plt.legend(loc='best')
    plt.tight_layout()

    if plot_save:
        plt.savefig(os.path.join(out_path, f'MSD_Replicate_{replicate_num}.png'), dpi=300)
    if plot_show:
        plt.show()
    plt.close()


def find_msd_file_by_replicate(out_path, temperature_str, replicate_num):
    """
    根据温度和重复编号查找MSD文件
    """
    input_dir = os.path.dirname(out_path)

    patterns = [
        f"msd_tracer_{temperature_str}_{replicate_num}.data",
        f"msd_tracer_{temperature_str}.0_{replicate_num}.data",
        f"msd_tracer_{temperature_str}_{replicate_num}.0.data"
    ]

    for pattern in patterns:
        full_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(full_pattern)
        if files:
            return files[0]

    all_files = glob.glob(os.path.join(input_dir, f"msd_tracer_*.data"))
    for file in all_files:
        file_temp, file_replicate = parse_filename(file)
        if file_temp == temperature_str and file_replicate == replicate_num:
            return file

    return None


def create_msd_plots_by_replicate(all_results, out_path, start_proportion, end_proportion,
                                  time_step=1e-15, plot_save=True, plot_show=False, figsize=(20, 10)):
    """
    按重复编号创建MSD曲线图
    所有温度下同一重复编号的数据放在同一个图中
    """
    all_replicate_numbers = set()
    for result in all_results:
        if 'replicate_numbers' in result and result['replicate_numbers']:
            all_replicate_numbers.update(result['replicate_numbers'])

    if not all_replicate_numbers:
        print("Warning: No replicate numbers found in results")
        return

    for replicate_num in sorted(all_replicate_numbers):
        create_single_replicate_msd_plot(all_results, replicate_num, out_path,
                                         start_proportion, end_proportion,
                                         time_step, plot_save=plot_save,
                                         plot_show=plot_show, figsize=figsize)


def list_diffusion_with_average(grouped_files, start_proportion, end_proportion, out_path, specie='Li',
                                plot_temp_and_diffusion_bool=True,
                                plot_save=True, plot_show=False, time_step=1e-15,
                                dimension=3, figsize=(20, 10)):
    """Calculate diffusion coefficients and plot with average values"""

    # Process all replicates with truncation parameters
    all_results = process_all_replicates(grouped_files, start_proportion, end_proportion, time_step, dimension, specie)

    # Create DataFrame for CSV
    df = create_dataframe_for_csv(all_results)

    # 1. 绘制MSD曲线图 - 按重复编号分组
    if plot_temp_and_diffusion_bool and all_results:
        create_msd_plots_by_replicate(all_results, out_path, start_proportion, end_proportion,
                                      time_step, plot_save=plot_save, plot_show=plot_show,
                                      figsize=figsize)

    # Plot Arrhenius plots (using average values)
    if plot_temp_and_diffusion_bool and all_results:
        plot_data_sorted = sorted(all_results, key=lambda x: x['temperature(K)'])

        # Conductivity plot
        plt.figure(figsize=figsize)

        temps = [d['temperature(K)'] for d in plot_data_sorted]
        sigmas = [d['sigma_avg(mS/cm)'] for d in plot_data_sorted]
        sigma_stds = [d['sigma_std(mS/cm)'] for d in plot_data_sorted]

        plt.errorbar([1000 / t for t in temps], sigmas, yerr=sigma_stds,
                     fmt='o-', markersize=8, capsize=5, capthick=2)

        plt.xlabel('1000/T (K⁻¹)')
        plt.ylabel('σ (mS/cm)')
        plt.yscale('log')
        plt.tight_layout()

        if plot_save:
            plt.savefig(os.path.join(out_path, 'conductivity_arrhenius.png'), dpi=300)
        if plot_show:
            plt.show()
        plt.close()

        # Diffusion coefficient plot
        plt.figure(figsize=figsize)

        D_values = [d['D_avg(cm2/s)'] for d in plot_data_sorted]
        D_stds = [d['D_std(cm2/s)'] for d in plot_data_sorted]
        invT = [d['invT'] for d in plot_data_sorted]

        plt.errorbar(invT, D_values, yerr=D_stds,
                     fmt='o-', markersize=8, capsize=5, capthick=2)

        plt.xlabel('1000/T (K⁻¹)')
        plt.ylabel('D (cm²/s)')
        plt.yscale('log')
        plt.tight_layout()

        if plot_save:
            plt.savefig(os.path.join(out_path, 'diffusion_arrhenius.png'), dpi=300)
        if plot_show:
            plt.show()
        plt.close()

    return all_results, df


def main_analysics(input_dir, room_temp=298, specie='Li', time_step=1e-15,
                   start_proportion=0.3, end_proportion=0.0, uncertainty=False):
    """Main function to process multiple replicates"""

    # Find all MSD files
    pattern = os.path.join(input_dir, 'msd_tracer_*.data')
    files = glob.glob(pattern)

    if not files:
        print(f"No MSD files found matching pattern: {pattern}")
        return

    # Group files by temperature
    grouped_files = group_files_by_temperature(files)

    # Create output directory
    out_path = os.path.join(input_dir, 'results')
    os.makedirs(out_path, exist_ok=True)

    # Calculate results with truncation parameters
    all_results, df = list_diffusion_with_average(
        grouped_files, start_proportion, end_proportion, out_path, specie,
        plot_temp_and_diffusion_bool=True,
        plot_save=True, plot_show=False, time_step=time_step,
        dimension=3, figsize=(20, 10),
    )

    # Save results to CSV
    csv_file = os.path.join(out_path, 'diffusion_results.csv')
    df.to_csv(csv_file, index=False)

    # Calculate activation energies using average values
    if len(all_results) >= 2:
        temps_avg = [r['temperature(K)'] for r in all_results]
        D_avg = [r['D_avg(cm2/s)'] for r in all_results]
        sigma_avg = [r['sigma_avg(mS/cm)'] for r in all_results]

        if uncertainty:
            D_std = [r['D_std(cm2/s)'] for r in all_results]
            sigma_std = [r['sigma_std(mS/cm)'] for r in all_results]
        else:
            D_std = None
            sigma_std = None

        # 拟合扩散系数的Arrhenius方程
        D_fit_result = fit_arrhenius_diffusion_with_uncertainty(temps_avg, D_avg, D_std=D_std)

        print(
            f"Diffusion Activation Energy: {D_fit_result['activation_energy_eV']:.3f} ± {D_fit_result['activation_energy_std_eV']:.3f} eV R²: {D_fit_result['r_squared']:.4f}")

        # 拟合电导率的Arrhenius方程
        sigma_fit_result = fit_arrhenius_conductivity_with_uncertainty(temps_avg, sigma_avg, sigma_std=sigma_std)

        # 计算室温扩散系数
        D_room, D_room_std = calculate_room_temp_diffusivity_with_uncertainty(D_fit_result, room_temp=room_temp)
        print(f"Diffusion coefficient at {room_temp}K: {D_room:.3e} ± {D_room_std:.3e} cm²/s")

        # 获取最低温度的体积和原子数
        V_lowest, N_lowest, min_temp = get_lowest_temperature_volume(all_results, input_dir, specie)

        # 方法1：从扩散系数计算室温离子电导率
        sigma_room_NE, sigma_room_NE_std = calculate_room_temp_conductivity_from_D(
            D_room, D_room_std, V_lowest, N_lowest, room_temp=room_temp, z=1
        )
        print(
            f"Conductivity at {room_temp}K (from Nernst-Einstein): {sigma_room_NE:.3e} ± {sigma_room_NE_std:.3e} mS/cm")

        # 方法2：从电导率Arrhenius拟合计算
        sigma_room_fit, sigma_room_fit_std = calculate_room_temp_conductivity_from_fit(
            sigma_fit_result, room_temp=room_temp
        )
        print(
            f"Conductivity at {room_temp}K (from conductivity Arrhenius fit): {sigma_room_fit:.3e} ± {sigma_room_fit_std:.3e} mS/cm")

        # 保存外推结果
        extrapolation_file = os.path.join(out_path, 'extrapolation.txt')
        with open(extrapolation_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"DIFFUSION AND CONDUCTIVITY ANALYSIS RESULTS({specie})\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Fit range configuration:\n")
            f.write(f"  Discard from start: {start_proportion * 100:.1f}%\n")
            f.write(f"  Discard from end: {end_proportion * 100:.1f}%\n")
            f.write(f"  Effective range: {start_proportion * 100:.1f}% to {100 - end_proportion * 100:.1f}%\n\n")

            f.write("=== Diffusion Activation Energy Results ===\n")
            f.write(
                f"Activation energy: {D_fit_result['activation_energy_eV']} ± {D_fit_result['activation_energy_std_eV']} eV\n")
            f.write(
                f"Pre-exponential factor (D0): {D_fit_result['pre_exponential_factor']} ± {D_fit_result['pre_exponential_factor_std']} cm²/s\n")
            f.write(f"R²: {D_fit_result['r_squared']}\n")

            f.write("\n=== Conductivity Activation Energy Results ===\n")
            f.write(
                f"Activation energy: {sigma_fit_result['activation_energy_eV']} ± {sigma_fit_result['activation_energy_std_eV']} eV\n")
            f.write(
                f"σ₀' factor (σ₀' = σ×T pre-factor): {sigma_fit_result['sigma0_prime']} ± {sigma_fit_result['sigma0_prime_std']} mS·K/cm\n")
            f.write(f"R²: {sigma_fit_result['r_squared']}\n")

            f.write(f"\n=== Room Temperature ({room_temp}K) Extrapolation ===\n")
            f.write(f"Diffusion coefficient: {D_room} ± {D_room_std} cm²/s\n")
            f.write(f"Conductivity (Nernst-Einstein from D): {sigma_room_NE} ± {sigma_room_NE_std} mS/cm\n")
            f.write(f"Conductivity (from conductivity Arrhenius fit): {sigma_room_fit} ± {sigma_room_fit_std} mS/cm\n")

            f.write("\n=== Structural Parameters Used ===\n")
            f.write(f"Volume (from {min_temp}K structure): {V_lowest} Å³\n")
            f.write(f"Number of {specie} atoms: {N_lowest}\n")
    else:
        print("\nNot enough temperature points for activation energy calculation (need at least 2)")
        print(f"Number of temperature points: {len(all_results)}")

    return all_results, df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_directory> [room_temp] [specie] [start_proportion] [end_proportion]")
        print("Example: python script.py ./data 298 Li 0.3 0.1")
        print("Example: python script.py ./data 298 Li 0.2 0.0  (discard only first 20%)")
        print("Example: python script.py ./data 298 Li 0.0 0.1  (discard only last 10%)")
        print("Example: python script.py ./data 298 Li 0.2 0.1  (discard first 20% and last 10%)")
        sys.exit(1)

    input_dir = sys.argv[1]
    room_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 298
    specie = sys.argv[3] if len(sys.argv) > 3 else 'Li'
    start_proportion = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    end_proportion = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    print(f"Processing directory: {input_dir}")
    print(f"Room temperature: {room_temp} K")
    print(f"Species: {specie}")
    print(f"Fit range: discard {start_proportion * 100:.1f}% from start, {end_proportion * 100:.1f}% from end")

    main_analysics(input_dir, room_temp, specie, time_step=1e-15,
                   start_proportion=start_proportion, end_proportion=end_proportion)