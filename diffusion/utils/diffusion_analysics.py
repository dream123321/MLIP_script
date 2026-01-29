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
import math
import statsmodels.api as sm

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


def calculate_conductivity(traj, T, D, specie, dic_ele_valence):
    """Calculate conductivity from diffusion coefficient"""
    atoms = read(traj)

    V = atoms.get_volume()
    N = sum(atom.symbol == f'{specie}' for atom in atoms)

    e = 1.60e-19  # Electron charge (C)
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    unit_conversion = 10  # Unit conversion from SI to mS/cm (S/m)
    A3_to_m3 = 1e-30
    cm2tom2 = 1e-4

    sigma = N * abs(dic_ele_valence[specie]) ** 2 * e ** 2 / (V * A3_to_m3 * kB * T) * (D * cm2tom2) * unit_conversion
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
        # return {specie: 1}, 1
    index = msd_commands[specie]
    # print(f"从{lmpin_file}解析的MSD元素映射: {msd_commands} use:{index}")
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
    # print(grouped)
    return grouped


def process_all_replicates(grouped_files, start_proportion, end_proportion, time_step, dimension, specie,
                           dic_ele_valence):
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
                    sigma = calculate_conductivity(traj_file, T, D, specie, dic_ele_valence)
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


def ols_process(x_data, y_data):
    """执行OLS回归分析，返回统计量和模型对象"""
    # 检查y_data是否为嵌套列表（多个数据点）
    if isinstance(y_data[0], (list, np.ndarray)):
        # 展开嵌套数据
        x_expanded = []
        y_expanded = []
        for x, y_tuple in zip(x_data, y_data):
            for y_val in y_tuple:
                x_expanded.append(x)
                y_expanded.append(y_val)
        y_is_expanded = True
    else:
        # 已经是展开的数据
        x_expanded = x_data
        y_expanded = y_data
        y_is_expanded = False

    # 执行OLS回归
    X = sm.add_constant(x_expanded)
    model_ols = sm.OLS(y_expanded, X).fit()

    # 返回统计量字典和模型对象
    results = {
        'intercept': model_ols.params[0],  # ln(A)
        'slope': model_ols.params[1],  # -Ea/kB
        'adj_r2': model_ols.rsquared_adj,
        'r2': model_ols.rsquared,
        'slope_stderr': model_ols.bse[1],
        'intercept_stderr': model_ols.bse[0],
        'y_is_expanded': y_is_expanded,
        'model': model_ols,
        'x_expanded': x_expanded,
        'y_expanded': y_expanded
    }

    return results


def fit_arrhenius_diffusion_with_uncertainty(temperatures, D_list):
    """
    拟合扩散系数的Arrhenius方程（考虑1/T因子）
    """
    kB = 8.617333262145e-5  # eV/K
    x = [-1 / (kB * T) for T in temperatures]

    log_D = [[math.log(x) for x in sublist] for sublist in D_list]
    log_mean_D = [math.log(sum(sublist) / len(sublist)) for sublist in D_list]

    all = ols_process(x, log_D)
    mean = ols_process(x, log_mean_D)

    return {
        'all_data_model': all,
        'mean_data_model': mean,
    }


def fit_arrhenius_conductivity_with_uncertainty(temperatures, sigma_list):
    """
    拟合离子电导率的Arrhenius方程（包含1/T因子）
    """
    kB = 8.617333262145e-5  # eV/K
    x = [-1 / (kB * T) for T in temperatures]

    # 计算σT（单位：mS·K/cm）
    log_sigmaT = [[math.log(x * temperatures[i]) for x in sublist] for i, sublist in enumerate(sigma_list)]
    log_sigmaT_mean = [math.log(sum(sublist) / len(sublist) * temperatures[i]) for i, sublist in enumerate(sigma_list)]

    all = ols_process(x, log_sigmaT)
    mean = ols_process(x, log_sigmaT_mean)

    return {
        'all_data_model': all,
        'mean_data_model': mean,
    }


def calculate_room_temp_diffusivity_with_uncertainty(D_fit_result, T_pred=298):
    """
    计算室温扩散系数及其不确定度
    """

    def model_predict(model, x_pred):
        # 预测log(D) - 正确创建带有常数项的预测矩阵
        X_pred = sm.add_constant([x_pred], has_constant='add')
        log_D_pred = model.predict(X_pred)[0]

        # 计算预测区间
        prediction = model.get_prediction(X_pred)
        pred_interval = prediction.conf_int(alpha=0.05)

        # 转换回原始尺度
        D_pred = np.exp(log_D_pred)
        D_interval_lower = np.exp(pred_interval[0][0])
        D_interval_upper = np.exp(pred_interval[0][1])

        return D_pred, D_interval_lower, D_interval_upper

    kB = 8.617333262145e-5
    # 计算x值
    x_pred = -1 / np.array(T_pred) * 1 / kB

    # 获取模型对象
    model_all = D_fit_result['all_data_model']['model']
    model_mean = D_fit_result['mean_data_model']['model']
    all_pred, all_lower, all_upper = model_predict(model_all, x_pred)
    mean_pred, mean_lower, mean_upper = model_predict(model_mean, x_pred)

    return all_pred, all_lower, all_upper, mean_pred, mean_lower, mean_upper


def calculate_room_temp_conductivity_from_D(D_all, V, N, specie, dic_ele_valence, T_pred=298):
    """
    从室温扩散系数计算室温离子电导率（使用Nernst-Einstein方程）
    """

    def D2sigma(D, V, N, specie, T_pred):
        e = 1.602176634e-19
        kB = 1.380649e-23
        A3_to_m3 = 1e-30
        cm2_to_m2 = 1e-4
        unit_conversion = 10

        D_SI = D * cm2_to_m2
        V_SI = V * A3_to_m3

        sigma_SI = (N * abs(dic_ele_valence[specie]) ** 2 * e ** 2 / (V_SI * kB * T_pred)) * D_SI

        sigma = sigma_SI * unit_conversion
        return sigma

    temp = []
    for D in D_all:
        sigma = D2sigma(D, V, N, specie, T_pred)
        temp.append(sigma)
    all_pred, all_lower, all_upper, mean_pred, mean_lower, mean_upper = temp[0], temp[1], temp[2], temp[3], temp[4], \
                                                                        temp[5]

    return all_pred, all_lower, all_upper, mean_pred, mean_lower, mean_upper


def calculate_room_temp_conductivity_from_fit(sigma_fit_result, T_pred=298):
    """
    从电导率Arrhenius拟合计算室温离子电导率
    """

    def model_predict(model, T_val):
        kB = 8.617333262145e-5
        x_pred = -1 / (kB * T_val)

        # 预测ln(σT)
        X_pred = sm.add_constant([x_pred], has_constant='add')
        log_sigmaT_pred = model.predict(X_pred)[0]

        # 计算预测区间
        prediction = model.get_prediction(X_pred)
        pred_interval = prediction.conf_int(alpha=0.05)

        # 转换为σ = (σT)/T
        sigmaT_pred = np.exp(log_sigmaT_pred)
        sigma_pred = sigmaT_pred / T_val

        # 计算σ的置信区间
        sigmaT_lower = np.exp(pred_interval[0][0])
        sigmaT_upper = np.exp(pred_interval[0][1])
        sigma_lower = sigmaT_lower / T_val
        sigma_upper = sigmaT_upper / T_val

        return sigma_pred, sigma_lower, sigma_upper

    # 获取模型对象
    model_all = sigma_fit_result['all_data_model']['model']
    model_mean = sigma_fit_result['mean_data_model']['model']

    # 预测
    sigma_all, sigma_all_lower, sigma_all_upper = model_predict(model_all, T_pred)
    sigma_mean, sigma_mean_lower, sigma_mean_upper = model_predict(model_mean, T_pred)

    return sigma_all, sigma_all_lower, sigma_all_upper, sigma_mean, sigma_mean_lower, sigma_mean_upper


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


def create_single_temperature_msd_plot(all_results, target_temp, out_path, start_proportion, end_proportion,
                                       time_step=1e-15, plot_save=True, plot_show=False, figsize=(20, 10)):
    """
    创建指定温度的MSD图
    包含该温度下所有重复的MSD曲线
    """
    # 查找目标温度的数据
    target_result = None
    for result in all_results:
        if abs(float(result['temperature(K)']) - float(target_temp)) < 1e-6:
            target_result = result
            break

    if not target_result:
        print(f"Warning: No data found for temperature {target_temp}K")
        return

    temp_str = target_result.get('temperature_str', str(target_temp))
    n_replicates = target_result['n_replicates']
    replicate_numbers = target_result['replicate_numbers']

    plt.figure(figsize=figsize)

    # 使用不同的颜色和线型区分不同重复
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_replicates))
    line_styles = ['-', '--', '-.', ':'] * 3

    for idx, replicate_num in enumerate(replicate_numbers):
        msd_file = find_msd_file_by_replicate(out_path, temp_str, replicate_num)

        if msd_file and os.path.exists(msd_file):
            try:
                data = np.loadtxt(msd_file, skiprows=2)
                timesteps = data[:, 0] * time_step * 1e12  # 转换为ps
                msd = data[:, 1]

                # 计算扩散系数（使用相同的截断参数）
                D, r_value, times, msd_calc, start_idx, end_idx = diffusion(
                    msd_file, start_proportion, end_proportion, time_step, dimension=3)

                # 绘制完整曲线
                plt.plot(timesteps, msd,
                         color=colors[idx],
                         linestyle=line_styles[idx % len(line_styles)],
                         linewidth=2,
                         alpha=0.7,
                         label=f'Rep {replicate_num}: D={D:.2e} cm²/s, R²={r_value ** 2:.3f}')

                # 标记拟合区域
                timesteps_fit = timesteps[start_idx:end_idx]
                msd_fit = msd[start_idx:end_idx]
                plt.plot(timesteps_fit, msd_fit,
                         color=colors[idx],
                         linewidth=3,
                         alpha=0.9)

                # 标记起始点和终点
                plt.scatter(timesteps_fit[0], msd_fit[0],
                            color=colors[idx],
                            s=80,
                            marker='o',
                            zorder=5)
                plt.scatter(timesteps_fit[-1], msd_fit[-1],
                            color=colors[idx],
                            s=80,
                            marker='s',
                            zorder=5)

            except Exception as e:
                print(f"Error processing {msd_file} for temperature {target_temp}K: {str(e)}")
                continue

    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title(f'MSD vs Time - {target_temp}K\n'
              f'Fit range: discard {start_proportion * 100:.1f}% start, {end_proportion * 100:.1f}% end\n'
              f'({n_replicates} replicates)',
              )

    # 添加统计信息到图例
    D_avg = target_result['D_avg(cm2/s)']
    D_std = target_result['D_std(cm2/s)']
    R2_avg = target_result['R2_avg']

    plt.legend(loc='best', title=f'Average: D={D_avg:.2e}±{D_std:.2e} cm²/s, R²={R2_avg:.3f}')
    plt.tight_layout()

    if plot_save:
        plt.savefig(os.path.join(out_path, f'MSD_{target_temp}K.png'), dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close()


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
            msd_file = find_msd_file_by_replicate(out_path, temp_str, replicate_num)

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
                                dic_ele_valence=None,
                                plot_temp_and_diffusion_bool=True,
                                plot_save=True, plot_show=False, time_step=1e-15,
                                dimension=3, figsize=(20, 10), target_temps=None):
    """Calculate diffusion coefficients and plot with average values"""

    # Process all replicates with truncation parameters
    if dic_ele_valence is None:
        dic_ele_valence = {'Li': 1, 'Na': 1}
    all_results = process_all_replicates(grouped_files, start_proportion, end_proportion, time_step, dimension, specie,
                                         dic_ele_valence)

    # Create DataFrame for CSV
    df = create_dataframe_for_csv(all_results)

    # 1. 绘制MSD曲线图 - 按重复编号分组
    if plot_temp_and_diffusion_bool and all_results:
        create_msd_plots_by_replicate(all_results, out_path, start_proportion, end_proportion,
                                      time_step, plot_save=plot_save, plot_show=plot_show,
                                      figsize=figsize)

    # 2. 绘制指定温度的MSD图 - 同一温度下所有重复的数据
    if plot_temp_and_diffusion_bool and all_results and target_temps:
        for target_temp in target_temps:
            try:
                create_single_temperature_msd_plot(all_results, target_temp, out_path,
                                                   start_proportion, end_proportion,
                                                   time_step, plot_save=plot_save,
                                                   plot_show=plot_show, figsize=figsize)
            except:
                pass

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
                   start_proportion=0.3, end_proportion=0.0, dic_ele_valence=None, target_temps=None):
    """Main function to process multiple replicates"""

    # Find all MSD files
    if dic_ele_valence is None:
        dic_ele_valence = {'Li': 1, 'Na': 1}
    if target_temps is None:
        target_temps = [300,400]
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
        grouped_files, start_proportion, end_proportion, out_path, specie, dic_ele_valence,
        plot_temp_and_diffusion_bool=True,
        plot_save=True, plot_show=False, time_step=time_step,
        dimension=3, figsize=(20, 10),
        target_temps=target_temps  # 传递目标温度列表
    )

    # Save results to CSV
    csv_file = os.path.join(out_path, 'diffusion_results.csv')
    df.to_csv(csv_file, index=False)

    # Calculate activation energies using average values
    if len(all_results) >= 2:
        temps_list = [r['temperature(K)'] for r in all_results]
        D_list = [r['D_list'] for r in all_results]
        sigma_list = [r['sigma_list'] for r in all_results]

        # 拟合扩散系数的Arrhenius方程
        D_fit_result = fit_arrhenius_diffusion_with_uncertainty(temps_list, D_list)

        print(
            f"Diffusion Activation Energy: {D_fit_result['all_data_model']['slope']:.6f} ± {D_fit_result['all_data_model']['slope_stderr']:.6f} eV "
            f"R²: {D_fit_result['all_data_model']['adj_r2']:.6f}")

        # 拟合电导率的Arrhenius方程
        sigma_fit_result = fit_arrhenius_conductivity_with_uncertainty(temps_list, sigma_list)

        # 计算室温扩散系数
        D_all = all_pred, all_lower, all_upper, mean_pred, mean_lower, mean_upper = calculate_room_temp_diffusivity_with_uncertainty(
            D_fit_result, T_pred=room_temp)
        print(f"Diffusion coefficient at {room_temp}K: {all_pred:.3e} cm²/s, [{all_lower:.3e},{all_upper:.3e}] cm²/s")
        # 获取最低温度的体积和原子数
        V_lowest, N_lowest, min_temp = get_lowest_temperature_volume(all_results, input_dir, specie)

        # 方法1：从扩散系数计算室温离子电导率
        sigma_NE = calculate_room_temp_conductivity_from_D(D_all, V_lowest, N_lowest, specie, T_pred=room_temp,
                                                           dic_ele_valence=dic_ele_valence)
        print(
            f"Conductivity at {room_temp}K (from Nernst-Einstein): {sigma_NE[0]:.3f} mS/cm, [{sigma_NE[1]:.3f},{sigma_NE[2]:.3f}] mS/cm")

        # 方法2：从电导率Arrhenius拟合计算
        sigma_all, sigma_all_lower, sigma_all_upper, sigma_mean, sigma_mean_lower, sigma_mean_upper = calculate_room_temp_conductivity_from_fit(
            sigma_fit_result, T_pred=room_temp)
        print(
            f"Conductivity at {room_temp}K (from conductivity Arrhenius fit): {sigma_all:.3f} mS/cm, [{sigma_all_lower:.3f},{sigma_all_upper:.3f}] mS/cm")

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
            f.write(
                f"The OLS (Ordinary Least Squares) regression results denoted with an asterisk (***) is derived from all individual measurements. (Perhaps more accurate!)\n"
                f"The results without an asterisk is derived from mean values calculated from repeated measurements at each temperature.\n"
                f"(Nernst-Einstein from D): It is done by using extrapolation to obtain D at {room_temp}K, and then the Conductivity is directly calculated through the NE relationship.\n\n")

            c_1 = f"""=== Diffusion Activation Energy Results ===
Activation energy(all_data_model): {D_fit_result['all_data_model']['slope']} ± {D_fit_result['all_data_model']['slope_stderr']} eV ***
R²: {D_fit_result['all_data_model']['adj_r2']} ***
Activation energy(mean_data_model): {D_fit_result['mean_data_model']['slope']} ± {D_fit_result['mean_data_model']['slope_stderr']} eV 
R²: {D_fit_result['mean_data_model']['adj_r2']}"""
            #             f"""Pre-exponential factor (In(D0)): {D_fit_result['all_data_model']['intercept']:.6e} ± {D_fit_result['all_data_model']['intercept_stderr']:.6e} cm²/s #all_data_model
            # Pre-exponential factor (In(D0)): {D_fit_result['mean_data_model']['intercept']:.6e} ± {D_fit_result['mean_data_model']['intercept_stderr']:.6e} cm²/s #mean_data_model"""
            f.write(f"{c_1}\n")

            f.write(f"\n=== Room Temperature ({room_temp}K) Extrapolation ===\n")

            c_2 = f"""95% confidence interval:[xxx,xxx]
Diffusion coefficient: {all_pred:.6e} cm²/s, [{all_lower:.6e},{all_upper:.6e}] cm²/s ***
Conductivity (Nernst-Einstein from D): {sigma_NE[0]:.6f} mS/cm, [{sigma_NE[1]:.6f},{sigma_NE[2]:.6f}] mS/cm ***
Conductivity (from conductivity Arrhenius fit): {sigma_all:.6f} mS/cm, [{sigma_all_lower:.6f},{sigma_all_upper:.6f}] mS/cm ***
Diffusion coefficient: {mean_pred:.6e} cm²/s, [{mean_lower:.6e},{mean_upper:.6e}] cm²/s 
Conductivity (Nernst-Einstein from D): {sigma_NE[3]:.6f} mS/cm, [{sigma_NE[4]:.6f},{sigma_NE[5]:.6f}] mS/cm 
Conductivity (from conductivity Arrhenius fit): {sigma_mean:.6f} mS/cm, [{sigma_mean_lower:.6f},{sigma_mean_upper:.6f}] mS/cm 
"""
            f.write(f"{c_2}\n")
            f.write("\n=== Structural Parameters Used ===\n")
            f.write(f"Volume (from {min_temp}K structure): {V_lowest} Å³\n")
            f.write(f"Number of {specie} atoms: {N_lowest}\n")
    else:
        print("\nNot enough temperature points for activation energy calculation (need at least 2)")
        print(f"Number of temperature points: {len(all_results)}")

    return all_results, df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Usage: python script.py <input_directory> [room_temp] [specie] [start_proportion] [end_proportion] [target_temps...]")
        print("Example: python script.py ./data 298 Li 0.3 0.1")
        print("Example: python script.py ./data 298 Li 0.2 0.0  (discard only first 20%)")
        print("Example: python script.py ./data 298 Li 0.0 0.1  (discard only last 10%)")
        print("Example: python script.py ./data 298 Li 0.2 0.1  (discard first 20% and last 10%)")
        print("Example: python script.py ./data 298 Li 0.2 0.1 800 900  (指定绘制800K和900K的MSD图)")
        sys.exit(1)

    input_dir = sys.argv[1]
    room_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 298
    specie = sys.argv[3] if len(sys.argv) > 3 else 'Li'
    start_proportion = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    end_proportion = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    # 获取目标温度列表（从第6个参数开始）
    target_temps = []
    if len(sys.argv) > 6:
        for i in range(6, len(sys.argv)):
            try:
                temp_val = float(sys.argv[i])
                target_temps.append(temp_val)
            except ValueError:
                print(f"Warning: 跳过无效的温度参数: {sys.argv[i]}")

    print(f"Processing directory: {input_dir}")
    print(f"Room temperature: {room_temp} K")
    print(f"Species: {specie}")
    print(f"Fit range: discard {start_proportion * 100:.1f}% from start, {end_proportion * 100:.1f}% from end")
    if target_temps:
        print(f"Target temperatures for MSD plots: {target_temps} K")

    main_analysics(input_dir, room_temp, specie, time_step=1e-15,
                   start_proportion=start_proportion, end_proportion=end_proportion,
                   target_temps=target_temps)  # 传递目标温度列表