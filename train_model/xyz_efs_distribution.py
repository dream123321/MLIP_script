from ase.io import iread
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from typing import List, Tuple, Optional


class EFSDistributionAnalyzer:
    """
    从XYZ文件中提取能量、力和应力数据并绘制分布图
    """

    # 单位转换常数
    KBAR2eV_ANG3 = 0.0006241509073
    eV_ANG3_to_GPa = 1 / KBAR2eV_ANG3 / 10

    def __init__(self, xyz_file: str, force_threshold: Optional[float] = None):
        """
        初始化分析器

        参数:
        xyz_file: XYZ文件路径
        force_threshold: 力阈值 (eV/Å)，可选参数
                        - 如果为None，不进行筛选
                        - 如果提供数值，只保留最大原子力模长小于等于该阈值的帧
        """
        self.xyz_file = xyz_file
        self.force_threshold = force_threshold

        # 数据存储
        self.energies = []  # 能量列表 (eV/atom)
        self.forces = []  # 力列表 (eV/Å)
        self.stresses = []  # 应力列表 (GPa)

        # 统计信息
        self.stats = {
            'energy_count': 0,
            'force_count': 0,
            'stress_count': 0,
            'filtered_count': 0,
            'total_frames': 0
        }

        # 提取数据
        self._extract_data()

    def _extract_data(self):
        """从XYZ文件中提取能量、力和应力数据"""
        frames = iread(self.xyz_file)

        for frame in frames:
            self.stats['total_frames'] += 1
            include_frame = True

            # 检查是否需要进行力阈值筛选
            if self.force_threshold is not None:
                try:
                    forces = frame.get_forces()
                    force_magnitudes = np.linalg.norm(forces, axis=1)
                    max_force_magnitude = np.max(force_magnitudes)

                    # 如果最大力模长大于阈值，跳过该帧
                    if max_force_magnitude > self.force_threshold:
                        include_frame = False
                        self.stats['filtered_count'] += 1
                except:
                    pass

            if not include_frame:
                continue

            # 提取能量
            try:
                energy = frame.get_potential_energy() / frame.get_global_number_of_atoms()
                self.energies.append(energy)
                self.stats['energy_count'] += 1
            except:
                pass

            # 提取力
            try:
                forces = frame.get_forces()
                self.forces.extend(forces.flatten())
                self.stats['force_count'] += 1
            except:
                pass

            # 提取应力
            stress_extracted = False

            # 方法1：尝试 frame.get_stress()
            try:
                stress = frame.get_stress() * self.eV_ANG3_to_GPa
                self.stresses.extend(stress.flatten())
                stress_extracted = True
                self.stats['stress_count'] += 1
            except:
                pass

            # 方法2：如果方法1失败，尝试从frame.info中获取virial
            if not stress_extracted:
                try:
                    stress = frame.info['virial'] / frame.get_volume() * self.eV_ANG3_to_GPa
                    self.stresses.extend(stress.flatten())
                    self.stats['stress_count'] += 1
                except:
                    pass

    def get_statistics(self) -> dict:
        """获取数据统计信息"""
        stats = {}

        if self.energies:
            stats['energy'] = {
                'mean': np.mean(self.energies),
                'std': np.std(self.energies),
                'mad': np.mean(np.abs(self.energies - np.mean(self.energies))),
                'min': np.min(self.energies),
                'max': np.max(self.energies),
                'count': len(self.energies)
            }

        if self.forces:
            forces_array = np.array(self.forces)
            stats['force'] = {
                'mean': np.mean(forces_array),
                'std': np.std(forces_array),
                'mad': np.mean(np.abs(forces_array - np.mean(forces_array))),
                'min': np.min(forces_array),
                'max': np.max(forces_array),
                'count': len(forces_array)
            }

        if self.stresses:
            stresses_array = np.array(self.stresses)
            stats['stress'] = {
                'mean': np.mean(stresses_array),
                'std': np.std(stresses_array),
                'mad': np.mean(np.abs(stresses_array - np.mean(stresses_array))),
                'min': np.min(stresses_array),
                'max': np.max(stresses_array),
                'count': len(stresses_array)
            }

        # 添加处理统计
        stats['processing'] = self.stats.copy()

        return stats

    def plot_distribution(self,
                          bins: int = 50,
                          bin_list: Optional[List] = None,
                          figsize: Tuple[int, int] = (30, 10),
                          density: bool = False,
                          output_file: str = 'efs_distribution.jpg',
                          show_fit: bool = False) -> plt.Figure:
        """
        绘制能量、力和应力的分布图

        参数:
        bins: 直方图的分箱数（当bin_list为None时使用）
        bin_list: 可选的列表 [energy_bins, force_bins, stress_bins]
        figsize: 图像大小
        density: 如果为True，绘制概率密度分布；如果为False，绘制频数分布
        output_file: 输出图像文件名
        show_fit: 是否显示正态分布拟合曲线

        返回:
        matplotlib Figure对象
        """
        # 设置绘图样式
        self._setup_plot_style()

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 解析bin_list参数
        if bin_list is not None:
            energy_bins = bin_list[0] if len(bin_list) > 0 else bins
            force_bins = bin_list[1] if len(bin_list) > 1 else bins
            stress_bins = bin_list[2] if len(bin_list) > 2 else bins
        else:
            energy_bins = bins
            force_bins = bins
            stress_bins = bins

        # 获取统计信息用于图例
        stats = self.get_statistics()

        # 绘制能量分布
        if self.energies:
            self._plot_single_distribution(
                axes[0], self.energies, 'Energy (eV/atom)',
                'skyblue', energy_bins, density, stats.get('energy', {}), show_fit
            )

        # 绘制力分布
        if self.forces:
            self._plot_single_distribution(
                axes[1], self.forces, 'Force (eV/Å)',
                'lightgreen', force_bins, density, stats.get('force', {}), show_fit
            )

        # 绘制应力分布
        if self.stresses:
            self._plot_single_distribution(
                axes[2], self.stresses, 'Stress (GPa)',
                'salmon', stress_bins, density, stats.get('stress', {}), show_fit
            )

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


        return fig

    def _plot_single_distribution(self, ax, data, xlabel, color,
                                  bins, density, stats_dict, show_fit):
        """绘制单个分布图"""
        ylabel = 'Probability Density' if density else 'Frequency'

        # 绘制直方图
        counts, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7,
                                              color=color, edgecolor='black',
                                              density=density)

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')

        # 移除标题
        # ax.set_title(title)

        # 添加统计信息到图例
        if stats_dict:
            mean = stats_dict.get('mean', 0)
            mad = stats_dict.get('mad', 0)
            std = stats_dict.get('std', 0)

            # 创建图例文本，包含单位
            if 'Energy' in xlabel:
                unit = 'eV/atom'
            elif 'Force' in xlabel:
                unit = 'eV/Å'
            elif 'Stress' in xlabel:
                unit = 'GPa'
            else:
                unit = ''

        # 如果绘制密度图并启用拟合，添加正态分布拟合
        if density and show_fit and len(data) > 1:
            try:
                from scipy.stats import norm
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 200)

                if stats_dict:
                    p = norm.pdf(x, stats_dict['mean'], stats_dict['std'])
                    ax.plot(x, p, 'r--', linewidth=3, label='Normal fit')

                    # 如果已经添加了统计图例，需要更新图例
                    if 'legend_text' in locals():
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend([handles[0]] + [ax.lines[0]],
                                  [legend_text] + ['Normal fit'],
                                  loc='best', frameon=True, fontsize=20)
                    else:
                        ax.legend(['Normal fit'], loc='best', frameon=True, fontsize=20)
            except ImportError:
                print("Warning: scipy not installed, cannot show normal fit curve.")
            except Exception as e:
                print(f"Warning: Could not plot normal fit: {e}")

    def _setup_plot_style(self):
        """设置绘图样式"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams.update({
            'font.size': 28,
            'axes.titlesize': 28,
            'axes.titleweight': 'bold',
            'axes.labelsize': 28,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 26,
            'ytick.labelsize': 26,
            'legend.fontsize': 20,
            'legend.title_fontsize': 22,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.linewidth': 3,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.major.size': 12,
            'ytick.major.size': 12,
            'xtick.minor.width': 2,
            'ytick.minor.width': 2,
            'xtick.minor.size': 6,
            'ytick.minor.size': 6,
        })

    def print_summary(self):
        """打印数据汇总信息"""
        stats = self.get_statistics()

        print("=" * 70)
        print("EFS DISTRIBUTION ANALYSIS SUMMARY")
        print("=" * 70)

        # 处理统计
        proc_stats = stats['processing']
        print(f"\nProcessing Statistics:")
        print(f"  Total frames processed: {proc_stats['total_frames']}")
        print(f"  Frames filtered by force threshold: {proc_stats['filtered_count']}")

        self.stats = {
            'energy_count': 0,
            'force_count': 0,
            'stress_count': 0,
            'filtered_count': 0,
            'total_frames': 0
        }

        # 能量统计
        if 'energy' in stats:
            e_stats = stats['energy']
            print(f"\nEnergy Statistics (eV/atom):")
            print(f"  Number of structures: {proc_stats['energy_count']}")
            print(f"  Range: [{e_stats['min']:.4f}, {e_stats['max']:.4f}]")
            print(f"  Mean: {e_stats['mean']:.4f}")
            print(f"  Std Dev: {e_stats['std']:.4f}")
            print(f"  Mean Absolute Deviation (MAD): {e_stats['mad']:.4f}")

        # 力统计
        if 'force' in stats:
            f_stats = stats['force']
            print(f"\nForce Statistics (eV/Å):")
            print(f"  Number of structures: {proc_stats['force_count']}")
            print(f"  Range: [{f_stats['min']:.4f}, {f_stats['max']:.4f}]")
            print(f"  Mean: {f_stats['mean']:.4f}")
            print(f"  Std Dev: {f_stats['std']:.4f}")
            print(f"  Mean Absolute Deviation (MAD): {f_stats['mad']:.4f}")

        # 应力统计
        if 'stress' in stats:
            s_stats = stats['stress']
            print(f"\nStress Statistics (GPa):")
            print(f"  Number of structures: {proc_stats['stress_count']}")
            print(f"  Range: [{s_stats['min']:.4f}, {s_stats['max']:.4f}]")
            print(f"  Mean: {s_stats['mean']:.4f}")
            print(f"  Std Dev: {s_stats['std']:.4f}")
            print(f"  Mean Absolute Deviation (MAD): {s_stats['mad']:.4f}")

        print("\n" + "=" * 70)


def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(
        description='Extract and plot energy, force, and stress distributions from XYZ files'
    )

    parser.add_argument('xyz_file', help='Input XYZ file path')
    parser.add_argument('-f', '--force-threshold', type=float, default=None,
                        help='Force threshold in eV/Å (frames with max force > threshold are filtered)')
    parser.add_argument('-b', '--bins', type=int, default=120,
                        help='Number of histogram bins (default: 100)')
    parser.add_argument('-d', '--density', action='store_true',
                        help='Plot probability density instead of frequency')
    parser.add_argument('-o', '--output', default='efs_distribution.jpg',
                        help='Output image file name (default: efs_distribution.jpg)')
    parser.add_argument('--energy-bins', type=int,
                        help='Number of bins for energy histogram (overrides --bins)')
    parser.add_argument('--force-bins', type=int,
                        help='Number of bins for force histogram (overrides --bins)')
    parser.add_argument('--stress-bins', type=int,
                        help='Number of bins for stress histogram (overrides --bins)')
    parser.add_argument('--fit', action='store_true',
                        help='Show normal distribution fit curve (only works with --density)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[30, 10],
                        help='Figure size as width height (default: 30 10)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output image DPI (default: 300)')

    args = parser.parse_args()

    # 创建bin_list
    bin_list = []
    if args.energy_bins or args.force_bins or args.stress_bins:
        bin_list = [
            args.energy_bins if args.energy_bins else args.bins,
            args.force_bins if args.force_bins else args.bins,
            args.stress_bins if args.stress_bins else args.bins
        ]

    # 检查fit参数是否与density一起使用
    if args.fit and not args.density:
        print("Warning: --fit option only works with --density option. Fit curve will not be shown.")
        show_fit = False
    else:
        show_fit = args.fit

    # 创建分析器并执行分析
    analyzer = EFSDistributionAnalyzer(args.xyz_file, args.force_threshold)

    # 打印汇总信息
    analyzer.print_summary()

    # 绘制分布图
    analyzer.plot_distribution(
        bins=args.bins,
        bin_list=bin_list if bin_list else None,
        figsize=tuple(args.figsize),
        density=args.density,
        output_file=args.output,
        show_fit=show_fit
    )

    print(f"\nPlot saved to: {args.output}")


if __name__ == '__main__':
    main()