"""
Python调用CST计算天线S参数的完整代码
Complete Python Code for Antenna S-parameter Calculation using CST

作者: 豆包AI助手
版本: 1.0
日期: 2025年10月24日
"""

import os
import sys
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt


class CSTSParameterCalculator:
    """
    CST S参数计算器类
    专门用于计算天线的S参数
    """

    def __init__(self):
        self.cst = None
        self.project = None
        self.com_available = False
        self.connected = False

        self.check_environment()

    def check_environment(self):
        """检查运行环境"""
        print("=" * 60)
        print("CST S参数计算器 - 环境检查")
        print("=" * 60)

        # 检查操作系统
        if not sys.platform.startswith('win'):
            print(f"❌ 操作系统不支持: {sys.platform}")
            print("   仅支持Windows系统")
            return

        print(f"✅ 操作系统: Windows ({sys.platform})")

        # 检查pywin32
        try:
            import win32com.client
            self.com_available = True
            print("✅ pywin32库已安装")
        except ImportError:
            print("❌ pywin32库未安装")
            print("   请安装: pip install pywin32")
            return

        print("✅ 环境检查完成 - 支持CST调用")

    def connect_to_cst(self, max_attempts=3):
        """连接到CST"""
        if not self.com_available:
            print("❌ 无法连接CST - COM接口不可用")
            return False

        print("\n" + "=" * 60)
        print("连接到CST Microwave Studio")
        print("=" * 60)

        try:
            import win32com.client

            for attempt in range(1, max_attempts + 1):
                print(f"尝试连接 (第{attempt}/{max_attempts}次)...")

                try:
                    # 修复版连接方法 - 不设置Visible属性
                    start_time = time.time()

                    # 先尝试Dispatch
                    try:
                        self.cst = win32com.client.Dispatch("CSTStudio.Application")
                        print("  使用Dispatch成功创建CST实例")
                    except:
                        # 如果失败，尝试DispatchEx
                        self.cst = win32com.client.DispatchEx("CSTStudio.Application")
                        print("  使用DispatchEx成功创建CST实例")

                    connect_time = time.time() - start_time
                    print(f"  连接耗时: {connect_time:.2f}秒")

                    if self.cst is not None:
                        self.connected = True
                        print("✅ CST连接成功!")

                        # 尝试获取版本信息
                        try:
                            version = self.cst.Version
                            print(f"  CST版本: {version}")
                        except Exception as e:
                            print(f"  无法获取版本信息: {str(e)}")

                        return True

                except Exception as e:
                    print(f"  连接失败: {str(e)}")
                    if attempt < max_attempts:
                        print("  1秒后重试...")
                        time.sleep(1)

            print(f"❌ 所有{max_attempts}次连接尝试都失败")
            return False

        except Exception as e:
            print(f"❌ CST连接过程中发生错误: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def create_project(self, project_name="antenna_s_parameter"):
        """创建新的CST项目"""
        if not self.connected or not self.cst:
            print("❌ 未连接到CST，无法创建项目")
            return False

        print("\n" + "=" * 60)
        print(f"创建CST项目: {project_name}")
        print("=" * 60)

        try:
            # 创建新项目
            self.project = self.cst.NewProject("MWS")
            print("✅ 新项目创建成功")

            # 保存项目
            project_path = os.path.abspath(f"{project_name}.cst")
            self.project.SaveAs(project_path)
            print(f"✅ 项目保存到: {project_path}")

            return True

        except Exception as e:
            print(f"❌ 创建项目失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def create_microstrip_antenna(self, params=None):
        """创建微带天线模型"""
        if not self.project:
            print("❌ 没有当前项目，无法创建天线模型")
            return False

        print("\n" + "=" * 60)
        print("创建微带天线模型")
        print("=" * 60)

        # 默认参数
        if params is None:
            params = {
                'patch_length': 20.0,  # mm
                'patch_width': 15.0,  # mm
                'substrate_thickness': 1.6,  # mm
                'substrate_epsr': 4.4,  # FR-4介电常数
                'substrate_length': 30.0,  # mm
                'substrate_width': 25.0,  # mm
                'ground_length': 30.0,  # mm
                'ground_width': 25.0,  # mm
                'feed_position': 0.25  # 相对位置
            }

        print(f"天线参数: {params}")

        try:
            modeler = self.project.Modeler
            modeler.Units = "mm"
            print("✅ 模型器初始化成功 (单位: mm)")

            # 1. 创建接地平面
            ground = modeler.CreateRectangle(
                [0, 0, 0],
                [params['ground_length'], params['ground_width'], 0],
                "Ground"
            )
            ground.Material = "copper"
            print("✅ 接地平面创建成功")

            # 2. 创建介质板
            substrate = modeler.CreateBox(
                [0, 0, 0],
                [params['substrate_length'], params['substrate_width'], params['substrate_thickness']],
                "Substrate"
            )

            # 设置介质板材料
            substrate_materials = ["FR4_epoxy", "FR4", "FR-4", "Teflon"]
            material_set = False
            for mat_name in substrate_materials:
                try:
                    substrate.Material = mat_name
                    material_set = True
                    print(f"✅ 介质板材料设置为: {mat_name}")
                    break
                except:
                    continue

            if not material_set:
                print("⚠️ 未找到FR4材料，使用默认介质")
                substrate.Material = "dielectric"

            # 3. 创建辐射贴片
            patch_x = (params['substrate_length'] - params['patch_length']) / 2
            patch_y = (params['substrate_width'] - params['patch_width']) / 2

            patch = modeler.CreateRectangle(
                [patch_x, patch_y, params['substrate_thickness']],
                [params['patch_length'], params['patch_width'], 0],
                "Patch"
            )
            patch.Material = "copper"
            print("✅ 辐射贴片创建成功")

            # 4. 创建馈线
            feed_length = 10
            feed_width = 2
            feed_x = patch_x + params['feed_position'] * params['patch_length']
            feed_y = (params['substrate_width'] - feed_width) / 2

            feed = modeler.CreateRectangle(
                [feed_x, feed_y, params['substrate_thickness']],
                [feed_length, feed_width, 0],
                "Feed"
            )
            feed.Material = "copper"
            print("✅ 馈线创建成功")

            # 5. 创建端口
            port_pos_x = feed_x + feed_length
            try:
                port = self.project.Ports.AddPort(
                    [port_pos_x, params['substrate_width'] / 2, params['substrate_thickness'] / 2],
                    [0, params['substrate_width'], 0],
                    1
                )
                port.Name = "Port1"
                print("✅ 端口创建成功")
            except Exception as e:
                print(f"⚠️ 创建端口时出错: {str(e)}")

            return True

        except Exception as e:
            print(f"❌ 创建天线模型失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def set_simulation_parameters(self, start_freq=2.0, stop_freq=4.0, num_points=200):
        """设置仿真参数"""
        if not self.project:
            print("❌ 没有当前项目，无法设置仿真参数")
            return False

        print("\n" + "=" * 60)
        print(f"设置仿真参数: {start_freq}-{stop_freq}GHz")
        print("=" * 60)

        try:
            solver = self.project.Solver
            solver.FrequencyRange = f"{start_freq}GHz"
            solver.FrequencyRange2 = f"{stop_freq}GHz"
            solver.SweepType = "Linear"
            solver.NumberOfFrequencyPoints = num_points
            solver.AdaptiveMesh = True

            print(f"✅ 仿真参数设置完成:")
            print(f"   频率范围: {start_freq}-{stop_freq} GHz")
            print(f"   采样点数: {num_points}")
            print(f"   扫频类型: 线性")
            print(f"   网格类型: 自适应")

            return True
        except Exception as e:
            print(f"❌ 设置仿真参数失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def run_simulation(self):
        """运行仿真"""
        if not self.project:
            print("❌ 没有当前项目，无法运行仿真")
            return False

        print("\n" + "=" * 60)
        print("开始CST仿真")
        print("=" * 60)

        try:
            solver = self.project.Solver

            # 保存项目
            self.project.Save()
            print("✅ 项目已保存")

            # 运行仿真
            print("▶ 开始仿真计算...")
            start_time = time.time()

            solver.Run()

            end_time = time.time()
            simulation_time = end_time - start_time

            print(f"✅ 仿真完成!")
            print(f"⏱️ 仿真耗时: {simulation_time:.1f}秒")
            print(f"   = {simulation_time / 60:.1f}分钟")

            return True
        except Exception as e:
            print(f"❌ 仿真失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def extract_s_parameters(self):
        """提取S参数"""
        if not self.project:
            print("❌ 没有当前项目，无法提取S参数")
            return None

        print("\n" + "=" * 60)
        print("提取S参数结果")
        print("=" * 60)

        try:
            results = self.project.Results

            # 提取S11参数
            print("正在提取S11参数...")
            s11_data = results.GetSParameterData("S1,1")

            if s11_data is None or len(s11_data) < 2:
                print("❌ 无法获取S11数据")
                return None

            frequencies = np.array(s11_data[0]) / 1e9  # 转换为GHz
            s11_mag = np.array(s11_data[1])
            s11_phase = np.array(s11_data[2])

            print(f"✅ S参数提取完成:")
            print(f"   频率范围: {frequencies.min():.2f}-{frequencies.max():.2f} GHz")
            print(f"   数据点数: {len(frequencies)}")
            print(f"   S11最小值: {s11_mag.min():.2f} dB")

            # 找到谐振频率
            min_s11_idx = np.argmin(s11_mag)
            resonance_freq = frequencies[min_s11_idx]
            print(f"   谐振频率: {resonance_freq:.2f} GHz")

            # 计算带宽 (S11 <= -10dB)
            s11_threshold = -10
            bandwidth_points = frequencies[s11_mag <= s11_threshold]
            if len(bandwidth_points) > 1:
                bandwidth = bandwidth_points[-1] - bandwidth_points[0]
                print(f"   带宽: {bandwidth * 1000:.0f} MHz")
            else:
                bandwidth = 0
                print(f"   带宽: 无法计算 (S11未达到-10dB)")

            return {
                'frequencies': frequencies,
                's11_mag': s11_mag,
                's11_phase': s11_phase,
                'resonance_frequency': resonance_freq,
                'bandwidth': bandwidth * 1000,
                'min_s11': s11_mag.min()
            }

        except Exception as e:
            print(f"❌ 提取S参数失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return None

    def plot_s_parameters(self, s_params, save_path="s_parameters.png"):
        """绘制S参数图表"""
        if s_params is None:
            print("❌ 没有S参数数据，无法绘图")
            return

        print("\n" + "=" * 60)
        print("绘制S参数图表")
        print("=" * 60)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # S11幅度
        ax1.plot(s_params['frequencies'], s_params['s11_mag'], 'b-', linewidth=2, label='S11')
        ax1.axhline(y=-10, color='r', linestyle='--', alpha=0.7, label='-10dB')
        ax1.axvline(x=s_params['resonance_frequency'], color='g', linestyle='--', alpha=0.7,
                    label=f'谐振频率: {s_params["resonance_frequency"]:.2f}GHz')

        ax1.set_xlabel('频率 (GHz)', fontsize=12)
        ax1.set_ylabel('S11 (dB)', fontsize=12)
        ax1.set_title('天线S11参数', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim([-40, 0])

        # 添加关键信息文本
        textstr = f'最小S11: {s_params["min_s11"]:.2f} dB\n' \
                  f'谐振频率: {s_params["resonance_frequency"]:.2f} GHz\n' \
                  f'带宽: {s_params["bandwidth"]:.0f} MHz'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        # S11相位
        ax2.plot(s_params['frequencies'], s_params['s11_phase'], 'g-', linewidth=2, label='S11相位')
        ax2.set_xlabel('频率 (GHz)', fontsize=12)
        ax2.set_ylabel('相位 (度)', fontsize=12)
        ax2.set_title('S11相位特性', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ S参数图表已保存到: {save_path}")

    def save_results(self, s_params, save_path="s_parameters_results.npy"):
        """保存S参数结果"""
        if s_params is None:
            print("❌ 没有S参数数据，无法保存")
            return

        # 保存完整结果
        np.save(save_path, s_params)
        print(f"✅ S参数数据已保存到: {save_path}")

        # 保存简化的文本结果
        with open("s_parameters_summary.txt", "w", encoding='utf-8') as f:
            f.write("天线S参数计算结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"计算时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"频率范围: {s_params['frequencies'].min():.2f} - {s_params['frequencies'].max():.2f} GHz\n")
            f.write(f"数据点数: {len(s_params['frequencies'])}\n")
            f.write("\n关键性能指标:\n")
            f.write(f"谐振频率: {s_params['resonance_frequency']:.2f} GHz\n")
            f.write(f"最小S11: {s_params['min_s11']:.2f} dB\n")
            f.write(f"带宽 (-10dB): {s_params['bandwidth']:.0f} MHz\n")

        print("✅ S参数摘要已保存到: s_parameters_summary.txt")

    def close(self):
        """关闭CST和项目"""
        print("\n" + "=" * 60)
        print("清理资源")
        print("=" * 60)

        try:
            if self.project:
                try:
                    self.project.Close()
                    print("✅ 项目已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭项目时出错: {str(e)}")

            if self.cst:
                try:
                    self.cst.Quit()
                    print("✅ CST已成功关闭")
                except Exception as e:
                    print(f"⚠️ 关闭CST时出错: {str(e)}")
                    print("ℹ️ 请手动关闭CST窗口")

            self.connected = False
            self.cst = None
            self.project = None

        except Exception as e:
            print(f"❌ 清理资源时发生错误: {str(e)}")

    def calculate_s_parameters(self, antenna_params=None, start_freq=2.0, stop_freq=4.0):
        """完整的S参数计算流程"""
        print("=" * 60)
        print("开始天线S参数计算流程")
        print("=" * 60)

        try:
            # 1. 连接CST
            if not self.connect_to_cst():
                return None

            # 2. 创建项目
            if not self.create_project():
                self.close()
                return None

            # 3. 创建天线模型
            if not self.create_microstrip_antenna(antenna_params):
                self.close()
                return None

            # 4. 设置仿真参数
            if not self.set_simulation_parameters(start_freq, stop_freq):
                self.close()
                return None

            # 5. 运行仿真
            if not self.run_simulation():
                self.close()
                return None

            # 6. 提取S参数
            s_params = self.extract_s_parameters()

            # 7. 可视化结果
            if s_params:
                self.plot_s_parameters(s_params)
                self.save_results(s_params)

            # 8. 清理
            self.close()

            return s_params

        except Exception as e:
            print(f"\n❌ S参数计算流程失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            self.close()
            return None


def main():
    """主函数"""
    # 创建S参数计算器实例
    calculator = CSTSParameterCalculator()

    # 检查环境
    if not calculator.com_available:
        print("\n❌ 环境检查失败，无法继续")
        return

    # 自定义天线参数（可选）
    custom_antenna_params = {
        'patch_length': 18.0,  # mm
        'patch_width': 24.0,  # mm
        'substrate_thickness': 1.6,  # mm
        'substrate_epsr': 4.4,  # FR-4
        'substrate_length': 35.0,  # mm
        'substrate_width': 35.0,  # mm
        'ground_length': 35.0,  # mm
        'ground_width': 35.0,  # mm
        'feed_position': 0.3  # 相对位置
    }

    # 计算S参数
    s_parameters = calculator.calculate_s_parameters(
        antenna_params=custom_antenna_params,
        start_freq=2.0,
        stop_freq=3.0
    )

    # 显示结果摘要
    if s_parameters:
        print("\n" + "=" * 60)
        print("S参数计算完成！")
        print("=" * 60)
        print(f"谐振频率: {s_parameters['resonance_frequency']:.2f} GHz")
        print(f"最小S11: {s_parameters['min_s11']:.2f} dB")
        print(f"带宽: {s_parameters['bandwidth']:.0f} MHz")
        print("\n生成的文件:")
        print("- s_parameters.png: S参数图表")
        print("- s_parameters_results.npy: 完整数据")
        print("- s_parameters_summary.txt: 结果摘要")
    else:
        print("\n❌ S参数计算失败")


if __name__ == "__main__":
    main()