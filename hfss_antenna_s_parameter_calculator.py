"""
基于HFSS的天线S参数计算工具 - 兼容版
HFSS-based Antenna S-parameter Calculation Tool - Compatible Version

支持PyAEDT新旧版本架构
兼容pyaedt和ansys.aedt.core包结构
"""

import os
import sys
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt

class HFSSAntennaSimulator:
    """
    HFSS天线仿真器类
    兼容PyAEDT新旧版本架构
    """

    def __init__(self):
        self.aedt_app = None
        self.hfss = None
        self.project = None
        self.design = None
        self.solution = None

        self.pyedt_module = None
        self.pyedt_available = False
        self.aedt_installed = False
        self.new_api = False  # 是否使用新的ansys.aedt.core API

        self.check_environment()

    def check_environment(self):
        """检查运行环境和PyAEDT版本"""
        print("="*60)
        print("HFSS天线仿真器 - 环境检查")
        print("="*60)

        # 尝试导入PyAEDT（兼容新旧版本）
        try:
            # 先尝试新的API结构
            try:
                import ansys.aedt.core as pyaedt
                self.pyedt_module = pyaedt
                self.new_api = True
                print(f"✅ 检测到新API版本: ansys.aedt.core")
            except ImportError:
                # 再尝试旧的API结构
                import pyaedt
                self.pyedt_module = pyaedt
                self.new_api = False
                print(f"✅ 检测到旧API版本: pyaedt")

            self.pyedt_available = True
            print(f"   PyAEDT版本: {self.pyedt_module.__version__}")

        except ImportError:
            print("❌ PyAEDT库未安装")
            print("   请安装: pip install pyaedt")
            print("   或安装新版本: pip install ansys-aedt")
            return

        # 检查Ansys Electronics Desktop是否可用
        try:
            # 兼容新旧版本的检查方法
            if hasattr(self.pyedt_module, 'is_aedt_installed'):
                aedt_installed = self.pyedt_module.is_aedt_installed()
            elif hasattr(self.pyedt_module, 'desktop') and hasattr(self.pyedt_module.desktop, 'is_aedt_installed'):
                aedt_installed = self.pyedt_module.desktop.is_aedt_installed()
            else:
                aedt_installed = False

            if aedt_installed:
                self.aedt_installed = True
                print("✅ Ansys Electronics Desktop已安装")

                # 显示已安装的AEDT版本
                try:
                    if hasattr(self.pyedt_module, 'get_installed_aedt_versions'):
                        versions = self.pyedt_module.get_installed_aedt_versions()
                    elif hasattr(self.pyedt_module.desktop, 'get_installed_aedt_versions'):
                        versions = self.pyedt_module.desktop.get_installed_aedt_versions()
                    else:
                        versions = []

                    if versions:
                        print(f"   已安装版本: {', '.join(versions)}")
                except:
                    print("   无法获取已安装的AEDT版本")

            else:
                print("❌ Ansys Electronics Desktop未找到")
                print("   请确保已安装Ansys Electronics Desktop 2022 R1或更高版本")

        except Exception as e:
            print(f"⚠️ 检查AEDT安装时出错: {str(e)}")

        print("✅ 环境检查完成")

    def connect_to_hfss(self, version=None, non_graphical=False):
        """连接到HFSS - 兼容新旧API"""
        if not self.pyedt_available or not self.aedt_installed:
            print("❌ 无法连接HFSS - 环境检查未通过")
            return False

        print("\n" + "="*60)
        print("连接到HFSS")
        print("="*60)

        try:
            pyaedt = self.pyedt_module

            print(f"尝试启动Ansys Electronics Desktop...")
            print(f"使用版本: {'最新版本' if version is None else version}")
            print(f"图形界面: {'禁用' if non_graphical else '启用'}")
            print(f"API版本: {'新API (ansys.aedt.core)' if self.new_api else '旧API (pyaedt)'}")

            start_time = time.time()

            # 兼容新旧版本的HFSS类初始化
            if self.new_api:
                # 新API: ansys.aedt.core
                self.aedt_app = pyaedt.Hfss(
                    projectname="Antenna_Simulation",
                    designname="Microstrip_Antenna",
                    version=version,
                    non_graphical=non_graphical,
                    new_desktop_session=True
                )
            else:
                # 旧API: pyaedt
                self.aedt_app = pyaedt.Hfss(
                    projectname="Antenna_Simulation",
                    designname="Microstrip_Antenna",
                    version=version,
                    non_graphical=non_graphical,
                    new_desktop_session=True
                )

            connect_time = time.time() - start_time

            if self.aedt_app:
                self.hfss = self.aedt_app

                # 兼容新旧版本的项目和设计属性
                if hasattr(self.aedt_app, 'project'):
                    self.project = self.aedt_app.project
                elif hasattr(self.aedt_app, '_project'):
                    self.project = self.aedt_app._project
                else:
                    self.project = None

                if hasattr(self.aedt_app, 'design'):
                    self.design = self.aedt_app.design
                elif hasattr(self.aedt_app, '_design'):
                    self.design = self.aedt_app._design
                else:
                    self.design = None

                print(f"✅ HFSS连接成功!")
                print(f"   连接耗时: {connect_time:.2f}秒")
                if self.project:
                    print(f"   项目名称: {self.project.name}")
                if self.design:
                    print(f"   设计名称: {self.design.name}")

                return True
            else:
                print("❌ HFSS连接失败")
                return False

        except Exception as e:
            print(f"❌ 连接HFSS时发生错误: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")

            # 清理资源
            self.close()
            return False

    def create_microstrip_antenna(self, params=None):
        """创建微带天线模型 - 兼容新旧API"""
        if not self.hfss:
            print("❌ 未连接到HFSS，无法创建天线模型")
            return False

        print("\n" + "="*60)
        print("创建微带天线模型")
        print("="*60)

        # 默认参数
        if params is None:
            params = {
                'patch_length': 20.0,        # mm
                'patch_width': 15.0,         # mm
                'substrate_thickness': 1.6,  # mm
                'substrate_epsr': 4.4,       # FR-4介电常数
                'substrate_length': 30.0,    # mm
                'substrate_width': 25.0,     # mm
                'ground_length': 30.0,       # mm
                'ground_width': 25.0,        # mm
                'feed_position': 0.25,       # 相对位置
                'feed_length': 10.0,         # mm
                'feed_width': 2.0            # mm
            }

        print(f"天线参数: {params}")

        try:
            # 设置单位为毫米 - 兼容新旧版本
            if hasattr(self.hfss.modeler, 'model_units'):
                self.hfss.modeler.model_units = "mm"
            elif hasattr(self.hfss.modeler, 'set_units'):
                self.hfss.modeler.set_units("mm")
            print("✅ 单位设置为毫米")

            # 1. 创建接地平面
            ground = self.hfss.modeler.create_rectangle(
                position=[0, 0, 0],
                dimension_list=[params['ground_length'], params['ground_width']],
                name="Ground",
                material="copper"
            )
            print("✅ 接地平面创建成功")

            # 2. 创建介质板
            substrate = self.hfss.modeler.create_box(
                position=[0, 0, 0],
                dimension_list=[params['substrate_length'], params['substrate_width'], params['substrate_thickness']],
                name="Substrate"
            )

            # 设置介质板材料属性 - 兼容新旧版本
            try:
                material_name = "FR4_epoxy"

                # 检查材料是否存在
                if hasattr(self.hfss.materials, 'material_names'):
                    materials_list = self.hfss.materials.material_names
                elif hasattr(self.hfss, 'materials') and hasattr(self.hfss.materials, 'keys'):
                    materials_list = list(self.hfss.materials.keys())
                else:
                    materials_list = []

                # 创建FR-4材料（如果不存在）
                if material_name not in materials_list:
                    if hasattr(self.hfss.materials, 'create_material'):
                        self.hfss.materials.create_material(
                            material_name=material_name,
                            material_properties={"permittivity": params['substrate_epsr'], "loss_tangent": 0.02}
                        )
                    elif hasattr(self.hfss, 'create_material'):
                        self.hfss.create_material(
                            materialname=material_name,
                            matprop={"permittivity": params['substrate_epsr'], "loss_tangent": 0.02}
                        )
                    print("✅ 创建FR4材料成功")

                # 设置材料
                if hasattr(substrate, 'material_name'):
                    substrate.material_name = material_name
                elif hasattr(substrate, 'material'):
                    substrate.material = material_name

                print(f"✅ 介质板材料设置为: {material_name} (εr={params['substrate_epsr']})")

            except Exception as e:
                print(f"⚠️ 设置介质材料时出错: {str(e)}")
                try:
                    if hasattr(substrate, 'material_name'):
                        substrate.material_name = "dielectric"
                    elif hasattr(substrate, 'material'):
                        substrate.material = "dielectric"
                    print("⚠️ 使用默认介质材料")
                except:
                    print("⚠️ 无法设置材料属性")

            # 3. 创建辐射贴片
            patch_x = (params['substrate_length'] - params['patch_length']) / 2
            patch_y = (params['substrate_width'] - params['patch_width']) / 2

            patch = self.hfss.modeler.create_rectangle(
                position=[patch_x, patch_y, params['substrate_thickness']],
                dimension_list=[params['patch_length'], params['patch_width']],
                name="Patch",
                material="copper"
            )
            print("✅ 辐射贴片创建成功")

            # 4. 创建馈线
            feed_x = patch_x + params['feed_position'] * params['patch_length']
            feed_y = (params['substrate_width'] - params['feed_width']) / 2

            feed = self.hfss.modeler.create_rectangle(
                position=[feed_x, feed_y, params['substrate_thickness']],
                dimension_list=[params['feed_length'], params['feed_width']],
                name="Feed",
                material="copper"
            )
            print("✅ 馈线创建成功")

            # 5. 创建端口 - 兼容新旧版本
            port_x = feed_x + params['feed_length']
            port_y = params['substrate_width'] / 2
            port_z = params['substrate_thickness'] / 2

            try:
                # 尝试新的端口创建方法
                if hasattr(self.hfss, 'create_lumped_port'):
                    port = self.hfss.create_lumped_port(
                        name="Port1",
                        position=[port_x, port_y, port_z],
                        orientation=[1, 0, 0],
                        length=params['substrate_width'],
                        width=params['feed_width']
                    )
                elif hasattr(self.hfss, 'create_port'):
                    # 旧版本的端口创建方法
                    port = self.hfss.create_port(
                        name="Port1",
                        location=[port_x, port_y, port_z],
                        direction=[1, 0, 0],
                        porttype="lumped"
                    )
                else:
                    # 最兼容的方法
                    port = self.hfss.modeler.create_rectangle(
                        position=[port_x - 0.5, port_y - params['feed_width']/2, 0],
                        dimension_list=[1, params['feed_width'], params['substrate_thickness']],
                        name="Port1"
                    )
                    self.hfss.assign_lumped_port_to_sheet("Port1", "Port1")

                print("✅ 集总端口创建成功")

            except Exception as e:
                print(f"⚠️ 创建端口时出错: {str(e)}")
                # 尝试简化的端口创建
                try:
                    # 创建端口面
                    port_sheet = self.hfss.modeler.create_rectangle(
                        position=[port_x - 0.1, port_y - params['feed_width']/2, 0],
                        dimension_list=[0.2, params['feed_width'], params['substrate_thickness']],
                        name="Port1"
                    )
                    # 分配端口属性
                    if hasattr(self.hfss, 'assign_port'):
                        self.hfss.assign_port("Port1", "Port1")
                    print("⚠️ 使用简化方法创建端口成功")
                except Exception as e2:
                    print(f"❌ 无法创建端口: {str(e2)}")

            # 6. 创建辐射边界 - 兼容新旧版本
            try:
                if hasattr(self.hfss, 'create_radiation_boundary'):
                    self.hfss.create_radiation_boundary(
                        name="Radiation_Boundary",
                        distance=20
                    )
                elif hasattr(self.hfss, 'assign_radiation_boundary'):
                    # 为所有外表面分配辐射边界
                    self.hfss.assign_radiation_boundary_to_all_external_faces("Radiation_Boundary")
                else:
                    # 创建空气盒并分配辐射边界
                    air_box = self.hfss.modeler.create_box(
                        position=[-10, -10, -10],
                        dimension_list=[params['substrate_length'] + 20, params['substrate_width'] + 20, params['substrate_thickness'] + 20],
                        name="AirBox"
                    )
                    air_box.material_name = "air"
                    self.hfss.assign_radiation_boundary_to_faces("AirBox", ["+X", "-X", "+Y", "-Y", "+Z"], "Radiation_Boundary")

                print("✅ 辐射边界创建成功")

            except Exception as e:
                print(f"⚠️ 创建辐射边界时出错: {str(e)}")

            return True

        except Exception as e:
            print(f"❌ 创建天线模型失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def set_simulation_parameters(self, start_freq=2.0, stop_freq=4.0, num_points=200):
        """设置仿真参数 - 兼容新旧API"""
        if not self.hfss:
            print("❌ 未连接到HFSS，无法设置仿真参数")
            return False

        print("\n" + "="*60)
        print(f"设置仿真参数: {start_freq}-{stop_freq}GHz")
        print("="*60)

        try:
            # 创建驱动解决方案 - 兼容新旧版本
            if hasattr(self.hfss, 'create_setup'):
                self.solution = self.hfss.create_setup(
                    setupname="Setup1",
                    setuptype="DrivenModal"
                )
            elif hasattr(self.hfss, 'create_driven_setup'):
                self.solution = self.hfss.create_driven_setup(
                    name="Setup1",
                    solution_type="Modal"
                )
            else:
                print("❌ 无法创建解决方案")
                return False

            # 设置自适应频率 - 兼容新旧版本
            center_freq = (start_freq + stop_freq) / 2
            if hasattr(self.solution, 'props') and isinstance(self.solution.props, dict):
                self.solution.props["AdaptiveFrequency"] = f"{center_freq}GHz"
                self.solution.props["MaximumPasses"] = 10
                self.solution.props["MinimumConvergedPasses"] = 2
                self.solution.props["ConvergenceThreshold"] = 0.02
            elif hasattr(self.solution, 'set_property'):
                self.solution.set_property("AdaptiveFrequency", f"{center_freq}GHz")
                self.solution.set_property("MaximumPasses", 10)
                self.solution.set_property("MinimumConvergedPasses", 2)
                self.solution.set_property("ConvergenceThreshold", 0.02)
            else:
                # 直接设置属性
                self.solution.AdaptiveFrequency = f"{center_freq}GHz"
                self.solution.MaximumPasses = 10
                self.solution.MinimumConvergedPasses = 2
                self.solution.ConvergenceThreshold = 0.02

            print(f"✅ 自适应频率设置为: {center_freq} GHz")
            print("✅ 收敛标准设置完成")

            # 创建频率扫描 - 兼容新旧版本
            try:
                if hasattr(self.solution, 'create_frequency_sweep'):
                    sweep = self.solution.create_frequency_sweep(
                        sweepname="Sweep1",
                        freqstart=f"{start_freq}GHz",
                        freqstop=f"{stop_freq}GHz",
                        numpoints=num_points,
                        sweep_type="Linear"
                    )
                elif hasattr(self.hfss, 'create_frequency_sweep'):
                    sweep = self.hfss.create_frequency_sweep(
                        setupname="Setup1",
                        sweepname="Sweep1",
                        freqstart=f"{start_freq}GHz",
                        freqstop=f"{stop_freq}GHz",
                        numpoints=num_points,
                        sweep_type="Linear"
                    )
                else:
                    # 旧版本的扫描创建方法
                    sweep = self.hfss.create_sweep(
                        "Setup1",
                        "Sweep1",
                        "LinearCount",
                        f"{start_freq}GHz",
                        f"{stop_freq}GHz",
                        num_points
                    )

                print(f"✅ 频率扫描设置完成:")
                print(f"   频率范围: {start_freq}-{stop_freq} GHz")
                print(f"   采样点数: {num_points}")
                print(f"   扫频类型: 线性")

            except Exception as e:
                print(f"⚠️ 创建频率扫描时出错: {str(e)}")
                # 尝试直接设置扫描参数
                try:
                    self.hfss.set_frequency_sweep(
                        "Setup1",
                        f"{start_freq}GHz",
                        f"{stop_freq}GHz",
                        num_points,
                        "Linear"
                    )
                    print("✅ 使用替代方法设置频率扫描成功")
                except Exception as e2:
                    print(f"❌ 无法设置频率扫描: {str(e2)}")
                    return False

            return True
        except Exception as e:
            print(f"❌ 设置仿真参数失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def run_simulation(self):
        """运行仿真 - 兼容新旧API"""
        if not self.hfss:
            print("❌ 未连接到HFSS或未设置解决方案，无法运行仿真")
            return False

        print("\n" + "="*60)
        print("开始HFSS仿真")
        print("="*60)

        try:
            # 保存项目 - 兼容新旧版本
            if self.project:
                try:
                    self.project.save()
                except:
                    if hasattr(self.hfss, 'save_project'):
                        self.hfss.save_project()
                    else:
                        self.hfss.save()
                print("✅ 项目已保存")

            # 运行仿真 - 兼容新旧版本
            print("▶ 开始仿真计算...")
            start_time = time.time()

            try:
                if self.solution and hasattr(self.solution, 'analyze'):
                    self.solution.analyze()
                elif hasattr(self.hfss, 'analyze_setup'):
                    self.hfss.analyze_setup("Setup1")
                elif hasattr(self.hfss, 'analyze'):
                    self.hfss.analyze()
                else:
                    # 最基本的分析方法
                    self.hfss.run_analysis("Setup1")

            except Exception as e:
                print(f"⚠️ 主要分析方法失败: {str(e)}")
                # 尝试其他分析方法
                if hasattr(self.hfss, 'solve'):
                    self.hfss.solve()
                elif hasattr(self.hfss, 'run_simulation'):
                    self.hfss.run_simulation()
                else:
                    raise

            end_time = time.time()
            simulation_time = end_time - start_time

            print(f"✅ 仿真完成!")
            print(f"⏱️ 仿真耗时: {simulation_time:.1f}秒")
            print(f"   = {simulation_time/60:.1f}分钟")

            return True
        except Exception as e:
            print(f"❌ 仿真失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def extract_s_parameters(self):
        """提取S参数 - 兼容新旧API"""
        if not self.hfss:
            print("❌ 未连接到HFSS或未完成仿真，无法提取S参数")
            return None

        print("\n" + "="*60)
        print("提取S参数结果")
        print("="*60)

        try:
            # 获取S参数数据 - 兼容新旧版本
            print("正在提取S参数数据...")

            frequencies = None
            s11_mag = None
            s11_phase = None
            peak_gain = 0.0

            try:
                # 尝试新的后处理API
                if hasattr(self.hfss.post, 'get_solution_data'):
                    freq_data = self.hfss.post.get_solution_data(
                        expressions=["S(1,1)"],
                        setup_sweep_name="Setup1:Sweep1"
                    )

                    if freq_data:
                        frequencies = freq_data.frequencies
                        s11_complex = freq_data.data_real("S(1,1)") + 1j * freq_data.data_imag("S(1,1)")
                        s11_mag = 20 * np.log10(np.abs(s11_complex))
                        s11_phase = np.angle(s11_complex, deg=True)

                elif hasattr(self.hfss, 'post') and hasattr(self.hfss.post, 'solution_data'):
                    # 旧版本的后处理API
                    freq_data = self.hfss.post.solution_data(
                        expressions=["S(1,1)"],
                        setupname="Setup1",
                        sweepname="Sweep1"
                    )

                    if freq_data:
                        frequencies = freq_data.frequencies
                        s11_complex = freq_data["S(1,1)"].real + 1j * freq_data["S(1,1)"].imag
                        s11_mag = 20 * np.log10(np.abs(s11_complex))
                        s11_phase = np.angle(s11_complex, deg=True)

                else:
                    # 最基本的S参数提取方法
                    s_params = self.hfss.get_s_parameters("Setup1:Sweep1")
                    frequencies = s_params.frequencies
                    s11_complex = s_params["S11"]
                    s11_mag = 20 * np.log10(np.abs(s11_complex))
                    s11_phase = np.angle(s11_complex, deg=True)

            except Exception as e:
                print(f"⚠️ 使用主要方法提取S参数失败: {str(e)}")
                # 尝试直接从文件读取
                try:
                    # 导出S参数到文件
                    self.hfss.export_s_parameters("s_params.s2p")

                    # 读取S2P文件
                    with open("s_params.s2p", 'r') as f:
                        lines = f.readlines()

                    frequencies = []
                    s11_mag = []
                    s11_phase = []

                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('!') and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 3:
                                freq = float(parts[0])  # GHz
                                mag = float(parts[1])   # dB
                                phase = float(parts[2]) # degrees
                                frequencies.append(freq)
                                s11_mag.append(mag)
                                s11_phase.append(phase)

                    frequencies = np.array(frequencies) * 1e9  # 转换为Hz
                    s11_mag = np.array(s11_mag)
                    s11_phase = np.array(s11_phase)

                    print("✅ 使用S2P文件提取S参数成功")

                except Exception as e2:
                    print(f"❌ 所有S参数提取方法都失败: {str(e2)}")
                    return None

            if frequencies is None or s11_mag is None:
                print("❌ 无法获取S参数数据")
                return None

            # 转换频率单位为GHz
            frequencies_ghz = frequencies / 1e9

            print(f"✅ S参数提取完成:")
            print(f"   频率范围: {frequencies_ghz.min():.2f}-{frequencies_ghz.max():.2f} GHz")
            print(f"   数据点数: {len(frequencies_ghz)}")
            print(f"   S11最小值: {s11_mag.min():.2f} dB")

            # 找到谐振频率
            min_s11_idx = np.argmin(s11_mag)
            resonance_freq = frequencies_ghz[min_s11_idx]
            print(f"   谐振频率: {resonance_freq:.2f} GHz")

            # 计算带宽 (S11 <= -10dB)
            s11_threshold = -10
            bandwidth_points = frequencies_ghz[s11_mag <= s11_threshold]
            if len(bandwidth_points) > 1:
                bandwidth = bandwidth_points[-1] - bandwidth_points[0]
                print(f"   带宽: {bandwidth*1000:.0f} MHz")
            else:
                bandwidth = 0
                print(f"   带宽: 无法计算 (S11未达到-10dB)")

            # 获取增益数据 - 兼容新旧版本
            try:
                if hasattr(self.hfss.post, 'get_solution_data'):
                    gain_data = self.hfss.post.get_solution_data(
                        expressions=["GainTotal"],
                        setup_sweep_name="Setup1:Sweep1"
                    )
                    gain_values = gain_data.data_real("GainTotal")
                    peak_gain = np.max(gain_values)
                elif hasattr(self.hfss, 'get_gain_pattern'):
                    gain_pattern = self.hfss.get_gain_pattern("Setup1", "Sweep1", resonance_freq * 1e9)
                    peak_gain = np.max(gain_pattern.gain)
                elif hasattr(self.hfss, 'calculate_gain'):
                    peak_gain = self.hfss.calculate_gain("Setup1", resonance_freq * 1e9)
                else:
                    peak_gain = 0.0

                print(f"   峰值增益: {peak_gain:.1f} dBi")

            except Exception as e:
                peak_gain = 0.0
                print(f"⚠️ 无法提取增益数据: {str(e)}")

            return {
                'frequencies': frequencies_ghz,
                's11_mag': s11_mag,
                's11_phase': s11_phase,
                'resonance_frequency': resonance_freq,
                'bandwidth': bandwidth * 1000,
                'min_s11': s11_mag.min(),
                'peak_gain': peak_gain,
                'method': 'hfss_simulation',
                'api_version': 'new' if self.new_api else 'old'
            }

        except Exception as e:
            print(f"❌ 提取S参数失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return None

    def plot_s_parameters(self, s_params, save_path="hfss_s_parameters.png"):
        """绘制S参数图表"""
        if s_params is None:
            print("❌ 没有S参数数据，无法绘图")
            return

        print("\n" + "="*60)
        print("绘制S参数图表")
        print("="*60)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

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
                 f'带宽: {s_params["bandwidth"]:.0f} MHz\n' \
                 f'峰值增益: {s_params["peak_gain"]:.1f} dBi\n' \
                 f'API版本: {"新API" if s_params["api_version"] == "new" else "旧API"}'

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

        # 史密斯圆图
        from matplotlib.patches import Circle
        smith_ax = ax3
        smith_ax.set_xlim(-1.2, 1.2)
        smith_ax.set_ylim(-1.2, 1.2)
        smith_ax.set_aspect('equal')

        # 绘制史密斯圆图背景
        smith_ax.add_patch(Circle((0, 0), 1, fill=False, color='gray', linestyle='--'))
        smith_ax.axhline(y=0, color='gray', linestyle='--')
        smith_ax.axvline(x=0, color='gray', linestyle='--')

        # 计算阻抗数据
        s11_complex = 10 ** (s_params['s11_mag'] / 20) * np.exp(1j * np.radians(s_params['s11_phase']))
        z_normalized = (1 + s11_complex) / (1 - s11_complex)

        # 绘制S11轨迹
        smith_ax.plot(z_normalized.real, z_normalized.imag, 'b-', linewidth=2, label='S11轨迹')
        smith_ax.set_xlabel('实部', fontsize=12)
        smith_ax.set_ylabel('虚部', fontsize=12)
        smith_ax.set_title('史密斯圆图', fontsize=14, fontweight='bold')
        smith_ax.grid(True, alpha=0.3)
        smith_ax.legend(fontsize=10)

        # 天线参数表格
        ax4.axis('tight')
        ax4.axis('off')

        param_data = [
            ['参数名称', '数值', '单位'],
            ['谐振频率', f"{s_params['resonance_frequency']:.2f}", 'GHz'],
            ['最小S11', f"{s_params['min_s11']:.2f}", 'dB'],
            ['带宽 (-10dB)', f"{s_params['bandwidth']:.0f}", 'MHz'],
            ['峰值增益', f"{s_params['peak_gain']:.1f}", 'dBi'],
            ['仿真方法', 'HFSS', ''],
            ['API版本', '新API' if s_params['api_version'] == 'new' else '旧API', ''],
        ]

        table = ax4.table(cellText=param_data[1:], colLabels=param_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax4.set_title('仿真结果汇总', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ S参数图表已保存到: {save_path}")

    def save_results(self, s_params, save_path="hfss_s_parameters_results.npy"):
        """保存S参数结果"""
        if s_params is None:
            print("❌ 没有S参数数据，无法保存")
            return

        # 保存完整结果
        np.save(save_path, s_params)
        print(f"✅ S参数数据已保存到: {save_path}")

        # 保存简化的文本结果
        with open("hfss_s_parameters_summary.txt", "w", encoding='utf-8') as f:
            f.write("HFSS天线S参数计算结果\n")
            f.write("="*50 + "\n")
            f.write(f"计算时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"仿真软件: Ansys HFSS\n")
            f.write(f"API版本: {'新API (ansys.aedt.core)' if s_params['api_version'] == 'new' else '旧API (pyaedt)'}\n")
            f.write(f"频率范围: {s_params['frequencies'].min():.2f} - {s_params['frequencies'].max():.2f} GHz\n")
            f.write(f"数据点数: {len(s_params['frequencies'])}\n")
            f.write("\n关键性能指标:\n")
            f.write(f"谐振频率: {s_params['resonance_frequency']:.2f} GHz\n")
            f.write(f"最小S11: {s_params['min_s11']:.2f} dB\n")
            f.write(f"带宽 (-10dB): {s_params['bandwidth']:.0f} MHz\n")
            f.write(f"峰值增益: {s_params['peak_gain']:.1f} dBi\n")

        print("✅ S参数摘要已保存到: hfss_s_parameters_summary.txt")

    def close(self):
        """关闭HFSS和清理资源 - 兼容新旧API"""
        print("\n" + "="*60)
        print("清理HFSS资源")
        print("="*60)

        try:
            if self.aedt_app:
                try:
                    # 保存项目
                    if self.project:
                        try:
                            self.project.save()
                        except:
                            if hasattr(self.aedt_app, 'save_project'):
                                self.aedt_app.save_project()
                            elif hasattr(self.aedt_app, 'save'):
                                self.aedt_app.save()
                        print("✅ 项目已保存")

                    # 关闭AEDT应用 - 兼容新旧版本
                    if hasattr(self.aedt_app, 'release_desktop'):
                        self.aedt_app.release_desktop()
                    elif hasattr(self.aedt_app, 'close_desktop'):
                        self.aedt_app.close_desktop()
                    elif hasattr(self.aedt_app, 'quit'):
                        self.aedt_app.quit()
                    else:
                        # 旧版本的关闭方法
                        self.aedt_app.exit()

                    print("✅ HFSS已成功关闭")

                except Exception as e:
                    print(f"⚠️ 关闭HFSS时出错: {str(e)}")

            self.aedt_app = None
            self.hfss = None
            self.project = None
            self.design = None
            self.solution = None

        except Exception as e:
            print(f"❌ 清理资源时发生错误: {str(e)}")

    def calculate_s_parameters(self, antenna_params=None, start_freq=2.0, stop_freq=4.0, num_points=200):
        """完整的S参数计算流程"""
        print("="*60)
        print("开始HFSS天线S参数计算流程")
        print("="*60)

        try:
            # 1. 连接HFSS
            if not self.connect_to_hfss(non_graphical=False):
                return None

            # 2. 创建天线模型
            if not self.create_microstrip_antenna(antenna_params):
                self.close()
                return None

            # 3. 设置仿真参数
            if not self.set_simulation_parameters(start_freq, stop_freq, num_points):
                self.close()
                return None

            # 4. 运行仿真
            if not self.run_simulation():
                self.close()
                return None

            # 5. 提取S参数
            s_params = self.extract_s_parameters()

            # 6. 可视化结果
            if s_params:
                self.plot_s_parameters(s_params)
                self.save_results(s_params)

            # 7. 清理
            self.close()

            return s_params

        except Exception as e:
            print(f"\n❌ S参数计算流程失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            self.close()
            return None

def main():
    """主函数"""
    # 创建HFSS天线仿真器实例
    simulator = HFSSAntennaSimulator()

    # 检查环境
    if not simulator.pyedt_available or not simulator.aedt_installed:
        print("\n❌ 环境检查失败，无法继续")
        print("\n请确保：")
        print("1. 已安装PyAEDT库:")
        print("   - 旧版本: pip install pyaedt")
        print("   - 新版本: pip install ansys-aedt")
        print("2. 已安装Ansys Electronics Desktop 2022 R1或更高版本")
        print("3. 已配置Ansys许可证")
        return

    # 自定义天线参数
    custom_antenna_params = {
        'patch_length': 18.0,        # mm
        'patch_width': 24.0,         # mm
        'substrate_thickness': 1.6,  # mm
        'substrate_epsr': 4.4,       # FR-4
        'substrate_length': 35.0,    # mm
        'substrate_width': 35.0,     # mm
        'ground_length': 35.0,       # mm
        'ground_width': 35.0,        # mm
        'feed_position': 0.3,        # 相对位置
        'feed_length': 12.0,         # mm
        'feed_width': 1.5            # mm
    }

    # 计算S参数 (2-3GHz频率范围)
    s_parameters = simulator.calculate_s_parameters(
        antenna_params=custom_antenna_params,
        start_freq=2.0,
        stop_freq=3.0,
        num_points=200
    )

    # 显示结果摘要
    if s_parameters:
        print("\n" + "="*60)
        print("HFSS S参数计算完成！")
        print("="*60)
        print(f"谐振频率: {s_parameters['resonance_frequency']:.2f} GHz")
        print(f"最小S11: {s_parameters['min_s11']:.2f} dB")
        print(f"带宽: {s_parameters['bandwidth']:.0f} MHz")
        print(f"峰值增益: {s_parameters['peak_gain']:.1f} dBi")
        print(f"API版本: {'新API (ansys.aedt.core)' if s_parameters['api_version'] == 'new' else '旧API (pyaedt)'}")
        print("\n生成的文件:")
        print("- hfss_s_parameters.png: S参数图表 (包含史密斯圆图)")
        print("- hfss_s_parameters_results.npy: 完整数据")
        print("- hfss_s_parameters_summary.txt: 结果摘要")
    else:
        print("\n❌ S参数计算失败")

if __name__ == "__main__":
    main()