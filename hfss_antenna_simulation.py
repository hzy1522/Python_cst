"""
AEDT HFSS 微带贴片天线仿真 Python 脚本
增强版 - 支持非默认安装路径和早期AEDT版本
"""

import pyaedt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pkg_resources
import sys
import winreg
import subprocess
import time

class AdvancedHFSSEntennaSimulator:
    """增强版 AEDT HFSS 天线仿真器类"""

    def __init__(self, project_name="Antenna_Project", design_name="Microstrip_Patch",
                 aedt_version="2022.1", aedt_install_path=None, non_graphical=False):
        """
        初始化增强版 HFSS 仿真器

        Args:
            project_name: 项目名称
            design_name: 设计名称
            aedt_version: AEDT 版本号 (如 "2022.1", "2023.2")
            aedt_install_path: AEDT 安装路径 (如 "D:\\AnsysEM\\v221")
            non_graphical: 是否非图形化模式运行
        """
        self.project_name = project_name
        self.design_name = design_name
        self.aedt_version = aedt_version
        self.aedt_install_path = aedt_install_path
        self.non_graphical = non_graphical

        # 天线设计参数
        self.antenna_params = {
            "frequency": 2.4,          # 工作频率 (GHz)
            "substrate_thickness": 1.6, # 基板厚度 (mm)
            "substrate_er": 4.4,       # 基板介电常数
            "substrate_tand": 0.02,    # 基板损耗角正切
            "patch_length": 31.0,      # 贴片长度 (mm)
            "patch_width": 38.0,       # 贴片宽度 (mm)
            "ground_length": 60.0,     # 接地板长度 (mm)
            "ground_width": 60.0,      # 接地板宽度 (mm)
            "feed_position_x": 7.75,   # 馈电位置X (mm)
            "feed_position_y": 0,      # 馈电位置Y (mm)
            "feed_width": 3.0,         # 馈线宽度 (mm)
            "feed_length": 20.0        # 馈线长度 (mm)
        }

        self.app = None
        self.hfss = None
        self.setup = None
        self.pyaedt_version = self._get_pyaedt_version()
        self.aedt_process = None

    def _get_pyaedt_version(self):
        """获取 PyAEDT 版本号"""
        try:
            return pkg_resources.get_distribution("pyaedt").version
        except pkg_resources.DistributionNotFound:
            return "0.0.0"

    def _find_aedt_installation(self):
        """从注册表查找AEDT安装路径"""
        if self.aedt_install_path:
            if os.path.exists(self.aedt_install_path):
                print(f"使用指定的AEDT安装路径: {self.aedt_install_path}")
                return self.aedt_install_path
            else:
                print(f"指定的AEDT路径不存在: {self.aedt_install_path}")

        print("正在从注册表查找AEDT安装路径...")

        try:
            # 转换版本号为注册表格式 (如 2022.1 -> 22.1)
            version_parts = self.aedt_version.split('.')
            reg_version = f"{version_parts[0][-2:]}.{version_parts[1]}"

            # 尝试不同的注册表路径
            reg_paths = [
                f"SOFTWARE\\Ansys, Inc.\\ANSYS Electromagnetics\\{reg_version}",
                f"SOFTWARE\\Wow6432Node\\Ansys, Inc.\\ANSYS Electromagnetics\\{reg_version}",
                f"SOFTWARE\\Ansys, Inc.\\ANSYS EM\\{reg_version}",
                f"SOFTWARE\\Wow6432Node\\Ansys, Inc.\\ANSYS EM\\{reg_version}"
            ]

            for reg_path in reg_paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    winreg.CloseKey(key)

                    if os.path.exists(install_dir):
                        print(f"从注册表找到AEDT安装路径: {install_dir}")
                        self.aedt_install_path = install_dir
                        return install_dir
                except WindowsError:
                    continue

            print("未在注册表中找到AEDT安装信息")

        except Exception as e:
            print(f"查找注册表时出错: {e}")

        # 如果找不到，尝试常见的安装路径
        common_paths = [
            f"C:\\Program Files\\AnsysEM\\v{version_parts[0][-2:]}{version_parts[1]}",
            f"D:\\AnsysEM\\v{version_parts[0][-2:]}{version_parts[1]}",
            f"C:\\Program Files\\Ansys Inc\\v{version_parts[0][-2:]}{version_parts[1]}",
            f"D:\\Ansys Inc\\v{version_parts[0][-2:]}{version_parts[1]}"
        ]

        for path in common_paths:
            if os.path.exists(path):
                print(f"在常见路径找到AEDT: {path}")
                self.aedt_install_path = path
                return path

        print("未找到AEDT安装路径")
        return None

    def _set_environment_variables(self):
        """设置必要的环境变量"""
        if not self.aedt_install_path:
            return False

        print("正在设置环境变量...")

        try:
            # 添加AEDT安装目录到PATH
            if self.aedt_install_path not in os.environ['PATH']:
                os.environ['PATH'] = self.aedt_install_path + ";" + os.environ['PATH']

            # 添加Win64目录
            win64_path = os.path.join(self.aedt_install_path, "Win64")
            if os.path.exists(win64_path) and win64_path not in os.environ['PATH']:
                os.environ['PATH'] = win64_path + ";" + os.environ['PATH']

            # 设置AEDT特定的环境变量
            os.environ['ANSYSEM_ROOT'] = self.aedt_install_path
            os.environ['AEDT_ROOT'] = self.aedt_install_path

            print("环境变量设置完成")
            return True

        except Exception as e:
            print(f"设置环境变量时出错: {e}")
            return False

    def _start_aedt_process(self):
        """手动启动AEDT进程"""
        if not self.aedt_install_path:
            return False

        print("正在手动启动AEDT进程...")

        try:
            # 查找AEDT可执行文件
            aedt_exe = None
            exe_paths = [
                os.path.join(self.aedt_install_path, "Win64", "ansysedt.exe"),
                os.path.join(self.aedt_install_path, "ansysedt.exe"),
                os.path.join(self.aedt_install_path, "ElectronicsDesktop.exe")
            ]

            for path in exe_paths:
                if os.path.exists(path):
                    aedt_exe = path
                    break

            if not aedt_exe:
                print("未找到AEDT可执行文件")
                return False

            # 构建启动参数
            args = [aedt_exe, "-grpc", "port=50052"]
            if self.non_graphical:
                args.append("-ng")

            print(f"启动AEDT命令: {' '.join(args)}")

            # 启动AEDT进程
            self.aedt_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            # 等待AEDT启动
            print("等待AEDT启动完成...")
            time.sleep(15)  # 等待15秒

            print("AEDT进程启动成功")
            return True

        except Exception as e:
            print(f"启动AEDT进程时出错: {e}")
            if self.aedt_process:
                self.aedt_process.terminate()
            return False

    def _get_launch_parameters(self):
        """根据版本获取正确的启动参数"""
        params = {
            "non_graphical": self.non_graphical,
            "new_desktop_session": False,  # 使用已启动的会话
            "close_on_exit": True,
            "port": 50052  # 使用手动指定的端口
        }

        # 版本比较
        try:
            from packaging import version
            if version.parse(self.pyaedt_version) >= version.parse("0.18.0"):
                params["version"] = self.aedt_version
            else:
                params["specified_version"] = self.aedt_version
        except ImportError:
            params["version"] = self.aedt_version
            params["specified_version"] = self.aedt_version

        return params

    def connect_using_com_interface(self):
        """使用COM接口连接（适用于早期版本）"""
        print("尝试使用COM接口连接AEDT...")

        try:
            # 设置使用COM接口
            from ansys.aedt.core.generic.settings import settings
            settings.use_grpc_api = False

            # 使用COM接口启动
            self.app = pyaedt.launch_desktop(
                specified_version=self.aedt_version,
                non_graphical=self.non_graphical,
                new_desktop_session=True,
                close_on_exit=True
            )

            self.hfss = pyaedt.Hfss(
                projectname=self.project_name,
                designname=self.design_name,
                solution_type="DrivenModal"
            )

            print("使用COM接口成功连接到HFSS")
            return True

        except Exception as e:
            print(f"COM接口连接失败: {e}")
            return False

    def connect_hfss(self):
        """连接到HFSS的增强版方法"""
        print("="*60)
        print("开始连接AEDT HFSS...")
        print(f"PyAEDT版本: {self.pyaedt_version}")
        print(f"AEDT目标版本: {self.aedt_version}")
        print("="*60)

        # 1. 查找AEDT安装路径
        if not self._find_aedt_installation():
            print("无法找到AEDT安装路径，请手动指定aedt_install_path参数")
            return False

        # 2. 设置环境变量
        if not self._set_environment_variables():
            print("环境变量设置失败")
            return False

        # 3. 尝试多种连接方法
        connection_methods = [
            self._connect_method1,  # PyAEDT直接连接
            self._connect_method2,  # 手动启动后连接
            self._connect_method3,  # 使用COM接口
        ]

        for i, method in enumerate(connection_methods, 1):
            print(f"\n尝试连接方法 {i}:")
            if method():
                return True
            print(f"连接方法 {i} 失败")

        print("\n所有连接方法都失败了")
        return False

    def _connect_method1(self):
        """方法1: PyAEDT直接连接"""
        try:
            launch_params = self._get_launch_parameters()
            print(f"使用参数: {launch_params}")

            self.app = pyaedt.launch_desktop(**launch_params)
            self.hfss = pyaedt.Hfss(
                projectname=self.project_name,
                designname=self.design_name,
                solution_type="DrivenModal"
            )

            print("方法1连接成功")
            return True

        except Exception as e:
            print(f"方法1失败: {e}")
            return False

    def _connect_method2(self):
        """方法2: 手动启动AEDT后连接"""
        try:
            # 先手动启动AEDT
            if not self._start_aedt_process():
                return False

            # 然后连接到已启动的实例
            launch_params = self._get_launch_parameters()
            launch_params["new_desktop_session"] = False
            launch_params["port"] = 50052

            self.app = pyaedt.launch_desktop(**launch_params)
            self.hfss = pyaedt.Hfss(
                projectname=self.project_name,
                designname=self.design_name,
                solution_type="DrivenModal"
            )

            print("方法2连接成功")
            return True

        except Exception as e:
            print(f"方法2失败: {e}")
            return False

    def _connect_method3(self):
        """方法3: 使用COM接口连接"""
        return self.connect_using_com_interface()

    def create_materials(self):
        """创建材料"""
        print("正在创建材料...")

        try:
            # 创建基板材料
            substrate_material = self.hfss.materials.add_material(
                "Custom_FR4",
                permittivity=self.antenna_params["substrate_er"],
                loss_tangent=self.antenna_params["substrate_tand"]
            )

            # 使用铜作为金属材料
            if not self.hfss.materials.material_exists("Copper"):
                self.hfss.materials.add_material("Copper", conductivity=5.8e7)

            print("材料创建完成")
            return substrate_material.name

        except Exception as e:
            print(f"创建材料失败: {e}")
            return None

    def create_antenna_model(self):
        """创建天线模型"""
        print("正在创建天线模型...")

        try:
            # 设置单位为毫米
            self.hfss.modeler.set_units("mm")

            # 获取材料名称
            substrate_material = self.create_materials()
            if not substrate_material:
                return False

            # 创建接地板 (PEC)
            ground = self.hfss.modeler.create_rectangle(
                position=[-self.antenna_params["ground_length"]/2,
                         -self.antenna_params["ground_width"]/2,
                         -self.antenna_params["substrate_thickness"]],
                dimension_list=[self.antenna_params["ground_length"],
                               self.antenna_params["ground_width"]],
                name="Ground_Plane"
            )
            ground.material_name = "Copper"

            # 创建基板
            substrate = self.hfss.modeler.create_box(
                position=[-self.antenna_params["ground_length"]/2,
                         -self.antenna_params["ground_width"]/2,
                         -self.antenna_params["substrate_thickness"]],
                dimension_list=[self.antenna_params["ground_length"],
                               self.antenna_params["ground_width"],
                               self.antenna_params["substrate_thickness"]],
                name="Substrate"
            )
            substrate.material_name = substrate_material

            # 创建辐射贴片
            patch = self.hfss.modeler.create_rectangle(
                position=[-self.antenna_params["patch_length"]/2,
                         -self.antenna_params["patch_width"]/2,
                         0],
                dimension_list=[self.antenna_params["patch_length"],
                               self.antenna_params["patch_width"]],
                name="Radiating_Patch"
            )
            patch.material_name = "Copper"

            # 创建微带馈线
            feed_line = self.hfss.modeler.create_rectangle(
                position=[-self.antenna_params["feed_length"],
                         -self.antenna_params["feed_width"]/2,
                         0],
                dimension_list=[self.antenna_params["feed_length"],
                               self.antenna_params["feed_width"]],
                name="Feed_Line"
            )
            feed_line.material_name = "Copper"

            # 创建空气盒
            air_box_size = 300  # 空气盒尺寸 (mm)
            air_box = self.hfss.modeler.create_box(
                position=[-air_box_size/2, -air_box_size/2, -air_box_size/2],
                dimension_list=[air_box_size, air_box_size, air_box_size],
                name="Air_Box"
            )
            air_box.material_name = "air"

            print("天线模型创建完成")
            return True

        except Exception as e:
            print(f"创建天线模型失败: {e}")
            return False

    def set_boundary_conditions(self):
        """设置边界条件"""
        print("正在设置边界条件...")

        try:
            # 设置辐射边界条件
            air_faces = self.hfss.modeler.get_object_faces("Air_Box")
            for face in air_faces:
                self.hfss.assign_radiation_boundary_to_faces(
                    [face], name=f"Rad_Boundary_{face}"
                )

            # 设置集总端口
            feed_faces = self.hfss.modeler.get_object_faces("Feed_Line")
            for face in feed_faces:
                face_center = self.hfss.modeler.get_face_center(face)
                if face_center[0] < -self.antenna_params["feed_length"] + 1:
                    self.hfss.assign_lumped_port_to_face(
                        face,
                        name="Port1",
                        impedance=50,
                        location="Center"
                    )
                    break

            print("边界条件设置完成")
            return True

        except Exception as e:
            print(f"设置边界条件失败: {e}")
            return False

    def create_simulation_setup(self):
        """创建仿真设置"""
        print("正在创建仿真设置...")

        try:
            # 创建驱动模态设置
            self.setup = self.hfss.create_setup(
                setupname="Setup1",
                setuptype="HfssDrivenAuto"
            )

            # 设置求解频率和收敛条件
            self.setup.props["SolutionFrequency"] = f"{self.antenna_params['frequency']}GHz"
            self.setup.props["MaximumPasses"] = 15
            self.setup.props["MinimumPasses"] = 2
            self.setup.props["MaximumDeltaS"] = 0.02

            # 创建扫频设置
            sweep = self.setup.create_frequency_sweep(
                name="Sweep1",
                sweep_type="Fast",
                frequency_range=[f"{self.antenna_params['frequency']-0.2}GHz",
                                f"{self.antenna_params['frequency']+0.2}GHz"],
                number_of_freq_points=401
            )

            print("仿真设置创建完成")
            return True

        except Exception as e:
            print(f"创建仿真设置失败: {e}")
            return False

    def run_simulation(self):
        """运行仿真"""
        print("正在运行仿真...")

        try:
            # 保存项目
            self.hfss.save_project()

            # 运行仿真
            self.hfss.analyze_setup("Setup1")

            print("仿真运行完成")
            return True

        except Exception as e:
            print(f"仿真运行失败: {e}")
            return False

    def post_process_results(self):
        """后处理结果"""
        print("正在处理仿真结果...")

        try:
            # 获取 S 参数
            s_params = self.hfss.post.get_solution_data(
                expressions=["dB(S(1,1))"],
                setup_sweep_name="Setup1 : Sweep1"
            )

            # 绘制 S11 参数
            fig, ax = plt.subplots(figsize=(12, 8))
            freq_data = s_params.frequency_data
            s11_data = s_params.data_real("dB(S(1,1))")

            ax.plot(freq_data, s11_data, linewidth=2, label="S11 (dB)")
            ax.axhline(y=-10, color='r', linestyle='--', label="-10 dB 线")
            ax.set_xlabel("频率 (GHz)", fontsize=12)
            ax.set_ylabel("S参数 (dB)", fontsize=12)
            ax.set_title("微带贴片天线 S11 参数", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-40, 0)

            # 保存图片
            result_dir = os.path.join(os.getcwd(), "simulation_results")
            os.makedirs(result_dir, exist_ok=True)

            plt.savefig(os.path.join(result_dir, "s11_parameter.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()

            # 创建远场设置
            far_field_setup = self.hfss.create_far_field_setup(
                setupname="FarFieldSetup1",
                setup_name="Setup1",
                frequency=f"{self.antenna_params['frequency']}GHz"
            )

            # 计算远场数据
            far_field_data = self.hfss.post.get_far_field_data(
                setup_name="FarFieldSetup1",
                quantity_name="RealizedGainTotal",
                coordinate_system="Sphere",
                phi_angle=0,
                theta_angle=np.linspace(-180, 180, 361)
            )

            # 绘制远场方向图
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            theta = np.radians(far_field_data.theta)
            gain = far_field_data.data_real("RealizedGainTotal")

            ax.plot(theta, gain, linewidth=2)
            ax.set_title(f"微带贴片天线远场方向图 ({self.antenna_params['frequency']} GHz)",
                        fontsize=14, pad=20)
            ax.set_theta_zero_location('top')
            ax.set_theta_direction(-1)

            plt.savefig(os.path.join(result_dir, "far_field_pattern.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()

            print("结果后处理完成")
            return result_dir

        except Exception as e:
            print(f"结果后处理失败: {e}")
            return None

    def export_results(self, result_dir):
        """导出仿真结果"""
        try:
            # 导出 S 参数数据
            s_params = self.hfss.post.get_solution_data(
                expressions=["dB(S(1,1))"],
                setup_sweep_name="Setup1 : Sweep1"
            )

            results_data = np.column_stack((s_params.frequency_data,
                                          s_params.data_real("dB(S(1,1))")))

            np.savetxt(os.path.join(result_dir, "s11_data.txt"),
                      results_data,
                      header="Frequency (GHz), S11 (dB)",
                      delimiter=",")

            # 导出项目文件
            self.hfss.save_project(os.path.join(result_dir, f"{self.project_name}.aedt"))

            print("结果导出完成")

        except Exception as e:
            print(f"导出结果失败: {e}")

    def close_hfss(self):
        """关闭 HFSS"""
        print("正在关闭HFSS...")

        try:
            if self.hfss:
                self.hfss.release_desktop()
                print("HFSS 已通过PyAEDT关闭")

            if self.aedt_process:
                # 如果是手动启动的进程，强制关闭
                self.aedt_process.terminate()
                self.aedt_process.wait()
                print("AEDT进程已强制关闭")

        except Exception as e:
            print(f"关闭HFSS时出错: {e}")

    def run_full_simulation(self):
        """运行完整的仿真流程"""
        print("="*80)
        print("开始增强版微带贴片天线 HFSS 仿真流程")
        print(f"仿真时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyAEDT版本: {self.pyaedt_version}")
        print(f"AEDT版本: {self.aedt_version}")
        print(f"安装路径: {self.aedt_install_path}")
        print("="*80)

        success = False

        try:
            # 1. 连接 HFSS
            if not self.connect_hfss():
                return False

            # 2. 创建天线模型
            if not self.create_antenna_model():
                return False

            # 3. 设置边界条件
            if not self.set_boundary_conditions():
                return False

            # 4. 创建仿真设置
            if not self.create_simulation_setup():
                return False

            # 5. 运行仿真
            if not self.run_simulation():
                return False

            # 6. 后处理结果
            result_dir = self.post_process_results()
            if result_dir:
                self.export_results(result_dir)
                print(f"仿真结果已保存到: {result_dir}")

            print("="*80)
            print("仿真流程完全成功！")
            print("="*80)

            success = True

        except Exception as e:
            print(f"仿真流程异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 确保关闭 HFSS
            self.close_hfss()

        return success

def main():
    """主函数 - 提供详细的使用示例"""
    print("="*80)
    print("AEDT HFSS 天线仿真脚本 - 增强版")
    print("支持非默认安装路径和早期AEDT版本")
    print("="*80)

    # 请根据您的实际情况修改以下参数
    AEDT_VERSION = "2025R1"  # 您的AEDT版本号
    AEDT_INSTALL_PATH = None  # 如果知道安装路径，可以直接指定
    # AEDT_INSTALL_PATH = "D:\\AnsysEM\\v221"  # 示例路径

    # 创建仿真器实例
    simulator = AdvancedHFSSEntennaSimulator(
        project_name="Microstrip_Antenna_Project",
        design_name="Patch_Antenna_Design",
        aedt_version=AEDT_VERSION,
        aedt_install_path=AEDT_INSTALL_PATH,
        non_graphical=False  # 建议先使用图形化模式调试
    )

    # 可以根据需要修改天线参数
    # simulator.antenna_params["frequency"] = 5.8  # 改为5.8GHz
    # simulator.antenna_params["substrate_thickness"] = 0.8  # 更薄的基板

    # 运行完整仿真
    success = simulator.run_full_simulation()

    if success:
        print("\n" + "="*80)
        print("天线仿真成功完成！")
        print("生成的文件:")
        print("- s11_parameter.png: S11参数曲线图")
        print("- far_field_pattern.png: 远场方向图")
        print("- s11_data.txt: S11数据文件")
        print("- Microstrip_Antenna_Project.aedt: HFSS项目文件")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("天线仿真失败，请检查错误信息。")
        print("建议检查:")
        print("1. AEDT版本号是否正确")
        print("2. 安装路径是否正确")
        print("3. 是否有管理员权限")
        print("4. AEDT许可证是否有效")
        print("="*80)

if __name__ == "__main__":
    main()