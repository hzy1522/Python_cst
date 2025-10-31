import csv
import os
import tempfile
import time

import numpy as np
import ansys.aedt.core
import pyvista as pv
import pandas as pd
from pyvista import examples
pv.set_jupyter_backend("trame")
from ansys.aedt.core.modeler.advanced_cad.stackup_3d import Stackup3D
from ansys.aedt.core.visualization.advanced.farfield_visualization import FfdSolutionData
from django.contrib.messages import success
from ansys.aedt.core import Hfss
import matplotlib.pyplot as plt

class AdvancedHFSSEntennaSimulator:

    def __init__(self, temp_folder=None,   NG_MODE=None,   AEDT_VERSION=None,   NUM_CORES=None):

        self.temp_folder = temp_folder
        self.NG_MODE = NG_MODE
        self.AEDT_VERSION = AEDT_VERSION
        self.NUM_CORES = NUM_CORES
        self.project_name = os.path.join(self.temp_folder.name, "patch.aedt")

        self.antenna_params = {
            "unit": "GHz", #单位设置
            "start_frequency": 8,  # 起始工作频率 (GHz)
            "stop_frequency": 12,  #截止频率
            "center_frequency": 10,  #中心频率
            "sweep_type": "Fast", #扫描频率设置
            "ground_thickness": 0.035,  # 地板厚度 (mm)
            "signal_layer_thickness": 0.035, #信号线厚度(mm)
            "patch_length": 9.57, # 贴片长度(mm)
            "patch_width": 9.25, #
            "patch_name": "Patch",
            "freq_step" : "2GHz",
            "num_of_freq_points": 101,
        }
        self.disc_sweep = None
        self.interp_sweep = None
    def run_full_hfss_simulation(self):
        print("=" * 80)
        print("开始增强版微带贴片天线 HFSS 仿真流程")

        success = False

        try:
            # 1. 连接 HFSS
            if not self.start_hfss_simulation():
                return False
            #2. 设置单位
            if not self.designated_unit():
                return False
            #3. 创建贴片天线
            if not self.design_antenna():
                return False
            #4. 定义解决方案设置
            if not self.solution_set():
                return False
            #5. 从hfss实例中查询信息
            if not self.find_info():
                return False
            #6. 运行分析
            if not self.run_analysis():
                return False
            #7. 后处理
            if not self.post_processing():
                return False

            print("=" * 80)
            print("仿真流程完全成功！")
            print("=" * 80)

            success = True

        except Exception as e:
            print(f"仿真流程异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 确保关闭 HFSS#完成
            #保存项目
            self.hfss.save_project()
            self.hfss.release_desktop()
            # Wait 3 seconds to allow AEDT to shut down before cleaning the temporary directory.
            time.sleep(3)

            #清理
            self.temp_folder.cleanup()

        return success

    def start_hfss_simulation(self):
        print("=" * 80)
        print("启动HFSS")

        self.aedtapp = self.hfss = ansys.aedt.core.Hfss(
            project=self.project_name,
            solution_type="Terminal",
            design="patch",
            non_graphical=self.NG_MODE,
            new_desktop=True,
            version=self.AEDT_VERSION,
        )
        # self.hfss.set_variable("ANSYSSYS_GPU", "1")
        # self.hfss.hfss_set_solver_options(solver_type="gpu")

        print("=" * 80)
        return True

    def designated_unit(self):
        # 指定单位
        print("=" * 80)
        print("指定单位")
        self.length_units = "mm"
        self.freq_units = "GHz"
        self.hfss.modeler.model_units = self.length_units
        print("=" * 80)
        return True

    def design_antenna(self):
        print("=" * 80)
        print("创建贴片天线")
        print("贴片天线由接地层、介质基板和顶部信号层组成，贴片天线位于顶部信号层上。")
        #创建贴片天线
        #贴片天线由接地层、介质基板和顶部信号层组成，贴片天线位于顶部信号层上。
        stackup = Stackup3D(self.hfss)
        # ground = stackup.add_ground_layer(
        #     "ground", material="copper", thickness=0.035, fill_material="air"
        # )
        # dielectric = stackup.add_dielectric_layer(
        #     "dielectric", thickness="0.5" + self.length_units, material="Duroid (tm)"
        # )
        # signal = stackup.add_signal_layer(
        #     "signal", material="copper", thickness=0.035, fill_material="air"
        # )
        # patch = signal.add_patch(
        #     patch_length=9.57, patch_width=9.25, patch_name="Patch", frequency=1e10
        # )

        ground = stackup.add_ground_layer(
            "ground", material="copper", thickness=self.antenna_params["ground_thickness"], fill_material="air"
        )
        dielectric = stackup.add_dielectric_layer(
            "dielectric", thickness="0.5" + self.length_units, material="Duroid (tm)"
        )
        signal = stackup.add_signal_layer(
            "signal", material="copper", thickness=self.antenna_params["signal_layer_thickness"], fill_material="air"
        )
        patch = signal.add_patch(
            patch_length=self.antenna_params["patch_length"], patch_width=self.antenna_params["patch_width"], patch_name=self.antenna_params["patch_name"], frequency=self.antenna_params["center_frequency"]
        )

        stackup.resize_around_element(patch)
        pad_length = [3, 3, 3, 3, 3, 3]  # Air bounding box buffer in mm.
        region = self.hfss.modeler.create_region(pad_length, is_percentage=False)
        self.hfss.assign_radiation_boundary_to_objects(region)

        patch.create_probe_port(ground, rel_x_offset=0.485)
        print("=" * 80)
        return True

    def solution_set(self):
        print("=" * 80)
        print("定义解决方案设置")
        print("频率扫描用于指定将计算散射参数的范围。")
        #定义解决方案设置
        #频率扫描用于指定将计算散射参数的范围。
        setup = self.hfss.create_setup(name="Setup1", setup_type="HFSSDriven", Frequency="10GHz")

        setup.create_frequency_sweep(
            # unit="GHz",
            # name="Sweep1",
            # start_frequency=8,
            # stop_frequency=12,
            # sweep_type="Interpolating",
            unit=self.antenna_params["unit"],
            name="Sweep1",
            start_frequency=self.antenna_params["start_frequency"],
            stop_frequency=self.antenna_params["stop_frequency"],
            sweep_type=self.antenna_params["sweep_type"],
            num_of_freq_points=self.antenna_params["num_of_freq_points"],
            save_fields=True,
        )

        # self.disc_sweep = setup.add_sweep(name="DiscreteSweep", sweep_type="Discrete",
        #                              RangeStart=self.antenna_params["start_frequency"],
        #                              RangeEnd=self.antenna_params["stop_frequency"],
        #                              RangeStep=self.antenna_params["freq_step"],
        #                              SaveFields=True)
        #
        # self.interp_sweep = setup.add_sweep(name="InterpolatingSweep", sweep_type="Interpolating",
        #                                RangeStart=self.antenna_params["start_frequency"],
        #                                RangeEnd=self.antenna_params["stop_frequency"],
        #                                SaveFields=False)
        print("保存工程")
        self.hfss.save_project()  # Save the project.
        print("=" * 80)
        return True

    def find_info(self):
        print("=" * 80)
        print("展示如何从hfss实例中查询信息")
        print("=" * 80)
        #展示如何从hfss实例中查询信息
        message = "We have created a patch antenna"
        message += "using PyAEDT.\n\nThe project file is "
        message += f"located at \n'{self.hfss.project_file}'.\n"
        message += f"\nThe HFSS design is named '{self.hfss.design_name}'\n"
        message += f"and is comprised of "
        message += f"{len(self.hfss.modeler.objects)} objects whose names are:\n\n"
        message += "".join([f"- '{o.name}'\n" for _, o in self.hfss.modeler.objects.items()])
        print(message)
        print("=" * 80)
        return True

    def run_analysis(self):
        print("=" * 80)
        print("运行分析")
        print("以下命令在HFSS中运行电磁分析。")
        #运行分析
        #以下命令在HFSS中运行电磁分析。
        self.hfss.analyze(cores=self.NUM_CORES)
        return True

    def post_processing(self):
        try:
            choice = input("请输入数字1-2")
            num = int(choice)

            if num == 1:
                input("请按回车键继续...")
                self.extract_s_parameters()
                return True
            elif num == 2:
                print("=" * 80)
                print("后处理")
                #后处理
#--------------------------------------------------S参数-------------------------------------------------
                print("S参数")
                plot_data = self.hfss.get_traces_for_plot()
                print(f"polt_data {plot_data}")
                report = self.hfss.post.create_report(plot_data)
                solution = report.get_solution_data()
                plt = solution.plot(solution.expressions)
                plt.show()
                print("=" * 80)

                input("请按回车键继续1...")
                print("远区场辐射图")
# --------------------------------------------------远区场辐射图-------------------------------------------------
                print("=" * 80)
                ffdata = self.hfss.get_antenna_data(
                    setup=self.hfss.nominal_adaptive,
                    # sphere="Infinite Sphere1",
                    link_to_hfss = True)
                print("对象类型：", type(ffdata))
                input("请按回车键继续11...")
                metadata_file = ffdata.metadata_file
                farfield_data = FfdSolutionData(input_file=metadata_file)
                farfield_data.plot_3d(quantity_format="dB10",
                                      output_file='./3D.png',
                                      show=False,
                                      )
                print("对象类型：", type(ffdata.farfield_data))

                data = ffdata.farfield_data.combine_farfield(phi_scan=0.0, theta_scan=0.0)
                # print(data)
                print("=" * 80)
                self.save_farfield_data_to_csv(data)

                # 替换为你的CSV文件路径
                csv_file_path = "farfield_data_zidian.csv"
                output_file_path = "data_dict_pandas.json"
                self.data_dict = self.readcsv_to_dict(csv_file_path)
                self.save_to_jsonfile(self.data_dict, output_file_path)

                # input("请按回车键继续2...")
                # # 步骤1：定义外部 PyVista 实例
                # # 创建一个 PyVista 渲染器（renderer）或 plotting 对象
                # external_pv = pv.Plotter()  # 最常用的实例类型，用于创建可视化场景
                # # （可选）给实例添加额外元素（如网格、坐标轴等，不影响传入，仅丰富场景）
                # external_pv.add_axes()  # 添加坐标轴
                # external_pv.set_background('white')  # 设置背景色
                #
                # ffdata.farfield_data.plot_3d(
                #     quantity="RealizedGain",
                #     quantity_format="dB10",
                #     show=False,
                #     # output_file="./antenna_3d.png",
                #     show_as_standalone=True,
                #     pyvista_object=external_pv,
                # )
                # external_pv.show()
                # img_data = external_pv.screenshot(return_img=True)
                # plt.imshow(img_data)
                # plt.show()
                # input("请按回车键继续111111...")
                # ffdata.farfield_data.plot_cut(
                #     quantity="RealizedGain",
                #     primary_sweep="theta",
                #     title="Elevation",
                #     quantity_format="dB10",
                # )
                input("请按回车键继续6...")
                exported_files = self.aedtapp.export_results(export_folder='./RESULT')
                # import pandas as pd
                # df = pd.DataFrame.from_records(exported_files)
                # df.to_csv("./RESULT/hfss.csv", index=False,encoding='utf-8')

                input("请按回车键继续1100...")
                print("=" * 80)
                return True
            else:
                print("=" * 80)
                print("输入错误！请输入1-2之间的数字")
                return False
        except ValueError:
            print("输入错误！请输入有效的数字")
    def save_farfield_data_to_csv(self, data):
        # 你的原始字典数据（此处省略，替换为你的实际字典变量名即可）
        # data = 你的字典变量

        # 1. 提取核心角度数组（Theta和Phi）
        theta_list = data["Theta"]  # 长度37的1D数组
        phi_list = data["Phi"]  # 长度73的1D数组
        n_theta = len(theta_list)
        n_phi = len(phi_list)

        # 2. 定义CSV表头（角度列 + 所有物理量列，复数拆分为“实部/虚部”）
        header = ["Theta(deg)", "Phi(deg)"]  # 角度坐标列

        # 遍历字典，为每个物理量生成表头（复数拆分为实部、虚部，实数直接用原键名）
        for key in data.keys():
            # 跳过非数据类键（Theta/Phi是坐标，nTheta/nPhi是长度，无需重复）
            if key in ["Theta", "Phi", "nTheta", "nPhi"]:
                continue
            # 判断该键对应的数据是否为复数类型
            is_complex = np.iscomplexobj(data[key])
            if is_complex:
                header.append(f"{key}_Real")  # 复数实部列
                header.append(f"{key}_Imag")  # 复数虚部列
            else:
                header.append(key)  # 实数直接用原键名

        # 3. 生成CSV行数据（遍历所有Theta-Ph Phi组合）
        csv_rows = []
        for i in range(n_theta):
            for j in range(n_phi):
                row = []
                # 先添加当前角度坐标（Theta和Phi）
                row.append(theta_list[i])
                row.append(phi_list[j])
                # 再添加该角度下所有物理量的值
                for key in data.keys():
                    if key in ["Theta", "Phi", "nTheta", "nPhi"]:
                        continue
                    val = data[key][i, j]  # 提取二维数组中(i,j)位置的值
                    # 复数拆分为实部和虚部，实数直接添加
                    if np.iscomplexobj(val):
                        row.append(val.real)
                        row.append(val.imag)
                    else:
                        row.append(val)
                csv_rows.append(row)

        # 4. 保存到CSV文件
        with open("./farfield_data_zidian.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 写入表头
            writer.writerows(csv_rows)  # 写入所有数据行

        print(f"CSV文件已保存！共 {len(csv_rows)} 行数据，{len(header)} 列参数。")

    def readcsv_to_dict(self, csv_file_path):
        # 读取CSV，仅取前2行（表头+第二行数据）
        df = pd.read_csv(csv_file_path, nrows=1)  # nrows=1 表示仅读取1行数据（第二行）
        # 转换为字典（orient="records" 按行转换，取第一个元素即为目标字典）
        data_dict = df.to_dict(orient="records")[0]
        return data_dict

    def save_to_jsonfile(self, data_dict, output_file_path):
        # 保存字典到文件
        import json
        # 追加字典到文件末尾（每行一个JSON）
        with open(output_file_path, "a", encoding="utf-8") as f:
            # 字典转为JSON字符串，添加换行符（确保每行一个字典）
            json.dump(data_dict, f, ensure_ascii=False)
            f.write("\n")  # 换行，便于下次追加和读取

        print(f"字典已追加到 {output_file_path} 文件末尾")

        # 验证结果（可选）
        print("\n文件当前内容：")
        with open(output_file_path, "r", encoding="utf-8") as f:
            print(f.read())

        return data_dict
    def extract_s_parameters(self):
        spar_plot = self.hfss.create_scattering()


def main():
    print("=" * 80)
    print("python 调用 HFSS计算探针馈电贴片天线开始")
    print("=" * 80)

    # 定义常量
    AEDT_VERSION = "2025R1"
    NUM_CORES = 4
    NG_MODE = False  # Open AEDT UI when it is launched.

    # 创建临时目录
    temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")

    simulator = AdvancedHFSSEntennaSimulator(temp_folder, NG_MODE, AEDT_VERSION, NUM_CORES)

    success = simulator.run_full_hfss_simulation()

    if success:
        print("\n" + "="*80)
        print("天线仿真成功完成！")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("天线仿真失败")
        print("="*80)

if __name__ == "__main__":
    main()