import csv
import warnings
from datetime import datetime
import glob
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import ansys.aedt.core
import pyvista as pv
import pandas as pd
import json
import re

from pyvista import examples
pv.set_jupyter_backend("trame")
from ansys.aedt.core.modeler.advanced_cad.stackup_3d import Stackup3D
from ansys.aedt.core.visualization.advanced.farfield_visualization import FfdSolutionData
from django.contrib.messages import success
from ansys.aedt.core import Hfss
import matplotlib.pyplot as plt

from typing import List, Dict, Union, Optional, Any


class AdvancedHFSSEntennaSimulator:

    def __init__(self,
                 temp_folder=None,
                 NG_MODE=None,
                 AEDT_VERSION=None,
                 NUM_CORES=None,
                 antenna_params = None,):

        self.temp_folder = temp_folder
        self.NG_MODE = NG_MODE
        self.AEDT_VERSION = AEDT_VERSION
        self.NUM_CORES = NUM_CORES
        self.project_name = os.path.join(self.temp_folder.name, "patch.aedt")
        self.gain_value = None
        self.s_parms_min = None
        self.fre_value = None
        self.antenna_params = antenna_params
        # self.antenna_params = {
        #     "unit": "GHz", #单位设置
        #     "start_frequency": 8,  # 起始工作频率 (GHz)
        #     "stop_frequency": 12,  #截止频率
        #     "center_frequency": 10,  #中心频率
        #     "sweep_type": "Fast", #扫描频率设置
        #     "ground_thickness": 0.035,  # 地板厚度 (mm)
        #     "signal_layer_thickness": 0.035, #信号线厚度(mm)
        #     "patch_length": 9.57, # 贴片长度(mm)
        #     "patch_width": 9.25, #
        #     "patch_name": "Patch",
        #     "freq_step" : "2GHz",
        #     "num_of_freq_points": 101,
        # }
        self.disc_sweep = None
        self.interp_sweep = None
    def run_full_hfss_simulation(self, train_model):
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
            if not self.post_processing(train_model):
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

        return success, self.fre_value, self.gain_value, self.s_parms_min

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

    def post_processing(self, train_model):
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

        # input("请按回车键继续1...")
        print("远区场辐射图")
# --------------------------------------------------远区场辐射图-------------------------------------------------
        print("=" * 80)
        ffdata = self.hfss.get_antenna_data(
            setup=self.hfss.nominal_adaptive,
            # sphere="Infinite Sphere1",
            link_to_hfss = True)
        # print("对象类型：", type(ffdata))
        # input("请按回车键继续11...")
        metadata_file = ffdata.metadata_file
        farfield_data = FfdSolutionData(input_file=metadata_file)
        farfield_data.plot_3d(quantity_format="dB10",
                              output_file='./3D.png',
                              show=False,
                              )
        # print("对象类型：", type(ffdata.farfield_data))
# --------------------------------------------------保存原始数据到csv-------------------------------------------------
        data = ffdata.farfield_data.combine_farfield(phi_scan=0.0, theta_scan=0.0)
        # print(data)
        print("=" * 80)

        if train_model:
            csv_file_path_base = "./RESULT_Farfile/farfield_data_zidian.csv"
            output_path_base = "./RESULT/data_dict_pandas.csv"
        else:
            csv_file_path_base = "./TEST_RESULT/farfield_data_zidian.csv"
            output_path_base = "./TEST_RESULT/data_dict_pandas.csv"

        target_pattern = "RealizedGain"

        csv_file_path, output_path = self.add_timestamp_to_filename(csv_file_path_base, output_path_base)
        self.save_farfield_data_to_csv(data, csv_file_path)

# --------------------------------------------------提取增益最大结果并保存-------------------------------------------------
        # 替换为你的CSV文件路径


        extreme_data = self.find_csv_extreme_rows(
            csv_file_path=csv_file_path,
            target_header_pattern=target_pattern,
            extreme_type="max")
        # 3. 打印结果（可选）
        print(f"共找到 {len(extreme_data)} 个极值行：")
        for i, row in enumerate(extreme_data, 1):
            print(f"\n=== 第{i}个极值行 ===")
            print(f"极值列：{row['_极值列']} | 类型：{row['_极值类型']} | 数值：{row['_极值数值']:.6f}")
            print("核心数据（前5列）：")
            count = 0
            for key, value in row.items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
                    if count == 14:
                        self.gain_value = value
                    count += 1
                    if count >= 15:
                        break

        # 4. 保存结果
        # self.save_extreme_dicts(
        #     extreme_dicts=extreme_data,
        #     output_file=output_path,
        #     append=True,
        #     add_separator=True
        # )
        self.save_extreme_dicts_to_csv(
            extreme_dicts=self.convert_any_dict_to_list_dict(self.antenna_params),
            output_file=output_path,
            append=True,
            append_by_column=False
        )

        self.save_extreme_dicts_to_csv(
            extreme_dicts=extreme_data,
            output_file=output_path,
            append=True,
            append_by_column=True
        )

        # input("请按回车键继续6...")
        exported_files = self.aedtapp.export_results(export_folder='./RESULT_S')
        print(exported_files[0])
        # input("请按回车键继续6222...")


        # input("请按回车键继续77776...")
# --------------------------------------------------提取S最小结果并保存-------------------------------------------------
        csv_path = glob.glob("RESULT_S/patch_patch_Plot_*.csv")
        print(csv_path[0])
        # min_row_data = self.find_min_in_second_column("./RESULT_S/patch_patch_Plot_36L36E.csv",
        #                                               encoding="utf-8",)
        min_row_data = self.find_min_in_second_column(csv_path[0],encoding="utf-8", )
        # 打印结果（格式化输出，可读性强）
        if min_row_data:
            print("\n最小值所在行的完整数据（list[dict]格式）：")
            print(f"数据类型：{type(min_row_data)}")  # <class 'list'>
            print(f"列表长度：{len(min_row_data)}")  # 1
            print(f"列表元素类型：{type(min_row_data[0])}")  # <class 'dict'>

            # 格式化打印字典内容
            print("\n详细数据：")
            for key, value in min_row_data[0].items():
                print(f"  {key}: {value}（类型：{type(value).__name__}）")
                if key == 'Freq [GHz]':
                    self.fre_value = value
                elif key == '_最小值':
                    self.s_parms_min = value

#         # 替换为你的CSV文件路径
#         csv_file_path_list =  glob.glob("./RESULT_S/patch_patch_Plot_*.csv")
#         csv_file_path = csv_file_path_list[0]
#         print(csv_file_path)
#         extreme_data = self.find_csv_extreme_rows(
#                         csv_file_path=csv_file_path,
#                         target_header_pattern="dB(S(Probe_Port_T1,Probe_Port_T1))",
#                         extreme_type="min")
#
#         # 3. 打印结果（可选）
#         print(f"共找到 {len(extreme_data)} 个极值行：")
#         for i, row in enumerate(extreme_data, 1):
#             print(f"\n=== 第{i}个极值行 ===")
#             print(f"极值列：{row['_极值列']} | 类型：{row['_极值类型']} | 数值：{row['_极值数值']:.6f}")
#             print("核心数据（前5列）：")
#             count = 0
#             for key, value in row.items():
#                 if not key.startswith("_"):
#                     print(f"  {key}: {value}")
#                     count += 1
#                     if count >= 2:
#                         break

        # 4. 保存结果
        # self.save_extreme_dicts(
        #     extreme_dicts=min_row_data,
        #     output_file=output_path,
        #     append=True,
        #     add_separator=True
        # )
        self.save_extreme_dicts_to_csv(
            extreme_dicts=min_row_data,
            output_file=output_path,
            append=True,
            append_by_column=True
        )

        # input("请按回车键继续1100...")
        print("=" * 80)
        return True
    def save_farfield_data_to_csv(self, data, file_name):
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
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 写入表头
            writer.writerows(csv_rows)  # 写入所有数据行

        print(f"CSV文件已保存！共 {len(csv_rows)} 行数据，{len(header)} 列参数。")

 # ----------------------------------------------不在使用 begin-------------------------------------------------
 #    def readcsv_to_dict(self, csv_file_path):
 #        # 读取CSV，仅取前2行（表头+第二行数据）
 #        df = pd.read_csv(csv_file_path, nrows=1)  # nrows=1 表示仅读取1行数据（第二行）
 #        # 转换为字典（orient="records" 按行转换，取第一个元素即为目标字典）
 #        data_dict = df.to_dict(orient="records")[0]
 #        return data_dict
 #
 #    def save_to_jsonfile(self, data_dict, output_file_path):
 #        # 保存字典到文件
 #        # 追加字典到文件末尾（每行一个JSON）
 #        with open(output_file_path, "a", encoding="utf-8") as f:
 #            # 字典转为JSON字符串，添加换行符（确保每行一个字典）
 #            json.dump(data_dict, f, ensure_ascii=False)
 #            f.write("\n")  # 换行，便于下次追加和读取
 #
 #        print(f"字典已追加到 {output_file_path} 文件末尾")
 #
 #        # 验证结果（可选）
 #        print("\n文件当前内容：")
 #        with open(output_file_path, "r", encoding="utf-8") as f:
 #            print(f.read())
 #
 #        return data_dict
# ----------------------------------------------不在使用 end-------------------------------------------------
    def find_min_in_second_column(self, csv_file_path: str, encoding: str = "utf-8") -> Optional[List[Dict[str, Union[str, float]]]]:

        header: List[str] = []

        min_value: float = float("inf")
        min_row_dict: Optional[Dict[str, Union[str, float]]] = None

        try:
            # 验证输入是合法路径（避免再次传入非路径对象）
            if not isinstance(csv_file_path, (str, bytes, os.PathLike)):
                raise TypeError(f"csv_file_path 必须是字符串路径，当前类型：{type(csv_file_path).__name__}")

            with open(csv_file_path, "r", encoding=encoding) as f:
                reader = csv.reader(f)
                header = next(reader)
                if len(header) < 2:
                    print("错误：CSV文件至少需要2列数据")
                    return None

                for row_idx, row in enumerate(reader, 2):
                    if not row or len(row) < 2:
                        print(f"警告：第{row_idx}行数据不完整，已跳过")
                        continue

                    try:
                        second_col_value = float(row[1])
                    except ValueError:
                        print(f"警告：第{row_idx}行第二列数据 '{row[1]}' 不是有效数值，已跳过")
                        continue

                    if second_col_value < min_value:
                        min_value = second_col_value
                        row_dict = {}
                        for col_name, value in zip(header, row):
                            try:
                                row_dict[col_name] = float(value)
                            except ValueError:
                                row_dict[col_name] = value
                        min_row_dict = row_dict

            if min_row_dict is None:
                print("未找到第二列的有效数值数据")
                return None
            else:
                min_row_dict["_最小值"] = min_value
                min_row_dict["_最小值所在列名"] = header[1]
                print(f"找到第二列最小值：{min_value}（列名：{header[1]}）")
                return [min_row_dict]

        except FileNotFoundError:
            raise Exception(f"未找到CSV文件：{csv_file_path}")
        except Exception as e:
            raise Exception(f"读取CSV文件失败：{str(e)}")


    def find_csv_extreme_rows(self,
                                csv_file_path: str,
                                target_header_pattern: str,
                                extreme_type: str) -> List[Dict[str, Union[str, float]]]:
        print("find_csv_extreme_rows")
        # 步骤1：读取CSV并转换数据类型
        header: List[str] = []
        data_rows: List[Dict[str, Union[str, float]]] = []

        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # 读取表头
            header = next(reader)
            # 读取数据行并转换数值类型
            for row in reader:
                row_dict = {}
                for col_name, value in zip(header, row):
                    try:
                        row_dict[col_name] = float(value)
                    except ValueError:
                        row_dict[col_name] = value  # 非数值保留字符串
                data_rows.append(row_dict)

        # 校验数据行
        if not data_rows:
            raise ValueError("CSV文件中未包含有效数据行")

        # 步骤2：通配符匹配目标表头
        pattern = re.compile(f"^{target_header_pattern.replace('*', '.*')}$", re.IGNORECASE)
        target_headers = [col for col in header if pattern.match(col)]

        if not target_headers:
            raise ValueError(f"未找到匹配模式 '{target_header_pattern}' 的表头列")

        # 步骤3：计算每个目标列的极值行
        extreme_rows = []
        for target_col in target_headers:
            # 过滤可比较的数值行
            valid_rows = [row for row in data_rows if isinstance(row[target_col], float)]
            if not valid_rows:
                print(f"警告：列 '{target_col}' 无有效数值数据，已跳过")
                continue

            # 查找极值行
            if extreme_type == "max":
                extreme_row = max(valid_rows, key=lambda x: x[target_col])
            elif extreme_type == "min":
                extreme_row = min(valid_rows, key=lambda x: x[target_col])
            else:
                raise ValueError("extreme_type 必须为 'max' 或 'min'")

            # 添加辅助标识字段
            extreme_row.update({
                "_极值列": target_col,
                "_极值类型": extreme_type,
                "_极值数值": extreme_row[target_col]
            })
            extreme_rows.append(extreme_row)
        print("find_csv_extreme_rows end")
        return extreme_rows

    def save_extreme_dicts(self,
            extreme_dicts: List[Dict[str, Union[str, float]]],
            output_file: str,
            append: bool = True,
            add_separator: bool = True
    ) -> None:
        """
        将极值行字典列表保存到文件（JSON格式，支持追加）

        参数：
            extreme_dicts: List[Dict] - 由find_csv_extreme_rows返回的极值行字典列表
            output_file: str - 输出文件路径（如"extreme_results.json"）
            append: bool - 是否追加模式（True=追加，False=覆盖），默认True
            add_separator: bool - 是否添加字典分隔符（便于阅读），默认True

        异常：
            IOError: 文件写入失败
        """
        if not extreme_dicts:
            print("警告：无有效极值数据可保存")
            return

        mode = "a" if append else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for idx, data_dict in enumerate(extreme_dicts):
                # 写入字典（带缩进，支持中文）
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
                # 添加分隔符（最后一个字典后不添加）
                if add_separator and idx != len(extreme_dicts) - 1:
                    f.write("\n" + "-" * 60 + "\n\n")
                else:
                    f.write("\n")  # 换行保证后续追加格式正确

        print(f"数据已{'追加' if append else '保存'}到：{output_file}")

    def save_extreme_dicts_to_csv(self,
                                  extreme_dicts: List[Dict[str, Union[str, float]]],
                                  output_file: str,
                                  append: bool = True,
                                  encoding: str = "utf-8-sig",
                                  remove_braces: bool = True,  # 去除字典大括号（转为纯值）
                                  keep_header: bool = True,  # 是否保留表头
                                  append_by_column: bool = False  # 新增：是否按列追加（False=按行，True=按列）
                                  ) -> None:
        """
        将极值行字典列表保存到CSV文件（支持按行/按列追加，去除大括号）

        参数：
            extreme_dicts: List[Dict] - 极值行字典列表（每个字典对应一条记录）
            output_file: str - 输出CSV文件路径
            append: bool - 是否追加模式（True=追加，False=覆盖），默认True
            encoding: str - 文件编码（默认utf-8-sig，兼容Excel）
            remove_braces: bool - 是否去除大括号（转为纯值列表），默认True
            keep_header: bool - 是否保留表头（字段名），默认True
            append_by_column: bool - 追加方式（False=按行追加，True=按列追加），默认False

        异常：
            IOError: 文件写入失败
            ValueError: 按列追加时新旧数据行数不匹配
        """
        # 校验输入数据
        if not extreme_dicts:
            print("警告：极值字典列表为空，未写入任何数据")
            return

        # 提取所有字段（固定顺序）
        all_fields = set()
        for row_dict in extreme_dicts:
            if not isinstance(row_dict, dict):
                raise TypeError(f"极值列表元素必须是字典，当前存在类型：{type(row_dict).__name__}")
            all_fields.update(row_dict.keys())
        csv_headers = sorted(all_fields)
        new_data_rows = len(extreme_dicts)  # 新数据的行数（按行追加时=记录数，按列追加时=每条记录的列数）

        # 处理新数据：转为纯值列表（去除大括号）
        new_data = []
        if remove_braces:
            # 按表头顺序提取值，每个字典→一个值列表
            for row_dict in extreme_dicts:
                row_values = [row_dict.get(field, "") for field in csv_headers]
                new_data.append(row_values)
        else:
            new_data = extreme_dicts  # 保留字典格式（不推荐，仅兼容）

        # 按列追加模式：需要读取原有数据，合并后重新写入
        if append_by_column and append and Path(output_file).exists():
            # 1. 读取原有CSV数据
            old_data = []
            old_headers = []
            with open(output_file, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                # 读取表头（若存在）
                if keep_header:
                    old_headers = next(reader, [])
                # 读取数据行
                old_data = [row for row in reader]

            # 2. 校验新旧数据行数匹配（按列追加时，新数据的行数=旧数据的行数）
            if len(old_data) != new_data_rows:
                raise ValueError(
                    f"按列追加失败：新旧数据行数不匹配（原有数据行数：{len(old_data)}，新数据行数：{new_data_rows}）"
                )

            # 3. 合并表头（旧表头 + 新表头）
            merged_headers = old_headers + csv_headers if keep_header else []
            # 4. 合并数据（每行旧数据 + 对应行新数据）
            merged_data = []
            for old_row, new_row in zip(old_data, new_data):
                merged_row = old_row + new_row
                merged_data.append(merged_row)

            # 5. 覆盖写入合并后的数据（按列追加本质是合并后重写）
            with open(output_file, "w", encoding=encoding, newline="") as f:
                writer = csv.writer(f)
                if keep_header and merged_headers:
                    writer.writerow(merged_headers)
                writer.writerows(merged_data)

            print(f"按列追加成功！")
            print(f"- 新增列数：{len(csv_headers)} 列（{', '.join(csv_headers)}）")
            print(
                f"- 合并后总行数：{len(merged_data)} 行，总列数：{len(merged_headers) if merged_headers else len(merged_data[0])} 列")

        # 按行追加/覆盖模式（原有逻辑优化）
        else:
            file_mode = "a" if append else "w"
            file_exists = Path(output_file).exists()

            try:
                with open(output_file, file_mode, encoding=encoding, newline="") as f:
                    writer = csv.writer(f)
                    # 写表头（仅当需要保留且文件不存在/覆盖模式时）
                    if keep_header and (not file_exists or not append):
                        writer.writerow(csv_headers)

                    # 逐行写入新数据
                    writer.writerows(new_data)

                print(f"{'按行追加' if append else '覆盖'}成功！")
                print(f"- 写入行数：{len(new_data)} 行，列数：{len(csv_headers)} 列")
                print(f"- 表头：{'保留' if keep_header else '不保留'}，大括号：{'已去除' if remove_braces else '保留'}")

            except Exception as e:
                raise IOError(f"文件写入失败：{str(e)}") from e

    def add_timestamp_to_filename(self,
            base_filename: str,
            base_filename2: str,
            time_format: str = "%Y%m%d_%H%M%S",  # 时间戳格式：年-月-日_时-分-秒
            separator: str = "_"  # 基础名与时间戳的分隔符
    ) -> tuple[str, str]:
        """
        在文件名后添加时间戳（插入到基础名和后缀之间）

        参数：
            base_filename: 原始文件名（可带路径，如"result.csv"或"./data/extreme.json"）
            time_format: 时间戳格式（默认：%Y%m%d_%H%M%S → 20241025_143022）
            separator: 基础名与时间戳的连接符（默认下划线"_"）

        返回：
            带时间戳的新文件名（如"result_20241025_143022.csv"）
        """
        # 1. 拆分文件名：基础名 + 后缀（处理带路径的情况）
        # os.path.splitext 会拆分最后一个"."后的后缀（如"archive.tar.gz" → ("archive.tar", ".gz")）
        # 新增：严格校验输入类型（提前拦截错误）
        if not isinstance(base_filename, (str, bytes, os.PathLike)):
            raise TypeError(
                f"base_filename 必须是字符串路径，当前类型：{type(base_filename).__name__}，值：{base_filename}"
            )

        # 拆分路径和文件名（原逻辑不变）
        file_dir, file_name = os.path.split(base_filename)
        file_dir2, file_name2 = os.path.split(base_filename2)

        base_name, ext = os.path.splitext(file_name)
        base_name2, ext2 = os.path.splitext(file_name2)

        # 生成时间戳并拼接新文件名
        timestamp = datetime.now().strftime(time_format)

        new_file_name = f"{base_name}{separator}{timestamp}{ext}"
        new_file_name2 = f"{base_name2}{separator}{timestamp}{ext2}"

        new_filename = os.path.join(file_dir, new_file_name)
        new_filename2 = os.path.join(file_dir2, new_file_name2)

        return new_filename, new_filename2

    def extract_s_parameters(self):
        spar_plot = self.hfss.create_scattering()

    def find_csv_extreme_rows_all(self,
            csv_file_path: str = None,
            header_pattern:str = None,
            extreme_type:str = None,
            output_file: str = None):

        csv_files = glob.glob(csv_file_path)
        for file in csv_files:
            try:
                results = self.find_csv_extreme_rows(file, header_pattern, extreme_type)
                self.save_extreme_dicts_to_csv(results, output_file, append=True)
            except Exception as e:
                print(f"处理文件 {file} 失败：{e}")

    def convert_any_dict_to_list_dict(self,
            input_dict: Dict[str | Any, str | int | float | Any],
            key_convert_func: Optional[callable] = None,
            value_convert_func: Optional[callable] = None,
            ignore_invalid: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """
        将 dict[str | Any, str | int | float | Any] 转换为 list[dict[str, str | float]]

        参数：
            input_dict: 输入字典（键可能含 Any 类型，值可能含 int/Any 类型）
            key_convert_func: 自定义键转换函数（输入任意类型键，返回 str 类型；未指定则用默认逻辑）
            value_convert_func: 自定义值转换函数（输入任意类型值，返回 str/float；未指定则用默认逻辑）
            ignore_invalid: 是否忽略无效的键/值（True=跳过，False=抛出异常）

        返回：
            List[Dict[str, str | float]]: 列表套单个字典，键为 str，值为 str 或 float
        """
        # 初始化输出字典
        output_dict: Dict[str, Union[str, float]] = {}

        # 处理每个键值对
        for key, value in input_dict.items():
            # ---------------------- 处理键：确保为 str 类型 ----------------------
            try:
                if key_convert_func is not None:
                    # 使用自定义键转换函数
                    str_key = key_convert_func(key)
                    if not isinstance(str_key, str):
                        raise TypeError(f"自定义键转换函数返回类型必须是 str，当前为 {type(str_key).__name__}")
                else:
                    # 默认转换逻辑：直接转 str，无法转换则视为无效
                    if isinstance(key, (str, int, float, bool)):
                        str_key = str(key)
                    else:
                        # 对复杂类型（如对象），尝试取 __name__ 或 __str__
                        if hasattr(key, "__name__"):
                            str_key = key.__name__
                        elif hasattr(key, "__str__"):
                            str_key = str(key)
                        else:
                            raise ValueError(f"键 {key}（类型：{type(key).__name__}）无法转为 str")
            except Exception as e:
                msg = f"键 {key} 转换失败：{str(e)}"
                if ignore_invalid:
                    warnings.warn(msg)
                    continue
                else:
                    raise TypeError(msg) from e

            # ---------------------- 处理值：确保为 str 或 float 类型 ----------------------
            try:
                if value_convert_func is not None:
                    # 使用自定义值转换函数
                    valid_value = value_convert_func(value)
                    if not isinstance(valid_value, (str, float)):
                        raise TypeError(f"自定义值转换函数返回类型必须是 str/float，当前为 {type(valid_value).__name__}")
                else:
                    # 默认转换逻辑：int→float，其他类型尝试转 float/str
                    if isinstance(value, int):
                        valid_value = float(value)  # int 自动转 float
                    elif isinstance(value, float):
                        valid_value = value  # 直接保留 float
                    elif isinstance(value, str):
                        valid_value = value  # 直接保留 str
                    elif value is None:
                        valid_value = ""  # None 转为空字符串
                    else:
                        # 尝试转 float，失败则转 str
                        try:
                            valid_value = float(value)
                        except (ValueError, TypeError):
                            valid_value = str(value)
            except Exception as e:
                msg = f"值 {value}（键：{str_key}）转换失败：{str(e)}"
                if ignore_invalid:
                    warnings.warn(msg)
                    continue
                else:
                    raise TypeError(msg) from e

            # 加入输出字典（避免重复键，后出现的覆盖前一个）
            output_dict[str_key] = valid_value

        # 包装为 list[dict] 格式返回
        return [output_dict]

def calculate_from_hfss(antenna_params, train_model):
    print("=" * 80)
    print("python 调用 HFSS计算探针馈电贴片天线开始")
    print("=" * 80)

    # 定义常量
    AEDT_VERSION = "2025R1"
    NUM_CORES = 4
    # NG_MODE = False  # Open AEDT UI when it is launched.
    NG_MODE = True  # Not Open AEDT UI when it is launched.

    # 创建临时目录
    temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")

    simulator = AdvancedHFSSEntennaSimulator(temp_folder, NG_MODE, AEDT_VERSION, NUM_CORES, antenna_params)

    success, fre_value, gain_value, s_prams_min = simulator.run_full_hfss_simulation(train_model)

    if success:
        print("\n" + "="*80)
        print("天线仿真成功完成！")
        print("="*80)
        return success, fre_value, gain_value, s_prams_min
    else:
        print("\n" + "="*80)
        print("天线仿真失败")
        print("="*80)
        return False

def main():
    print("=" * 80)
    print("python 调用 HFSS计算探针馈电贴片天线开始")
    print("=" * 80)

    # 定义常量
    AEDT_VERSION = "2025R1"
    NUM_CORES = 4
    NG_MODE = False  # Open AEDT UI when it is launched.

    antenna_params = {
        "unit": "GHz",  # 单位设置
        "start_frequency": 8,  # 起始工作频率 (GHz)
        "stop_frequency": 12,  # 截止频率
        "center_frequency": 10,  # 中心频率
        "sweep_type": "Fast",  # 扫描频率设置
        "ground_thickness": 0.035,  # 地板厚度 (mm)
        "signal_layer_thickness": 0.035,  # 信号线厚度(mm)
        "patch_length": 9.57,  # 贴片长度(mm)
        "patch_width": 9.25,  #
        "patch_name": "Patch",
        "freq_step": "2GHz",
        "num_of_freq_points": 101,
    }
    # 创建临时目录
    temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")

    simulator = AdvancedHFSSEntennaSimulator(temp_folder, NG_MODE, AEDT_VERSION, NUM_CORES, antenna_params)

    success, fre_value, gain_value, s_prams_min  = simulator.run_full_hfss_simulation()

    if success:
        print("\n" + "="*80)
        print("天线仿真成功完成！")
        print(f"fre={fre_value}GHz, gain_value={gain_value}dB, s_prams_min=", s_prams_min)
        print("="*80)
    else:
        print("\n" + "="*80)
        print("天线仿真失败")
        print("="*80)

if __name__ == "__main__":
    main()