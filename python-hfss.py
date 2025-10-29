import os
import tempfile
import time

import ansys.aedt.core
import pyvista as pv
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
            "sweep_type": "Interpolating", #扫描频率设置
            "ground_thickness": 0.035,  # 地板厚度 (mm)
            "signal_layer_thickness": 0.035, #信号线厚度(mm)
            "patch_length": 9.57, # 贴片长度(mm)
            "patch_width": 9.25, #
            "patch_name": "Patch",
            "freq_step" : "0.5GHz",
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

        self.hfss = ansys.aedt.core.Hfss(
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
        )
        # disc_sweep = setup.add_sweep(name="DiscreteSweep", sweep_type="Discrete",
        #                              RangeStart=self.antenna_params["start_frequency"],
        #                              RangeEnd=self.antenna_params["stop_frequency"],
        #                              RangeStep=self.antenna_params["freq_step"],
        #                              SaveFields=True)

        # interp_sweep = setup.add_sweep(name="InterpolatingSweep", sweep_type="Interpolating",
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
        print("=" * 80)
        print("后处理")
        #后处理
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

        # variations = self.hfss.available_variations.nominal_values
        # variations["Theta"] = ["All"]
        # variations["Phi"] = ["All"]
        # variations["Freq"] = ["10GHz"]
        # elevation_ffd_plot = self.hfss.post.create_report(
        #                     expressions = "db(GainTotal)",
        #                     setup_sweep_name = self.hfss.nominal_adaptive,
        #                     variations = variations,
        #                     primary_sweep_variable = "Theta",
        #                     context="Elevation",# Far-field setup is pre-defined.
        #                     report_category = "Far Fields",
        #                     plot_type = "Radiation Pattern",
        #                     plot_name="Elevation Gain (dB)"
        #                     )
        # elevation_ffd_plot.children["Legend"].properties["Show Trace Name"] = False
        # elevation_ffd_plot.children["Legend"].properties["Show Solution Name"] = False
        #
        # report_3d = self.hfss.post.reports_by_category.far_field("db(RealizedGainTheta)", "Setup : LastAdaptive", "3D_Sphere")
        #
        # report_3d.report_type = "3D Polar Plot"
        # report_3d.create(name="Realized Gain (dB)")
        #
        # report_3d_data = report_3d.get_solution_data()
        # new_plot = report_3d_data.plot_3d()
        # new_plot.show()


        print("=" * 80)
        ffdata = self.hfss.get_antenna_data(
            setup=self.hfss.nominal_adaptive,
            sphere="Infinite Sphere1",
            link_to_hfss = True)
        input("请按回车键继续11...")
        ffdata.farfield_data.plot_cut(primary_sweep="theta", theta=0)
        ffdata.farfield_data.plot_cut(
            quantity="RealizedGain",
            primary_sweep="phi",
            title="Elevation",
            quantity_format="dB10",
        )
        input("请按回车键继续2...")

        # 步骤1：定义外部 PyVista 实例
        # 创建一个 PyVista 渲染器（renderer）或 plotting 对象
        external_pv = pv.Plotter()  # 最常用的实例类型，用于创建可视化场景
        # （可选）给实例添加额外元素（如网格、坐标轴等，不影响传入，仅丰富场景）
        external_pv.add_axes()  # 添加坐标轴
        external_pv.set_background('white')  # 设置背景色

        ffdata.farfield_data.plot_3d(
            quantity="RealizedGain",
            quantity_format="dB10",
            show=False,
            # show_as_standalone=True,
            pyvista_object=external_pv,
        )
        # img_data = external_pv.screenshot(return_img=True)
        # plt.imshow(img_data)
        # plt.show()
        external_pv.show()
        input("请按回车键继续4...")

        print("=" * 80)

        return True



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