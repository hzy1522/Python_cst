import os
import tempfile
import time

import ansys.aedt.core
from ansys.aedt.core.modeler.advanced_cad.stackup_3d import Stackup3D

#定义常量
AEDT_VERSION = "2025R1"
NUM_CORES = 4
NG_MODE = False  # Open AEDT UI when it is launched.

#创建临时目录
temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")

#启动HFSS
project_name = os.path.join(temp_folder.name, "patch.aedt")
hfss = ansys.aedt.core.Hfss(
    project=project_name,
    solution_type="Terminal",
    design="patch",
    non_graphical=NG_MODE,
    new_desktop=True,
    version=AEDT_VERSION,
)

#指定单位
length_units = "mm"
freq_units = "GHz"
hfss.modeler.model_units = length_units

#创建贴片天线
#贴片天线由接地层、介质基板和顶部信号层组成，贴片天线位于顶部信号层上。
stackup = Stackup3D(hfss)
ground = stackup.add_ground_layer(
    "ground", material="copper", thickness=0.035, fill_material="air"
)
dielectric = stackup.add_dielectric_layer(
    "dielectric", thickness="0.5" + length_units, material="Duroid (tm)"
)
signal = stackup.add_signal_layer(
    "signal", material="copper", thickness=0.035, fill_material="air"
)
patch = signal.add_patch(
    patch_length=9.57, patch_width=9.25, patch_name="Patch", frequency=1e10
)

stackup.resize_around_element(patch)
pad_length = [3, 3, 3, 3, 3, 3]  # Air bounding box buffer in mm.
region = hfss.modeler.create_region(pad_length, is_percentage=False)
hfss.assign_radiation_boundary_to_objects(region)

patch.create_probe_port(ground, rel_x_offset=0.485)

#定义解决方案设置
#频率扫描用于指定将计算散射参数的范围。
setup = hfss.create_setup(name="Setup1", setup_type="HFSSDriven", Frequency="10GHz")

setup.create_frequency_sweep(
    unit="GHz",
    name="Sweep1",
    start_frequency=8,
    stop_frequency=12,
    sweep_type="Interpolating",
)

hfss.save_project()  # Save the project.

#展示如何从hfss实例中查询信息
message = "We have created a patch antenna"
message += "using PyAEDT.\n\nThe project file is "
message += f"located at \n'{hfss.project_file}'.\n"
message += f"\nThe HFSS design is named '{hfss.design_name}'\n"
message += f"and is comprised of "
message += f"{len(hfss.modeler.objects)} objects whose names are:\n\n"
message += "".join([f"- '{o.name}'\n" for _, o in hfss.modeler.objects.items()])
print(message)

#运行分析
#以下命令在HFSS中运行电磁分析。
hfss.analyze(cores=NUM_CORES)

#后处理
plot_data = hfss.get_traces_for_plot()
report = hfss.post.create_report(plot_data)
solution = report.get_solution_data()
plt = solution.plot(solution.expressions)

#完成
#保存项目
hfss.save_project()
hfss.release_desktop()
# Wait 3 seconds to allow AEDT to shut down before cleaning the temporary directory.
time.sleep(3)

#清理
temp_folder.cleanup()