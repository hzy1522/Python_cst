import win32com.client
import datetime
import os
import csv

def append_array_data_to_csv(csv_file_path, array_matrix):
    """
    在CSV文件中追加一列，第一行存放阵列数据

    参数:
        csv_file_path (str): CSV文件路径
        array_matrix (list): 二维数组矩阵数据

    返回:
        None
    """
    # 将二维矩阵转换为字符串格式
    # 可以选择不同的表示方式，这里使用扁平化的方式
    array_data_str = ""
    for i, row in enumerate(array_matrix):
        if i > 0:
            array_data_str += ";"
        array_data_str += ",".join(map(str, row))

    # 读取原CSV文件内容
    rows = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # 在第一行追加阵列数据列标题
    if rows:
        rows[0].append("Array_Data")

        # 在后续每一行追加空值占位（或根据需要填入具体数据）
        for i in range(1, len(rows)):
            rows[i].append("")

        # 在第一行数据位置填入阵列数据
        rows[1][len(rows[1])-1] = array_data_str if len(rows) > 1 else array_data_str

    # 写回CSV文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def add_timestamp_to_filename(original_filepath):
    """
    在文件名后添加当前时间戳

    参数:
        original_filepath (str): 原始文件路径

    返回:
        str: 添加时间戳后的新文件路径
    """
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 分解文件路径
    file_dir = os.path.dirname(original_filepath)
    file_name = os.path.basename(original_filepath)
    name, ext = os.path.splitext(file_name)

    # 构造新的文件名
    new_file_name = f"{name}_{timestamp}{ext}"
    new_file_path = os.path.join(file_dir, new_file_name)

    return new_file_path


def calculate_array_antenna_gain_by_hfss(matrices,
                                         project_name = None,
                                         file_name = None):
    """计算阵列天线增益"""
    """HFSS仿真"""
    if project_name is None:project_name = "E:\HFSS\patch.aedt"
    oAnsoftApp = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()  # 可以考虑注释掉
    oDesktop.OpenProject(project_name)  # 可以考虑注释掉
    oProject = oDesktop.SetActiveProject(file_name)

    for matrix_idx, matrix in enumerate(matrices):
        print(f"处理第 {matrix_idx + 1} 个矩阵")

        oProject.CopyDesign(file_name)
        oProject.Paste()
        file_name1 = file_name + "1"
        oDesign = oProject.SetActiveDesign(file_name1)
        oEditor = oDesign.SetActiveEditor("3D Modeler")

        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0

        # 遍历矩阵，查找值为1的位置
        for i in range(rows):  # 行 (Y方向)
            for j in range(cols):  # 列 (X方向)
                # if matrix[i][j] == 1:
                if (i != 0 or j != 0) and matrix[i][j] == 1:
                    # 计算复制位置坐标 (假设每个单元间距为92mm)
                    x_offset = f"{j * 92}mm"
                    y_offset = f"{i * 92}mm"

                    # 在指定位置复制天线单元
                    oEditor.DuplicateAlongLine(
                        [
                            "NAME:Selections",
                            "Selections:=", "RogersRT,feed,ground,patch,lumped_port",
                            "NewPartsModelFlag:=", "Model"
                        ],
                        [
                            "NAME:DuplicateToAlongLineParameters",
                            "CreateNewObjects:=", True,
                            "XComponent:=", x_offset,
                            "YComponent:=", y_offset,
                            "ZComponent:=", "0mm",
                            "NumClones:=", "2"
                        ],
                        [
                            "NAME:Options",
                            "DuplicateAssignments:=", True
                        ],
                        [
                            "CreateGroupsForNewObjects:=", False
                        ])

        oModule = oDesign.GetModule("RadField")
        oModule.InsertInfiniteSphereSetup(
            [
                "NAME:Infinite Sphere1",
                "UseCustomRadiationSurface:=", False,
                "CSDefinition:=", "Theta-Phi",
                "Polarization:=", "Linear",
                "Boresight:="	, "Z Axis",
                "ThetaStart:="		, "-180deg",
                "ThetaStop:="		, "180deg",
                "ThetaStep:="		, "2deg",
                "PhiStart:="		, "-180deg",
                "PhiStop:="		, "180deg",
                "PhiStep:="		, "2deg",
                "UseLocalCS:="		, False
            ])
        oDesign.SetSolutionType("HFSS Hybrid Terminal Network",
                                [
                                    "NAME:Options",
                                    "EnableAutoOpen:=", False
                                ])
        oProject.Save()
        oDesign.AnalyzeAll()
        oModule = oDesign.GetModule("ReportSetup")
        oModule.CreateReport("Gain Plot1", "Far Fields", "3D Polar Plot", "Setup1 : LastAdaptive",
                             [
                                 "Context:="	, "Infinite Sphere1"
                             ],
                             [
                                 "Phi:="			, ["All"],
                                 "Theta:="		, ["All"],
                                 "Freq:="		, ["2.5GHz"]
                             ],
                             [
                                 "Phi Component:="	, "Phi",
                                 "Theta Component:="	, "Theta",
                                 "Mag Component:="	, ["dB(GainTotal)"]
                             ])
        new_path = add_timestamp_to_filename("E:\PythonProject-NNAntenna\ARRAY-RESULT\Gain Plot1.csv")
        oModule.ExportToFile("Gain Plot1", new_path)
        oProject.Save()
        oProject.DeleteDesign(file_name1)
        oProject.Save()
        append_array_data_to_csv(new_path, matrix)
    oDesktop.CloseProject("patch")
