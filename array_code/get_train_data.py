import numpy as np
from scipy.stats import qmc
from python_script_hfss import *
import os

def generate_binary_matrices_lhs(n=10, size=15):
    """
    使用拉丁超立方采样生成n个15×15的0/1矩阵

    参数:
    n: 生成矩阵的数量
    size: 矩阵大小 (默认15)

    返回:
    包含n个二值矩阵的列表
    """
    # 创建拉丁超立方采样器 (225个维度对应15×15矩阵的每个位置)
    sampler = qmc.LatinHypercube(d=size*size)

    # 生成n个采样点
    samples = sampler.random(n=n)

    # 将连续值转换为二值(0/1)矩阵
    binary_matrices = []
    for sample in samples:
        # 将一维采样 reshape 成 size×size 矩阵
        matrix = sample.reshape(size, size)
        # 转换为二值矩阵 (>=0.5为1，否则为0)
        binary_matrix = (matrix >= 0.5).astype(int)
        binary_matrices.append(binary_matrix.tolist())

    return binary_matrices

def draw_patch_antenna(prameters =  None, project_name = None, file_name = None):
    if not project_name:
        project_name = "E:\\HFSS\\PY-PATCH-ANTENNA\\patch.aedt"
    if prameters is None:
        class Parameters:
            pass
        prameters = Parameters()

        prameters.patch_x = "-19.5mm"
        prameters.patch_y = "-24.2mm"
        prameters.patch_z = "1.575mm"
        prameters.patch_x_size = "39mm"
        prameters.patch_y_size = "48.4mm"
        prameters.lumped_port_x = "7.5353828152376mm"
        prameters.lumped_port_y = "0mm"
        prameters.lumped_port_z = "0mm"
        prameters.lumped_port_r = "0.575mm"
        prameters.ground_x = "-25mm"
        prameters.ground_y = "-30mm"
        prameters.ground_z = "0mm"
        prameters.ground_x_size = "50mm"
        prameters.ground_y_size = "60mm"
        prameters.RogersRT_x = "-25mm"
        prameters.RogersRT_y = "-30mm"
        prameters.RogersRT_z = "0mm"
        prameters.RogersRT_x_size = "50mm"
        prameters.RogersRT_y_size = "60mm"
        prameters.RogersRT_z_size = "1.575mm"
        prameters.feed_x = "7.5353828152376mm"
        prameters.feed_y = "0mm"
        prameters.feed_z = "0mm"
        prameters.feed_r = "0.5mm"
        prameters.feed_h = "1.575mm"
        prameters.frequency = "2.45GHz"
        prameters.start_frequency = "2GHz"
        prameters.stop_frequency = "3GHz"
        prameters.points = 201

    oAnsoftApp = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()  # 可以考虑注释掉

    oProject = oDesktop.NewProject()
    oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    oEditor = oDesign.SetActiveEditor("3D Modeler")

    oEditor.CreateBox(
        [
            "NAME:BoxParameters",
            # "XPosition:="	, "-25mm",
            # "YPosition:="		, "-30mm",
            # "ZPosition:="		, "0mm",
            # "XSize:="		, "50mm",
            # "YSize:="		, "60mm",
            # "ZSize:="		, "1.575mm"
            "XPosition:=", prameters.RogersRT_x,
            "YPosition:="	, prameters.RogersRT_y,
            "ZPosition:="		, prameters.RogersRT_z,
            "XSize:="		, prameters.RogersRT_x_size,
            "YSize:="		, prameters.RogersRT_y_size,
            "ZSize:="		, prameters.RogersRT_z_size
        ],
        [
            "NAME:Attributes",
            "Name:="		, "RogersRT",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"Rogers RT/duroid 5880 (tm)\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oEditor.CreateCylinder(
        [
            "NAME:CylinderParameters",
            # "XCenter:="	, "7.5353828152376mm",
            # "YCenter:="		, "0mm",
            # "ZCenter:="		, "0mm",
            # "Radius:="		, "0.5mm",
            # "Height:="		, "1.575mm",
            "XCenter:=", prameters.feed_x,
            "YCenter:="	, prameters.feed_y,
            "ZCenter:="		, prameters.feed_z,
            "Radius:="		, prameters.feed_r,
            "Height:="		, prameters.feed_h,
            "WhichAxis:="		, "Z",
            "NumSides:="		, "0"
        ],
        [
            "NAME:Attributes",
            "Name:="		, "feed",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"copper\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])
    oEditor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:="	, True,
            # "XStart:="		, "25mm",
            # "YStart:="		, "-30mm",
            # "ZStart:="		, "0mm",
            # "Width:="		, "-50mm",
            # "Height:="		, "60mm",
            "XStart:="	, prameters.ground_x,
            "YStart:="		, prameters.ground_y,
            "ZStart:="		, prameters.ground_z,
            "Width:="		, prameters.ground_x_size,
            "Height:="		, prameters.ground_y_size,
            "WhichAxis:="		, "Z"
        ],
        [
            "NAME:Attributes",
            "Name:="		, "ground",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oEditor.CreateCircle(
        [
            "NAME:CircleParameters",
            "IsCovered:="	, True,
            # "XCenter:="		, "7.5353828152376mm",
            # "YCenter:="		, "0mm",
            # "ZCenter:="		, "0mm",
            # "Radius:="		, "0.575mm",
            "XCenter:="	, prameters.lumped_port_x,
            "YCenter:="		, prameters.lumped_port_y,
            "ZCenter:="		, prameters.lumped_port_z,
            "Radius:="		, prameters.lumped_port_r,
            "WhichAxis:="		, "Z",
            "NumSegments:="		, "0"
        ],
        [
            "NAME:Attributes",
            "Name:="		, "lumped_port",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])
    oEditor.Subtract(
        [
            "NAME:Selections",
            "Blank Parts:="	, "ground",
            "Tool Parts:="		, "lumped_port"
        ],
        [
            "NAME:SubtractParameters",
            "KeepOriginals:="	, True,
            "TurnOnNBodyBoolean:="	, True
        ])
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    oEditor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:="	, True,
            # "XStart:="		, "-19.5mm",
            # "YStart:="		, "-24.2mm",
            # "ZStart:="		, "1.575mm",
            # "Width:="		, "39mm",
            # "Height:="		, "48.4mm",
            "XStart:="	, prameters.patch_x,
            "YStart:="		, prameters.patch_y,
            "ZStart:="		, prameters.patch_z,
            "Width:="		, prameters.patch_x_size,
            "Height:="		, prameters.patch_y_size,
            "WhichAxis:="		, "Z"
        ],
        [
            "NAME:Attributes",
            "Name:="		, "patch",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])
    oModule = oDesign.GetModule("BoundarySetup")
    oModule.AssignPerfectE(
        [
            "NAME:PerfE1",
            "Objects:="	, ["patch" ,"ground"],
            "InfGroundPlane:="	, False
        ])
    oModule = oDesign.GetModule("ModelSetup")
    oModule.CreateOpenRegion(
        [
            "NAME:Settings",
            # "OpFreq:="	, "2.45GHz",
            "OpFreq:=", prameters.frequency,
            "Boundary:="		, "Radiation",
            "ApplyInfiniteGP:="	, False
        ])
    # oEditor.ChangeProperty(
    #     [
    #         "NAME:AllTabs",
    #         [
    #             "NAME:Geometry3DAttributeTab",
    #             [
    #                 "NAME:PropServers",
    #                 "RadiatingSurface"
    #             ],
    #             [
    #                 "NAME:ChangedProps",
    #                 [
    #                     "NAME:Name",
    #                     "Value:="	, "air"
    #                 ],
    #                 [
    #                     "NAME:Material",
    #                     "Value:="		, "\"air\""
    #                 ]
    #             ]
    #         ]
    #     ])
    oModule = oDesign.GetModule("BoundarySetup")
    oModule.AssignLumpedPort(
        [
            "NAME:1",
            "Objects:="	, ["lumped_port"],
            "LumpedPortType:="	, "Modal",
            "DoDeembed:="		, False,
            "ImpedanceType:="	, "Impedance",
            [
                "NAME:Modes",
                [
                    "NAME:Mode1",
                    "ModeNum:="		, 1,
                    "UseIntLine:="		, True,
                    [
                        "NAME:IntLine",
                        "Coordinate System:="	, "Global",
                        "Start:="		, ["7.5353828152376mm" ,"5.55584580027407e-33mm" ,"0mm"],
                        "End:="			, ["7.5353828152376mm" ,"0.575mm" ,"0mm"]
                    ],
                    "AlignmentGroup:="	, 0,
                    "CharImp:="		, "Zpi"
                ]
            ],
            "Impedance:="		, "50ohm"
        ])
    # oModule.AssignLumpedPort(
    #     [
    #         "NAME:1",
    #         "Objects:="	, ["lumped_port"],
    #         "LumpedPortType:="	, "Modal",
    #         "DoDeembed:="		, False,
    #         "ImpedanceType:="	, "Impedance",
    #         [
    #             "NAME:Modes",
    #             [
    #                 "NAME:Mode1",
    #                 "ModeNum:="		, 1,
    #                 "UseIntLine:="		, True,
    #                 [
    #                     "NAME:IntLine",
    #                     "Coordinate System:="	, "Global",
    #                     "Start:="		, ["7.5353828152376mm" ,"4.8311702611079e-33mm" ,"0mm"],
    #                     "End:="			, ["8.0353828152376mm" ,"0mm" ,"0mm"]
    #                 ],
    #                 "AlignmentGroup:="	, 0,
    #                 "CharImp:="		, "Zpi"
    #             ]
    #         ],
    #         "Impedance:="		, "50ohm"
    #     ])
    oModule = oDesign.GetModule("AnalysisSetup")
    oModule.InsertSetup("HfssDriven",
                        [
                            "NAME:Setup1",
                            "SolveType:="	, "Single",
                            # "Frequency:="		, "2.5GHz",
                            "Frequency:="	, prameters.frequency,
                            "MaxDeltaS:="		, 0.02,
                            "UseMatrixConv:="	, False,
                            "MaximumPasses:="	, 6,
                            "MinimumPasses:="	, 1,
                            "MinimumConvergedPasses:=", 1,
                            "PercentRefinement:="	, 30,
                            "IsEnabled:="		, True,
                            [
                                "NAME:MeshLink",
                                "ImportMesh:="		, False
                            ],
                            "BasisOrder:="		, 1,
                            "DoLambdaRefine:="	, True,
                            "DoMaterialLambda:="	, True,
                            "SetLambdaTarget:="	, False,
                            "Target:="		, 0.3333,
                            "UseMaxTetIncrease:="	, False,
                            "PortAccuracy:="	, 2,
                            "UseABCOnPort:="	, False,
                            "SetPortMinMaxTri:="	, False,
                            "DrivenSolverType:="	, "Direct Solver",
                            "EnhancedLowFreqAccuracy:=", False,
                            "EnhancedFEBIPreconditioner:=", False,
                            "SaveRadFieldsOnly:="	, False,
                            "SaveAnyFields:="	, True,
                            "IESolverType:="	, "Auto",
                            "LambdaTargetForIESolver:=", 0.15,
                            "UseDefaultLambdaTgtForIESolver:=", True,
                            "IE Solver Accuracy:="	, "Balanced",
                            "InfiniteSphereSetup:="	, "",
                            "MaxPass:="		, 10,
                            "MinPass:="		, 1,
                            "MinConvPass:="		, 1,
                            "PerError:="		, 1,
                            "PerRefine:="		, 30
                        ])
    oModule.InsertFrequencySweep("Setup1",
                                 [
                                     "NAME:Sweep",
                                     "IsEnabled:="	, True,
                                     "RangeType:="		, "LinearCount",
                                     # "RangeStart:="		, "1GHz",
                                     # "RangeEnd:="		, "3GHz",
                                     # "RangeCount:="		, 201,
                                     "RangeStart:="	, prameters.start_frequency,
                                     "RangeEnd:="		, prameters.stop_frequency,
                                     "RangeCount:="		, prameters.points,
                                     "Type:="		, "Interpolating",
                                     "SaveFields:="		, True,
                                     "SaveRadFields:="	, False,
                                     "InterpTolerance:="	, 0.5,
                                     "InterpMaxSolns:="	, 250,
                                     "InterpMinSolns:="	, 0,
                                     "InterpMinSubranges:="	, 1,
                                     "InterpUseS:="		, True,
                                     "InterpUsePortImped:="	, True,
                                     "InterpUsePropConst:="	, True,
                                     "UseDerivativeConvergence:=", False,
                                     "InterpDerivTolerance:=", 0.2,
                                     "UseFullBasis:="	, True,
                                     "EnforcePassivity:="	, True,
                                     "PassivityErrorTolerance:=", 0.0001,
                                     "EnforceCausality:="	, False
                                 ])

    oProject.SaveAs(project_name, True)
    oModule = oDesign.GetModule("RadField")
    oModule.EditInfiniteSphereSetup("3D",
                                    [
                                        "NAME:3D",
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
    oProject.Save()
    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    oDesign.AnalyzeAll()
    # oDesign.Analyze("Setup1 : Sweep")
    oModule = oDesign.GetModule("ReportSetup")
    oModule.CreateReport("Gain Plot1", "Far Fields", "3D Polar Plot", "Setup1 : LastAdaptive",
                         [
                             "Context:="	, "3D"
                         ],
                         [
                             "Phi:="			, ["All"],
                             "Theta:="		, ["All"],
                             "Freq:="		, ["2.45GHz"]
                         ],
                         [
                             "Phi Component:="	, "Phi",
                             "Theta Component:="	, "Theta",
                             "Mag Component:="	, ["dB(GainTotal)"]
                         ])
    oProject.Save()
    oDesign.RenameDesignInstance("HFSSDesign1", file_name)
    oProject.Save()
    oDesktop.CloseProject(file_name)

if __name__ == '__main__':

    numbers_of_array_data = 5
    matrices = generate_binary_matrices_lhs(n=numbers_of_array_data, size=15)

    # 打印第一个矩阵查看结果
    print("第一个15×15矩阵:")
    for row in matrices[0]:
        print(row)

    file_name = "patch_array_demo"
    project_filename = file_name + ".aedt"  # 动态拼接扩展名
    project_dir = os.path.join("E:\\PythonProject-NNAntenna", "HFSS-Project")
    project_name = os.path.join(project_dir, project_filename)
    draw_patch_antenna(project_name = project_name, file_name = file_name)

    calculate_array_antenna_gain_by_hfss(matrices, project_name, file_name)

    print("计算完成")



