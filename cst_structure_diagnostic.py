"""
CST对象结构诊断工具
CST Object Structure Diagnostic Tool

用于检查CST COM接口的对象结构和可用属性
"""

import os
import sys
import time
import traceback


class CSTStructureDiagnostic:
    """CST对象结构诊断类"""

    def __init__(self):
        self.cst = None
        self.project = None
        self.com_available = False

        self.check_environment()

    def check_environment(self):
        """检查运行环境"""
        print("=" * 60)
        print("CST对象结构诊断工具")
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

        print("✅ 环境检查完成")

    def connect_to_cst(self):
        """连接到CST"""
        if not self.com_available:
            return False

        print("\n" + "=" * 60)
        print("连接到CST")
        print("=" * 60)

        try:
            import win32com.client

            # 连接CST
            try:
                self.cst = win32com.client.Dispatch("CSTStudio.Application")
                print("✅ 使用Dispatch成功创建CST实例")
            except:
                self.cst = win32com.client.DispatchEx("CSTStudio.Application")
                print("✅ 使用DispatchEx成功创建CST实例")

            if self.cst is not None:
                print("✅ CST连接成功!")

                # 检测CST版本
                try:
                    version = self.cst.Version
                    print(f"✅ CST版本: {version}")
                except Exception as e:
                    print(f"⚠️ 无法获取CST版本: {str(e)}")

                return True

        except Exception as e:
            print(f"❌ CST连接失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

        return False

    def explore_object(self, obj, obj_name, depth=0, max_depth=3):
        """探索对象的属性和方法"""
        if depth > max_depth:
            return

        indent = "  " * depth
        print(f"{indent}{obj_name}: {type(obj).__name__}")

        try:
            # 获取对象的属性
            import win32com.client
            attrs = win32com.client.Dispatch(obj).__dict__

            for attr_name, attr_value in attrs.items():
                if not attr_name.startswith('_'):
                    try:
                        attr_type = type(attr_value).__name__
                        print(f"{indent}  .{attr_name}: {attr_type}")

                        # 如果是对象，递归探索
                        if attr_type in ['Dispatch', 'CDispatch']:
                            self.explore_object(attr_value, f"{obj_name}.{attr_name}", depth + 1, max_depth)

                    except Exception as e:
                        print(f"{indent}  .{attr_name}: [无法访问] {str(e)}")

        except Exception as e:
            print(f"{indent}  [无法探索对象属性] {str(e)}")

    def check_application_structure(self):
        """检查CST应用程序对象结构"""
        if not self.cst:
            print("❌ 未连接到CST")
            return

        print("\n" + "=" * 60)
        print("CST应用程序对象结构")
        print("=" * 60)

        self.explore_object(self.cst, "CSTStudio.Application")

    def create_test_project(self):
        """创建测试项目并检查结构"""
        if not self.cst:
            print("❌ 未连接到CST")
            return

        print("\n" + "=" * 60)
        print("创建测试项目并检查结构")
        print("=" * 60)

        try:
            # 尝试创建项目
            project_types = [
                ("字符串 'MWS'", "MWS"),
                ("整数 1", 1),
                ("字符串 'MicrowaveStudio'", "MicrowaveStudio"),
                ("整数 0", 0)
            ]

            for type_name, project_type in project_types:
                try:
                    print(f"\n尝试使用{type_name}创建项目...")
                    self.project = self.cst.NewProject(project_type)
                    print(f"✅ 成功创建项目 (类型: {type_name})")

                    # 检查项目对象结构
                    print(f"\n项目对象结构 ({type_name}):")
                    self.explore_object(self.project, "Project", max_depth=2)

                    # 尝试查找模型器
                    self.find_modeler()

                    # 关闭项目
                    try:
                        self.project.Close()
                        print("✅ 项目已关闭")
                    except Exception as e:
                        print(f"⚠️ 关闭项目时出错: {str(e)}")

                    return True

                except Exception as e:
                    print(f"❌ 创建项目失败 ({type_name}): {str(e)}")
                    continue

            print("\n❌ 所有项目创建尝试都失败")
            return False

        except Exception as e:
            print(f"❌ 创建测试项目时发生错误: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            return False

    def find_modeler(self):
        """查找模型器对象"""
        if not self.project:
            return

        print("\n" + "=" * 40)
        print("查找模型器对象")
        print("=" * 40)

        # 尝试各种可能的模型器访问方式
        modeler_paths = [
            "Modeler",
            "Models",
            "Design",
            "DesignModeler",
            "Document.Modeler",
            "Document.Models"
        ]

        for path in modeler_paths:
            try:
                parts = path.split('.')
                current_obj = self.project

                for part in parts:
                    current_obj = getattr(current_obj, part)

                print(f"✅ 找到模型器: {path}")
                print(f"   类型: {type(current_obj).__name__}")

                # 探索模型器对象
                print(f"\n{path} 对象结构:")
                self.explore_object(current_obj, path, max_depth=1)

                return current_obj

            except AttributeError:
                print(f"❌ 未找到: {path}")
            except Exception as e:
                print(f"⚠️ 访问{path}时出错: {str(e)}")

        print("\n❌ 未找到模型器对象")
        return None

    def run_diagnostic(self):
        """运行完整诊断"""
        if not self.com_available:
            return

        # 连接CST
        if not self.connect_to_cst():
            return

        # 检查应用程序结构
        self.check_application_structure()

        # 创建测试项目
        self.create_test_project()

        # 清理
        if self.cst:
            try:
                self.cst.Quit()
                print("\n✅ CST已关闭")
            except Exception as e:
                print(f"\n⚠️ 关闭CST时出错: {str(e)}")

    def get_suggestions(self):
        """根据诊断结果提供建议"""
        print("\n" + "=" * 60)
        print("基于诊断结果的建议")
        print("=" * 60)

        suggestions = [
            "1. 检查CST版本兼容性",
            "2. 尝试不同的项目创建参数",
            "3. 查找正确的模型器访问路径",
            "4. 更新CST到最新版本",
            "5. 查阅CST COM接口文档"
        ]

        for suggestion in suggestions:
            print(f"   {suggestion}")


def main():
    """主函数"""
    diagnostic = CSTStructureDiagnostic()
    diagnostic.run_diagnostic()
    diagnostic.get_suggestions()


if __name__ == "__main__":
    main()