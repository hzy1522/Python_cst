"""
CST Python接口环境检查工具
CST Python Interface Environment Checker

详细检查运行CST Python接口所需的环境
"""

import os
import sys
import platform
import subprocess
import importlib.util


def print_separator():
    """打印分隔线"""
    print("=" * 60)


def check_operating_system():
    """检查操作系统"""
    print_separator()
    print("1. 操作系统检查")
    print_separator()

    print(f"操作系统名称: {platform.system()}")
    print(f"操作系统版本: {platform.version()}")
    print(f"操作系统发布: {platform.release()}")
    print(f"处理器架构: {platform.machine()}")

    if platform.system() != 'Windows':
        print("❌ 错误: CST Python接口仅支持Windows系统")
        print(f"   当前系统: {platform.system()}")
        return False
    else:
        print("✅ Windows系统检查通过")
        return True


def check_python_version():
    """检查Python版本"""
    print_separator()
    print("2. Python版本检查")
    print_separator()

    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

    # 检查Python版本是否兼容
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 6):
        print("❌ 警告: Python版本较旧，建议使用Python 3.6或更高版本")
        return False
    else:
        print("✅ Python版本检查通过")
        return True


def check_pywin32():
    """检查pywin32库"""
    print_separator()
    print("3. pywin32库检查")
    print_separator()

    try:
        import win32com.client
        print("✅ win32com.client模块导入成功")

        # 检查pywin32版本
        try:
            import pywintypes
            print(f"pywin32版本: {pywintypes.__version__}")
        except:
            print("ℹ️ 无法获取pywin32版本信息")

        return True

    except ImportError as e:
        print(f"❌ win32com.client模块导入失败: {e}")

        # 检查是否安装了pywin32
        try:
            import win32api
            print("ℹ️ win32api模块存在，但win32com.client缺失")
        except ImportError:
            print("ℹ️ pywin32库未安装")

        return False


def check_cst_installation():
    """检查CST安装"""
    print_separator()
    print("4. CST安装检查")
    print_separator()

    # 常见的CST安装路径
    cst_paths = [
        r"C:\Program Files\CST Studio Suite *",
        r"C:\Program Files (x86)\CST Studio Suite *",
        r"C:\CST Studio Suite *",
        r"D:\Program Files\CST Studio Suite *",
        r"D:\Program Files (x86)\CST Studio Suite *"
    ]

    cst_found = False
    for path_pattern in cst_paths:
        import glob
        matches = glob.glob(path_pattern)
        for match in matches:
            if os.path.isdir(match):
                print(f"✅ 找到CST安装: {match}")
                cst_found = True

                # 检查CST可执行文件
                cst_exe = os.path.join(match, "CST DESIGN ENVIRONMENT.exe")
                if os.path.exists(cst_exe):
                    print(f"   CST可执行文件: {cst_exe}")
                else:
                    print(f"⚠️ 未找到CST可执行文件: {cst_exe}")

    if not cst_found:
        print("❌ 未找到CST安装")
        print("   请确保已安装CST Studio Suite")
        return False
    else:
        return True


def check_com_registration():
    """检查CST COM组件注册"""
    print_separator()
    print("5. CST COM组件注册检查")
    print_separator()

    try:
        # 检查CST COM组件是否注册
        import winreg

        # 检查32位注册表
        try:
            key_path = r"SOFTWARE\Classes\CSTStudio.Application\CLSID"
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path):
                print("✅ 32位CST COM组件已注册")
        except FileNotFoundError:
            print("❌ 32位CST COM组件未注册")

        # 检查64位注册表
        try:
            key_path = r"SOFTWARE\Classes\CSTStudio.Application\CLSID"
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path, 0, winreg.KEY_READ | 0x10000000):
                print("✅ 64位CST COM组件已注册")
        except FileNotFoundError:
            print("❌ 64位CST COM组件未注册")
        except PermissionError:
            print("⚠️ 没有权限检查64位注册表")

        return True

    except Exception as e:
        print(f"❌ COM组件检查失败: {e}")
        return False


def check_permissions():
    """检查权限"""
    print_separator()
    print("6. 权限检查")
    print_separator()

    try:
        # 检查是否以管理员身份运行
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

        if is_admin:
            print("✅ 以管理员身份运行")
        else:
            print("⚠️ 未以管理员身份运行，可能会影响CST COM接口访问")

        return True

    except Exception as e:
        print(f"⚠️ 权限检查失败: {e}")
        return False


def test_cst_connection():
    """测试CST连接"""
    print_separator()
    print("7. CST连接测试")
    print_separator()

    try:
        import win32com.client

        print("尝试连接CST COM接口...")

        try:
            cst = win32com.client.Dispatch("CSTStudio.Application")
            print("✅ 使用Dispatch成功创建CST实例")

            # 检查CST版本
            try:
                version = cst.Version
                print(f"   CST版本: {version}")
            except Exception as e:
                print(f"⚠️ 无法获取CST版本: {e}")

            # 关闭CST
            try:
                cst.Quit()
                print("✅ CST已成功关闭")
            except Exception as e:
                print(f"⚠️ 关闭CST时出错: {e}")

            return True

        except Exception as e1:
            print(f"❌ Dispatch连接失败: {e1}")

            try:
                cst = win32com.client.DispatchEx("CSTStudio.Application")
                print("✅ 使用DispatchEx成功创建CST实例")

                # 关闭CST
                try:
                    cst.Quit()
                    print("✅ CST已成功关闭")
                except Exception as e:
                    print(f"⚠️ 关闭CST时出错: {e}")

                return True

            except Exception as e2:
                print(f"❌ DispatchEx连接也失败: {e2}")
                return False

    except ImportError as e:
        print(f"❌ 无法测试CST连接: {e}")
        return False


def generate_report(results):
    """生成检查报告"""
    print_separator()
    print("环境检查报告")
    print_separator()

    all_passed = all(results.values())

    if all_passed:
        print("✅ 所有检查项都通过！CST Python接口环境正常")
    else:
        print("❌ 部分检查项未通过，需要解决以下问题:")
        print()

        for check_name, passed in results.items():
            if not passed:
                print(f"   • {check_name}")

    print()
    print("建议的解决方案:")
    print("1. 确保在Windows系统上运行")
    print("2. 安装或修复pywin32: pip install pywin32")
    print("3. 确保CST Studio Suite已正确安装")
    print("4. 以管理员身份运行Python")
    print("5. 如果问题持续，尝试重新注册CST COM组件")


def main():
    """主函数"""
    print("CST Python接口环境检查工具")
    print("版本: 1.0")
    print()

    results = {}

    # 运行各项检查
    results["操作系统"] = check_operating_system()
    results["Python版本"] = check_python_version()
    results["pywin32库"] = check_pywin32()
    results["CST安装"] = check_cst_installation()

    # 如果是Windows系统，继续其他检查
    if results["操作系统"]:
        results["COM注册"] = check_com_registration()
        results["权限"] = check_permissions()

        # 如果pywin32已安装，测试CST连接
        if results["pywin32库"]:
            results["CST连接"] = test_cst_connection()
        else:
            results["CST连接"] = False

    # 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()