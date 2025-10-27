"""
CST COM组件注册修复工具
CST COM Component Registration Fixer

帮助修复CST COM组件未注册的问题
"""

import os
import sys
import subprocess
import winreg
import glob


def print_separator():
    """打印分隔线"""
    print("=" * 60)


def is_admin():
    """检查是否以管理员身份运行"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False


def check_cst_installation():
    """检查CST安装路径"""
    print_separator()
    print("1. 查找CST安装路径")
    print_separator()

    # 常见的CST安装路径模式
    cst_path_patterns = [
        r"C:\Program Files\CST Studio Suite *",
        r"C:\Program Files (x86)\CST Studio Suite *",
        r"C:\CST Studio Suite *",
        r"D:\Program Files\CST Studio Suite *",
        r"D:\CST Studio Suite *",
        r"D:\Program Files (x86)\CST Studio Suite *"
    ]

    cst_install_paths = []

    for pattern in cst_path_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isdir(match):
                cst_install_paths.append(match)

    if not cst_install_paths:
        print("❌ 未找到CST安装路径")
        return None

    print(f"✅ 找到{len(cst_install_paths)}个CST安装:")
    for i, path in enumerate(cst_install_paths, 1):
        print(f"   {i}. {path}")

    # 选择最新的版本
    cst_install_paths.sort(reverse=True)  # 按字母顺序倒序，通常新版本在后面
    selected_path = cst_install_paths[0]
    print(f"\n✅ 选择最新版本: {selected_path}")

    return selected_path


def find_cst_com_dll(cst_path):
    """查找CST COM组件DLL文件"""
    print_separator()
    print("2. 查找CST COM组件DLL")
    print_separator()

    # 常见的CST COM DLL文件名
    dll_names = [
        "CSTStudio.dll",
        "CSTDesignEnvironment.dll",
        "CSTStudioApplication.dll",
    ]

    # 在CST安装目录中搜索
    dll_paths = []

    for root, dirs, files in os.walk(cst_path):
        for file in files:
            if file in dll_names:
                full_path = os.path.join(root, file)
                dll_paths.append(full_path)

    if not dll_paths:
        print("❌ 未找到CST COM DLL文件")
        return None

    print(f"✅ 找到{len(dll_paths)}个COM DLL文件:")
    for i, path in enumerate(dll_paths, 1):
        print(f"   {i}. {path}")

    # 通常第一个就是主要的COM DLL
    return dll_paths[0]


def register_com_component(dll_path):
    """注册COM组件"""
    print_separator()
    print(f"3. 注册COM组件: {dll_path}")
    print_separator()

    if not os.path.exists(dll_path):
        print(f"❌ DLL文件不存在: {dll_path}")
        return False

    # 使用regsvr32注册
    try:
        # 检查系统位数
        if sys.maxsize > 2 ** 32:
            regsvr32_path = r"C:\Windows\System32\regsvr32.exe"
        else:
            regsvr32_path = r"C:\Windows\SysWOW64\regsvr32.exe"

        print(f"使用regsvr32路径: {regsvr32_path}")

        # 注册命令
        cmd = [regsvr32_path, "/s", "/i", dll_path]
        result = subprocess.run(cmd, shell=True, check=True)

        if result.returncode == 0:
            print("✅ COM组件注册成功！")
            return True
        else:
            print(f"❌ COM组件注册失败，返回码: {result.returncode}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ 注册过程中发生错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 注册失败: {e}")
        return False


def check_registration():
    """检查CST COM组件是否已注册"""
    print_separator()
    print("4. 检查CST COM组件注册状态")
    print_separator()

    cst_progids = [
        "CSTStudio.Application",
        "CSTDesignEnvironment.Application",
        "CSTStudioSuite.Application",
    ]

    registered = False

    for progid in cst_progids:
        try:
            # 检查ProgID是否存在
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid):
                print(f"✅ {progid} 已注册")

                # 获取CLSID
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, f"{progid}\\CLSID") as clsid_key:
                    clsid, _ = winreg.QueryValueEx(clsid_key, "")
                    print(f"   CLSID: {clsid}")

                registered = True

        except FileNotFoundError:
            print(f"❌ {progid} 未注册")
        except Exception as e:
            print(f"⚠️ 检查{progid}时出错: {e}")

    return registered


def test_cst_connection():
    """测试CST连接"""
    print_separator()
    print("5. 测试CST COM连接")
    print_separator()

    try:
        import win32com.client

        print("尝试创建CST实例...")

        try:
            cst = win32com.client.Dispatch("CSTStudio.Application")
            print("✅ 使用Dispatch成功创建CST实例")

            # 获取版本信息
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
        print(f"❌ 缺少pywin32库: {e}")
        print("   请安装: pip install pywin32")
        return False


def run_cst_repair(cst_path):
    """运行CST修复程序"""
    print_separator()
    print("6. 运行CST修复程序")
    print_separator()

    # 查找CST修复程序
    repair_exe = os.path.join(cst_path, "unins000.exe")

    if os.path.exists(repair_exe):
        print(f"找到CST修复程序: {repair_exe}")

        try:
            # 运行修复程序
            cmd = [repair_exe, "/repair"]
            print("正在运行CST修复程序...")
            print("⚠️ 这可能需要几分钟时间，请耐心等待...")

            subprocess.run(cmd, check=True)
            print("✅ CST修复程序运行完成")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ CST修复程序运行失败: {e}")
            return False
        except Exception as e:
            print(f"❌ 运行修复程序时出错: {e}")
            return False
    else:
        print("❌ 未找到CST修复程序")
        return False


def main():
    """主函数"""
    print("CST COM组件注册修复工具")
    print("版本: 1.0")
    print()

    # 检查是否以管理员身份运行
    if not is_admin():
        print("❌ 请以管理员身份运行此工具！")
        print("   右键点击命令提示符，选择'以管理员身份运行'")
        sys.exit(1)

    print("✅ 已以管理员身份运行")
    print()

    # 1. 查找CST安装路径
    cst_path = check_cst_installation()
    if not cst_path:
        print("\n❌ 无法继续，未找到CST安装")
        sys.exit(1)

    # 2. 查找CST COM DLL
    dll_path = find_cst_com_dll(cst_path)

    # 3. 注册COM组件
    registration_success = False
    if dll_path:
        registration_success = register_com_component(dll_path)
    else:
        print("⚠️ 无法直接注册COM组件，尝试其他方法")

    # 4. 如果直接注册失败，运行CST修复
    if not registration_success:
        print("\n尝试运行CST修复程序...")
        repair_success = run_cst_repair(cst_path)

        if repair_success:
            # 修复后再次尝试注册
            dll_path = find_cst_com_dll(cst_path)
            if dll_path:
                registration_success = register_com_component(dll_path)

    # 5. 检查注册状态
    print("\n" + "=" * 60)
    print("最终检查")
    print("=" * 60)

    registered = check_registration()

    if registered:
        print("\n✅ CST COM组件注册状态良好！")

        # 测试连接
        print("\n正在测试CST连接...")
        connection_success = test_cst_connection()

        if connection_success:
            print("\n🎉 CST COM组件注册修复成功！")
            print("   现在可以正常使用Python调用CST了")
        else:
            print("\n⚠️ COM组件已注册，但连接测试失败")
            print("   可能需要重启计算机或重新启动CST")
    else:
        print("\n❌ CST COM组件注册修复失败")
        print("\n建议的解决方案:")
        print("1. 重新启动计算机后重试")
        print("2. 以管理员身份重新安装CST")
        print("3. 检查是否有足够的磁盘空间")
        print("4. 检查Windows Installer服务是否正常运行")


if __name__ == "__main__":
    main()