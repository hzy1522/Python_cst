"""
CSV文件合并工具 - 用户专用版（表头前40列匹配）
专门解决：每个文件只有一行表头一行数据的情况 + 只检查表头前40列一致性
"""

import os
import csv
import glob
import chardet  # 用于自动检测文件编码（需安装：pip install chardet）

def detect_file_encoding(file_path):
    """自动检测文件编码（解决中文/特殊编码导致的表头读取失败）"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取前10KB检测编码
    return result['encoding'] or 'utf-8'

def merge_single_line_csv_files(input_pattern, output_file, header_check_count=40):
    """
    合并每个只有一行表头一行数据的CSV文件（只检查表头前N列一致性）

    参数:
    input_pattern: 输入文件匹配模式 (如 '*.csv' 或 'data_*.csv')
    output_file: 输出文件名
    header_check_count: 表头检查列数（默认前40列）
    """
    print("=" * 60)
    print("CSV文件合并工具 - 用户专用版（表头前40列匹配）")
    print(f"专门处理：每个文件只有一行表头一行数据 + 只检查表头前{header_check_count}列一致性")
    print("核心改进：强制确保表头写入，支持多编码自动检测")
    print("=" * 60)

    # 先删除旧的输出文件（避免残留数据干扰）
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧输出文件: {output_file}")

    # 获取所有匹配的文件
    all_files = glob.glob(input_pattern)
    if not all_files:
        print(f"没有找到匹配 '{input_pattern}' 的文件")
        return

    print(f"找到 {len(all_files)} 个文件:")
    for file in all_files[:5]:  # 只显示前5个，避免输出过长
        print(f"  - {file}")
    if len(all_files) > 5:
        print(f"  ... 还有 {len(all_files)-5} 个文件")

    # 读取第一个文件的表头（关键：自动检测编码，确保表头正确读取）
    first_file = all_files[0]
    file_encoding = detect_file_encoding(first_file)
    print(f"\n使用第一个文件 '{first_file}' 的表头（前{header_check_count}列为标准）")
    print(f"自动检测编码: {file_encoding}")

    try:
        with open(first_file, 'r', encoding=file_encoding) as f:
            reader = csv.reader(f)
            # 过滤空行和纯空格行，确保正确提取表头
            lines = []
            for line in reader:
                stripped_line = [col.strip() for col in line]
                if any(stripped_line):  # 非空行（至少有一列有内容）
                    lines.append(line)

            if len(lines) < 2:
                print(f"错误: 第一个文件 '{first_file}' 行数不足")
                print(f"实际有效行数: {len(lines)} (需要至少2行：表头+数据)")
                return

            header = lines[0]  # 第一行确认为表头
            # 验证表头有效性（至少有header_check_count列，避免表头过短）
            if len(header) < header_check_count:
                print(f"错误: 第一个文件表头列数不足{header_check_count}列（实际{len(header)}列）")
                return
            if all(col.strip() == '' for col in header[:header_check_count]):
                print(f"错误: 表头前{header_check_count}列全是空格，无效")
                return

            print(f"提取到表头（前{header_check_count}列）: {header[:header_check_count]}")
            print(f"表头总列数: {len(header)}，检查列数: {header_check_count}")

    except Exception as e:
        print(f"读取第一个文件失败: {e}")
        print(f"尝试编码: {file_encoding}，如果仍失败请手动指定编码")
        return

    # 强制写入表头（单独处理，写入后立即验证）
    try:
        # 用w模式创建文件并写入完整表头（不是只写前40列）
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
            # utf-8-sig 可以处理BOM，避免表头前有隐藏字符
            writer = csv.writer(outfile)
            writer.writerow(header)  # 写入完整表头，不是只写前40列

        # 立即验证表头是否写入成功
        if not os.path.exists(output_file):
            raise Exception("输出文件创建失败")

        with open(output_file, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline().strip()
            if not first_line:
                raise Exception("表头写入为空")
            print(f"✓ 表头写入成功！验证（前{header_check_count}列）: {header[:header_check_count]}")

    except Exception as e:
        print(f"错误: 表头写入失败 - {e}")
        return

    # 开始合并数据（所有文件只取第二行数据，追加写入）
    total_rows = 0
    skipped_files = []
    print(f"\n开始合并数据...（只检查表头前{header_check_count}列一致性）")

    for i, file in enumerate(all_files):
        file_encoding = detect_file_encoding(file)
        try:
            with open(file, 'r', encoding=file_encoding) as f:
                reader = csv.reader(f)
                # 过滤空行，获取有效行
                lines = []
                for line in reader:
                    stripped_line = [col.strip() for col in line]
                    if any(stripped_line):
                        lines.append(line)

                # 检查有效行数
                if len(lines) < 2:
                    print(f"  跳过 '{os.path.basename(file)}': 有效行数不足 ({len(lines)}行)")
                    skipped_files.append(file)
                    continue

                # 关键修改：只检查表头前header_check_count列的一致性（忽略后续列）
                file_header = lines[0]
                # 取两个表头的前header_check_count列，去除空格后比较
                standard_header_slice = [col.strip() for col in header[:header_check_count]]
                file_header_slice = [col.strip() for col in file_header[:header_check_count]]

                if file_header_slice != standard_header_slice:
                    print(f"  跳过 '{os.path.basename(file)}': 表头前{header_check_count}列不一致")
                    print(f"    标准前{header_check_count}列: {standard_header_slice}")
                    print(f"    该文件前{header_check_count}列: {file_header_slice}")
                    skipped_files.append(file)
                    continue

                # 获取数据行（第二行有效数据）
                data_line = lines[1]
                # 检查数据列数至少不小于表头列数（避免数据列数过少）
                if len(data_line) < len(header):
                    print(f"  跳过 '{os.path.basename(file)}': 数据列数不足（表头{len(header)}列，数据{len(data_line)}列）")
                    skipped_files.append(file)
                    continue

                # 追加写入数据（使用a模式，不会覆盖表头）
                with open(output_file, 'a', encoding='utf-8-sig', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data_line)

                total_rows += 1
                if i % 10 == 0:  # 每处理10个文件显示一次进度
                    print(f"  已处理 {i+1}/{len(all_files)} 个文件: {os.path.basename(file)}")

        except Exception as e:
            print(f"  失败 '{os.path.basename(file)}': {str(e)[:50]}...")
            skipped_files.append(file)

    # 最终验证输出文件
    print(f"\n" + "="*50)
    print(f"合并完成！")
    print(f"输出文件: {output_file}")
    print(f"成功处理: {total_rows} 个文件")
    print(f"跳过文件: {len(skipped_files)} 个")

    # 详细验证输出文件内容
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"输出文件大小: {file_size} 字节")

        # 读取输出文件检查结构
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            lines = [line for line in reader if any(col.strip() for col in line)]

            if len(lines) == 0:
                print(f"⚠️  警告: 输出文件为空")
            else:
                output_header = lines[0]
                data_rows_count = len(lines) - 1
                print(f"✅ 表头验证成功（前{header_check_count}列）: {output_header[:header_check_count]}")
                print(f"✅ 表头总列数: {len(output_header)}")
                print(f"✅ 数据行数: {data_rows_count} 行")
                print(f"✅ 文件结构正确")

                # 显示前2行数据预览
                if data_rows_count > 0:
                    print(f"数据预览（前1行，前{header_check_count}列）: {lines[1][:header_check_count]}")
    else:
        print(f"❌ 错误: 输出文件不存在！")

    return output_file

def main():
    """主函数"""
    print("使用说明:")
    print("1. 每个CSV文件必须是「一行表头 + 一行数据」格式")
    print("2. 自动检测文件编码（支持UTF-8、GBK、GB2312等）")
    print("3. 只检查表头前40列一致性，40列之后的差异忽略")
    print("4. 输出文件会自动覆盖旧文件，表头保留第一个文件的完整表头")
    print()

    # 配置参数（可修改）
    input_pattern = "./RESULT/data_dict_pandas_*.csv"  # 输入文件路径+匹配模式
    output_file = "merged_detailed_antenna_data.csv"   # 输出文件名
    header_check_count = 40  # 只检查表头前40列一致性（可根据需要调整）

    # 如需手动输入，取消下面注释
    # input_pattern = input("请输入文件匹配模式 (如 './RESULT/*.csv'): ")
    # output_file = input("请输入输出文件名 (如 'merged.csv'): ")
    # header_check_count = int(input("请输入需要检查的表头列数（默认40）: ") or 40)

    # 运行合并
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)
    print(f"\n操作完成！请查看输出文件: {output_file}")

if __name__ == "__main__":
    main()