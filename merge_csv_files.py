"""
CSV文件合并工具 - 用户专用版
专门解决：每个文件只有一行表头一行数据的情况
"""

import os
import csv
import glob

def merge_single_line_csv_files(input_pattern, output_file):
    """
    合并每个只有一行表头一行数据的CSV文件

    参数:
    input_pattern: 输入文件匹配模式 (如 '*.csv' 或 'data_*.csv')
    output_file: 输出文件名
    """
    print("=" * 60)
    print("CSV文件合并工具 - 用户专用版")
    print("专门处理：每个文件只有一行表头一行数据的情况")
    print("=" * 60)

    # 获取所有匹配的文件
    all_files = glob.glob(input_pattern)

    if not all_files:
        print(f"没有找到匹配 '{input_pattern}' 的文件")
        return

    print(f"找到 {len(all_files)} 个文件:")
    for file in all_files:
        print(f"  - {file}")

    # 检查第一个文件
    if not all_files:
        return

    first_file = all_files[0]
    print(f"\n使用第一个文件 '{first_file}' 的表头")

    # 读取第一个文件获取表头
    try:
        with open(first_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)

            if len(lines) < 2:
                print(f"错误: 第一个文件行数不足，需要至少2行 (表头+数据)")
                return

            header = lines[0]
            print(f"表头: {header}")
            print(f"表头列数: {len(header)}")

    except Exception as e:
        print(f"读取第一个文件失败: {e}")
        return

    # 开始合并
    total_rows = 0
    skipped_files = []

    print(f"\n开始合并...")

    # 写入表头
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)

    # 逐个处理文件
    for i, file in enumerate(all_files):
        try:
            # 读取文件
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = list(reader)

                # 检查行数
                if len(lines) < 2:
                    print(f"  跳过 '{file}': 行数不足 (需要至少2行)")
                    skipped_files.append(file)
                    continue

                # 检查表头
                file_header = lines[0]
                if file_header != header:
                    print(f"  跳过 '{file}': 表头不一致")
                    skipped_files.append(file)
                    continue

                # 获取数据行
                data_line = lines[1]

                # 检查列数
                if len(data_line) != len(header):
                    print(f"  跳过 '{file}': 数据列数不匹配")
                    skipped_files.append(file)
                    continue

                # 写入数据
                with open(output_file, 'a', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data_line)

                total_rows += 1
                print(f"  处理完成: {file}")

        except Exception as e:
            print(f"  处理 '{file}' 时出错: {e}")
            skipped_files.append(file)

    # 完成报告
    print(f"\n" + "="*50)
    print(f"合并完成！")
    print(f"输出文件: {output_file}")
    print(f"成功处理: {total_rows} 个文件")
    print(f"跳过文件: {len(skipped_files)} 个")

    if skipped_files:
        print(f"跳过的文件:")
        for file in skipped_files:
            print(f"  - {file}")

    # 验证输出文件
    if os.path.exists(output_file):
        output_size = os.path.getsize(output_file)
        print(f"输出文件大小: {output_size} 字节")

        # 读取验证
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)

            if len(lines) == 0:
                print(f"警告: 输出文件为空")
            else:
                data_rows = len(lines) - 1  # 减去表头
                print(f"验证: 输出文件包含 {data_rows} 行数据")
                print(f"验证: 输出文件格式正确")

    return output_file

def main():
    """主函数"""
    # 示例使用
    print("使用说明:")
    print("1. 本工具专门处理每个CSV文件只有一行表头一行数据的情况")
    print("2. 所有文件必须具有相同的表头")
    print("3. 输出文件将包含一个表头和所有数据行")
    print()

    # 让用户输入参数
    # input_pattern = input("请输入文件匹配模式 (如 '*.csv' 或 'data_*.csv'): ")
    # output_file = input("请输入输出文件名 (如 'merged.csv'): ")
    input_pattern = "./RESULT/data_dict_pandas_*.csv"
    output_file = "merged_detailed_antenna_data.csv"
    # 运行合并
    merge_single_line_csv_files(input_pattern, output_file)

    print(f"\n使用完成！")

if __name__ == "__main__":
    main()