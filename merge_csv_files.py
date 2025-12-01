"""
CSV文件合并工具 - 简洁版
解决：只处理了一个文件的问题
"""

import os
import csv
import glob
import chardet

def detect_file_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding'] or 'utf-8'


# 添加曲线验证函数
def validate_s_parameter_curve(row):
    """
    验证S参数曲线是否具有中心凹陷的典型形状
    """
    try:
        # 提取数值数据（假设S参数从某列开始）
        s_values = []
        for i in range(49, min(250, len(row))):  # 索引从49开始对应第50列
            try:
                s_values.append(float(row[i]))
            except ValueError:
                continue
        # for val in row:
        #     try:
        #         s_values.append(float(val))
        #     except ValueError:
        #         continue

        if len(s_values) < 201:  # 确保有足够的数据点
            # print(f"数据点数不足，无法进行验证")
            return False
        else:
            print(f"len={len(s_values)}")


        # 简单的形状验证逻辑
        # 1. 检查是否存在明显的最小值点
        min_idx = s_values.index(min(s_values))

        # 2. 检查最小值是否在中间区域
        if min_idx < 50 or min_idx > 150:
            # print(f"数据点不在中间区域，无法进行验证")
            return False

        # 3. 检查曲线趋势（两端高，中间低）
        left_avg = sum(s_values[:50]) / 50
        right_avg = sum(s_values[-50:]) / 50
        center_avg = sum(s_values[75:125]) / 50

        # 中间应该比两边低
        if center_avg >= min(left_avg, right_avg):
            # print(f"数据点趋势不符合要求，无法进行验证")
            return False

        return True

    except:
        return False

def merge_single_line_csv_files(input_pattern, output_file, header_check_count=40):
    """合并每个只有一行表头一行数据的CSV文件"""
    print("=" * 60)
    print("CSV文件合并工具 - 简洁版")
    print(f"处理模式：一行表头 + 一行数据 + 前{header_check_count}列匹配")
    print("筛选规则：删除'_最小值'列中数值 > -5dB 的行")
    print("=" * 60)

    # 删除旧的输出文件
    if os.path.exists(output_file):
        os.remove(output_file)

    # 获取所有匹配的文件
    all_files = glob.glob(input_pattern)
    if not all_files:
        print(f"❌ 没有找到匹配 '{input_pattern}' 的文件")
        return

    print(f"📁 找到 {len(all_files)} 个文件")

    # 读取第一个文件的表头作为标准
    first_file = all_files[0]
    file_encoding = detect_file_encoding(first_file)
    print(f"🔍 使用 '{os.path.basename(first_file)}' 作为标准模板")

    try:
        with open(first_file, 'r', encoding=file_encoding) as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                stripped_line = [col.strip() for col in line]
                if any(stripped_line):
                    lines.append(line)

            if len(lines) < 2:
                print(f"❌ 第一个文件行数不足（需要至少2行）")
                return

            header = lines[0]
            if len(header) < header_check_count:
                print(f"❌ 表头列数不足{header_check_count}列")
                return

            print(f"📋 标准表头：{len(header)}列，前{header_check_count}列为匹配基准")

    except Exception as e:
        print(f"❌ 读取第一个文件失败: {e}")
        return

    # 写入表头
    try:
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
        print(f"✓ 表头写入成功")

    except Exception as e:
        print(f"❌ 表头写入失败: {e}")
        return

    # 开始合并数据
    total_rows = 0
    skipped_files = []
    success_files = []

    print(f"\n🚀 开始合并数据...")
    print("-" * 60)

    for i, file in enumerate(all_files):
        file_basename = os.path.basename(file)

        # 显示进度（每10个文件显示一次）
        if i % 10 == 0 or i == len(all_files) - 1:
            print(f"进度: {i+1}/{len(all_files)} 文件", end='\r')

        try:
            file_encoding = detect_file_encoding(file)
            with open(file, 'r', encoding=file_encoding) as f:
                reader = csv.reader(f)
                lines = []
                for line in reader:
                    stripped_line = [col.strip() for col in line]
                    if any(stripped_line):
                        lines.append(line)

            # 检查文件结构
            if len(lines) < 2:
                skipped_files.append((file_basename, "行数不足"))
                continue

            # 检查表头前N列
            file_header = lines[0]
            standard_header_slice = [col.strip() for col in header[:header_check_count]]
            file_header_slice = [col.strip() for col in file_header[:header_check_count]]

            if file_header_slice != standard_header_slice:
                skipped_files.append((file_basename, "表头不匹配"))
                continue

            # 检查数据列数
            data_line = lines[1]
            if len(data_line) < len(header):
                skipped_files.append((file_basename, "数据列数不足"))
                continue

            # 写入数据
            with open(output_file, 'a', encoding='utf-8-sig', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(data_line)

            total_rows += 1
            success_files.append(file_basename)

        except Exception as e:
            error_msg = str(e)[:50]
            skipped_files.append((file_basename, f"错误: {error_msg}"))

    # 数据清理步骤
    print(f"\n\n🧹 开始数据筛选...")

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            all_lines = [line for line in reader if any(col.strip() for col in line)]

        if len(all_lines) > 1:
            header_row = all_lines[0]
            data_rows = all_lines[1:]

            # 找到"_最小值"列
            min_value_col_index = -1
            for i, col_name in enumerate(header_row):
                if "_最小值" in str(col_name):
                    min_value_col_index = i
                    break

            if min_value_col_index != -1:
                min_value_col_name = header_row[min_value_col_index]
                filtered_data = []
                removed_count = 0
                shape_filtered_count = 0  # 新增计数器

                for row in data_rows:
                    try:
                        # 原有的最小值检查
                        min_value = float(row[min_value_col_index])
                        if min_value > -5.0:
                            removed_count += 1
                            continue

                        # 新增：曲线形状验证
                        print(row)
                        if not validate_s_parameter_curve(row):
                            shape_filtered_count += 1
                            continue

                        filtered_data.append(row)
                    except (ValueError, IndexError):
                        filtered_data.append(row)

                print(f"📊 筛选统计:")
                print(f"   原始数据: {len(data_rows)} 行")
                print(f"   筛选后: {len(filtered_data)} 行")
                print(f"   删除行数: {removed_count} 行 (S11 > -5dB)")
                print(f"   曲线异常删除: {shape_filtered_count} 行")

                # 保存筛选后的数据
                with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header_row)
                    writer.writerows(filtered_data)

                print(f"✓ 数据筛选完成")
            else:
                print(f"⚠️  未找到'_最小值'列，跳过筛选")
        else:
            print(f"⚠️  数据不足，跳过筛选")

    # 输出总结报告
    print(f"\n" + "="*60)
    print("📋 合并完成报告")
    print("="*60)
    print(f"📁 总文件数: {len(all_files)}")
    print(f"✅ 成功处理: {len(success_files)} 个文件")
    print(f"❌ 跳过文件: {len(skipped_files)} 个文件")

    if skipped_files:
        # 统计跳过原因
        reason_stats = {}
        for file, reason in skipped_files:
            reason_stats[reason] = reason_stats.get(reason, 0) + 1

        print(f"\n🔍 跳过原因统计:")
        for reason, count in reason_stats.items():
            print(f"   • {reason}: {count} 个文件")

    # 最终验证
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        with open(output_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            lines = [line for line in reader if any(col.strip() for col in line)]

        if len(lines) > 0:
            final_data_rows = len(lines) - 1
            print(f"\n📄 输出文件: {output_file}")
            print(f"📊 文件大小: {file_size} 字节")
            print(f"📈 最终数据: {final_data_rows} 行 × {len(lines[0])} 列")
            print("✅ 合并任务完成！")
        else:
            print(f"\n❌ 错误: 输出文件为空！")

    return output_file

def main():
    """主函数"""
    # 配置参数
    input_pattern = "./Train_data/data_dict_pandas_*.csv"
    output_file = "merged_detailed_antenna_data.csv"
    header_check_count = 40

    # 运行合并
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)

if __name__ == "__main__":
    main()