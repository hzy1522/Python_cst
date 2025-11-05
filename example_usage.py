"""
贴片天线设计系统完整使用示例
Patch Antenna Design System - Complete Usage Example

这个示例展示了如何在Python代码中完整使用贴片天线设计系统，
包括数据加载、模型训练、参数优化、结果分析等全流程。
"""

import sys
import os

import calculate_by_hfss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patch_antenna_design import PatchAntennaDesignSystem
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  calculate_by_hfss import Generate_test_data

from merge_csv_files import *
import time

def complete_workflow_demo():
    """完整工作流程演示"""
    print("=" * 70)
    print("贴片天线设计系统 - 完整使用示例")
    print("=" * 70)

    # 1. 创建天线设计系统
    print("\n1. 创建贴片天线设计系统...")
    system = PatchAntennaDesignSystem()

    # 2. 加载数据（使用合成数据进行演示）
    print("\n2. 加载天线数据...")

    # 方法A: 使用合成数据（用于演示和测试）
    # print("   使用合成数据进行演示...")
    # X_scaled, y, X_original, y_original = system.generate_synthetic_data(
    #     num_samples=8000  # 生成8000个样本
    # )

    #生成天线数据
    num_samples = 100 #生成10000个天线数据
    # Generate_test_data(num_samples)

    print("合并所有数据:")
    #合并所有数据到 merged_detailed_antenna_data。csv 文件

    # 让用户输入参数
    # input_pattern = input("请输入文件匹配模式 (如 '*.csv' 或 'data_*.csv'): ")
    # output_file = input("请输入输出文件名 (如 'merged.csv'): ")
    input_pattern = "./RESULT/data_dict_pandas_*.csv"
    output_file = "merged_detailed_antenna_data.csv"
    # 运行合并
    merge_single_line_csv_files(input_pattern, output_file)

    print(f"\n合并完成！")


    # 方法B: 使用真实CSV数据（请替换为您的实际数据文件）
    print("   使用真实CSV数据...")
    X_scaled, y, X_original, y_original = system.load_csv_data(
        csv_file='./merged_detailed_antenna_data.csv',
        param_cols=['patch_length', 'patch_width', 'ground_thickness', 'signal_layer_thickness'],
        perf_cols=['_最小值', 'Freq [GHz]', 'Gain_dB']
    )

    print(f"   数据加载完成: {X_original.shape[0]}个样本")

    # 3. 数据预处理和划分
    print("\n3. 数据预处理...")

    # 显示数据统计信息
    print("   参数统计:")
    for i, name in enumerate(system.param_names):
        print(f"     {name}: 均值={X_original[:, i].mean():.3f}, 标准差={X_original[:, i].std():.3f}")

    print("   性能指标统计:")
    for i, name in enumerate(system.perf_names):
        print(f"     {name}: 均值={y_original[:, i].mean():.3f}, 标准差={y_original[:, i].std():.3f}")

    # 划分训练集、验证集和测试集
    print("\n   划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"     训练集: {X_train.shape[0]}个样本")
    print(f"     验证集: {X_val.shape[0]}个样本")
    print(f"     测试集: {X_test.shape[0]}个样本")

    # 4. 创建和训练模型
    print("\n4. 模型训练...")

    # 创建ResNet模型（推荐）
    # 参数:
    #         model_type: 模型类型 ('mlp', 'resnet', 'cnn')

    # print("   创建ResNet模型...")
    # model = system.create_model(model_type='resnet')

    # 创建ResNet模型（推荐）
    print("   创建ResNet模型...")
    model = system.create_model(model_type='cnn')

    # 训练模型
    print("   开始训练...")
    training_start_time = time.time()

    history = system.train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=250,       # 训练轮数
        batch_size=128,   # 批次大小
        lr=0.001          # 学习率
    )

    training_time = time.time() - training_start_time
    print(f"   训练完成！耗时: {training_time:.2f}秒")

    # 5. 模型评估
    print("\n5. 模型性能评估...")

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test).cpu().numpy()

    # 计算R²分数和RMSE
    from sklearn.metrics import r2_score, mean_squared_error

    print("   测试集性能指标:")
    for i, name in enumerate(system.perf_names):
        r2 = r2_score(y_test.cpu().numpy()[:, i], y_pred_test[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy()[:, i], y_pred_test[:, i]))
        print(f"     {name}: R²={r2:.4f}, RMSE={rmse:.4f}")

    # 6. 生成可视化结果
    print("\n6. 生成可视化结果...")
    system.visualize_results(history, y_test.cpu().numpy(), y_pred_test)

    # 7. 天线参数优化
    print("\n7. 天线参数优化...")

    # 定义设计目标（根据实际需求调整）
    target_specs = [
        -20.0,   # S11最小值目标: -32dB (越小越好)
        10,    # 工作频率目标: 2.45GHz (WiFi频段)
        7.0      # 远区场增益目标: 7.0dBi (越大越好)
    ]

    print("   设计目标:")
    for i, (name, target) in enumerate(zip(system.perf_names, target_specs)):
        print(f"     {name}: {target}")

    # 定义参数边界（根据实际制造能力调整）
    param_bounds = np.array([
        [5.0, 15.0],    # 贴片长度范围 (mm)
        [5.0, 15.0],    # 贴片宽度范围 (mm)
        [0.01, 0.05],      # GND厚度范围 (mm)
        [0.01, 0.05]       # 信号线厚度范围 (mm)
    ])

    print("   参数优化边界:")
    for i, name in enumerate(system.param_names):
        print(f"     {name}: {param_bounds[i, 0]:.1f} - {param_bounds[i, 1]:.1f}")

    # 执行优化
    print("   开始参数优化...")
    optimization_start_time = time.time()

    optimal_params, predicted_performance, optimization_loss = system.optimize_antenna(
        model, target_specs, param_bounds,
        num_iterations=3000,  # 优化迭代次数
        learning_rate=0.01    # 优化学习率
    )

    optimization_time = time.time() - optimization_start_time
    print(f"   优化完成！耗时: {optimization_time:.2f}秒")

    # 8. 优化结果分析
    print("\n8. 优化结果分析:")

    print("   最优设计参数:")
    for i, name in enumerate(system.param_names):
        print(f"     {name}: {optimal_params[i]:.3f}")

    print("\n   预测性能指标:")
    performance_metrics = []
    for i, (name, pred, target) in enumerate(zip(system.perf_names, predicted_performance, target_specs)):
        diff = abs(pred - target)
        satisfied = None
        # 根据指标类型判断是否满足要求
        if name == system.perf_names[0]:  # S11最小值 (越小越好)
            satisfied = pred <= target
            status = "✓" if satisfied else "⚠️"
            print(f"     {status} {name}: {pred:.3f}dB (目标: ≤{target}dB)")
        elif name == system.perf_names[1]:  # 工作频率 (越接近目标越好)
            satisfied = abs(diff) < 0.05  # 允许±50MHz误差
            status = "✓" if satisfied else "⚠️"
            print(f"     {status} {name}: {pred:.3f}GHz (目标: {target}GHz ±50MHz)")
        elif name == system.perf_names[2]:  # 远区场增益 (越大越好)
            satisfied = pred >= target
            status = "✓" if satisfied else "⚠️"
            print(f"     {status} {name}: {pred:.3f}dBi (目标: ≥{target}dBi)")

        performance_metrics.append({
            'name': name,
            'predicted': pred,
            'target': target,
            'satisfied': satisfied
        })

    # 9. HFSS仿真验证
    print("\n9. HFSS仿真验证...")
    simulated_performance = system.hfss_interface(optimal_params)

    # 10. 设计可行性分析
    print("\n10. 设计可行性分析:")

    # 检查所有性能指标是否满足要求
    all_satisfied = all(metric['satisfied'] for metric in performance_metrics)

    if all_satisfied:
        print("   🎉 设计成功！所有性能指标均满足要求。")
    else:
        print("   ⚠️  设计基本完成，但部分指标需要进一步优化。")

        # 提供改进建议
        print("\n   改进建议:")
        for metric in performance_metrics:
            if not metric['satisfied']:
                if metric['name'] == system.perf_names[0]:  # S11
                    print("     - 调整天线尺寸或添加匹配网络改善S11")
                elif metric['name'] == system.perf_names[1]:  # 频率
                    print("     - 调整贴片长度以调整工作频率")
                elif metric['name'] == system.perf_names[2]:  # 增益
                    print("     - 增加贴片尺寸或优化基板材料提高增益")

    # 11. 保存完整设计报告
    print("\n11. 保存设计报告...")

    design_report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'optimal_parameters': optimal_params,
        'predicted_performance': predicted_performance,
        'simulated_performance': simulated_performance,
        'target_specifications': target_specs,
        'model_performance': {
            'r2_scores': [r2_score(y_test.cpu().numpy()[:, i], y_pred_test[:, i]) for i in range(3)],
            'rmse_scores': [np.sqrt(mean_squared_error(y_test.cpu().numpy()[:, i], y_pred_test[:, i])) for i in range(3)]
        },
        'training_info': {
            'model_type': 'resnet',
            'epochs': 250,
            'batch_size': 128,
            'training_time': training_time,
            'optimization_time': optimization_time
        },
        'is_feasible': all_satisfied
    }

    # 保存报告
    np.save('patch_antenna_results/complete_design_report.npy', design_report)
    print("   完整设计报告已保存到 patch_antenna_results/complete_design_report.npy")

    # 12. 输出设计总结
    print("\n" + "=" * 70)
    print("设计总结")
    print("=" * 70)

    print(f"设计时间: {design_report['timestamp']}")
    print(f"模型类型: {design_report['training_info']['model_type']}")
    print(f"总耗时: {training_time + optimization_time:.2f}秒")
    print(f"设计可行性: {'可行' if all_satisfied else '需要优化'}")

    print("\n最终设计参数:")
    for i, name in enumerate(system.param_names):
        print(f"  {name}: {optimal_params[i]:.3f}")

    print("\n预测性能:")
    for i, name in enumerate(system.perf_names):
        print(f"  {name}: {predicted_performance[i]:.3f}")

    print("\n结果文件已保存到 patch_antenna_results 目录:")
    print("  - design_result.npy: 设计结果数据")
    print("  - complete_design_report.npy: 完整设计报告")
    print("  - training_curves.png: 训练曲线图")
    print("  - prediction_scatter.png: 预测性能图")
    print("  - error_distribution.png: 误差分布图")
    print("  - correlation_analysis.png: 参数相关性图")

    return design_report

def batch_optimization_demo():
    """批量优化演示"""
    print("\n" + "=" * 70)
    print("批量天线设计演示")
    print("=" * 70)

    system = PatchAntennaDesignSystem()
    # 2. 加载数据（使用合成数据进行演示）
    print("\n2. 加载天线数据...")

    # 生成天线数据
    num_samples = 50  # 生成10000个天线数据
    Generate_test_data(num_samples)

    print("合并所有数据:")
    # 合并所有数据到 merged_detailed_antenna_data。csv 文件
    input_pattern = "./RESULT/data_dict_pandas_*.csv"
    output_file = "merged_detailed_antenna_data.csv"
    header_check_count = 40
    # 运行合并
    merge_single_line_csv_files(input_pattern, output_file, header_check_count)

    print(f"\n合并完成！")

    # 方法B: 使用真实CSV数据（请替换为您的实际数据文件）
    print("   使用真实CSV数据...")
    X_scaled, y, X_original, y_original = system.load_csv_data(
        csv_file='./merged_detailed_antenna_data.csv',
        param_cols=['patch_length', 'patch_width', 'ground_thickness', 'signal_layer_thickness'],
        perf_cols=['_最小值', 'Freq [GHz]', 'Gain_dB']
    )

    print(f"   数据加载完成: {X_original.shape[0]}个样本")
    # # 加载数据
    # X_scaled, y, X_original, y_original = system.generate_synthetic_data(num_samples=5000)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 训练模型
    model = system.create_model('resnet')
    history = system.train_model(model, X_train, y_train, X_val, y_val, epochs=200)

    # 定义多个设计目标
    design_targets = [
        # Patch标准天线
        {
            'name': 'IoT_Miniaturized',
            'targets': [-20.0, 10, 7],
            'bounds': [[5, 15], [5, 15], [0.01, 0.05], [0.01, 0.05]]
        },
        # WiFi 2.4GHz 高增益设计
        {
            'name': 'WiFi_2.4GHz_HighGain',
            'targets': [-30.0, 2.45, 7.5],
            'bounds': [[15, 45], [15, 45], [0.8, 2.5], [0.2, 0.8]]
        },
        # WiFi 5GHz 设计
        {
            'name': 'WiFi_5GHz_Design',
            'targets': [-28.0, 5.2, 6.0],
            'bounds': [[8, 25], [8, 25], [0.5, 2.0], [0.1, 0.6]]
        },
        # IoT设备小型化设计
        {
            'name': 'IoT_Miniaturized',
            'targets': [-25.0, 2.4, 5.0],
            'bounds': [[10, 25], [10, 25], [0.5, 1.5], [0.1, 0.4]]
        }
    ]

    print(f"开始批量设计 {len(design_targets)} 个天线...")

    batch_results = []
    for i, target_info in enumerate(design_targets):
        print(f"\n设计 {i+1}/{len(design_targets)}: {target_info['name']}")

        optimal_params, predicted_perf, loss = system.optimize_antenna(
            model, target_info['targets'], np.array(target_info['bounds']),
            num_iterations=2000
        )

        result = {
            'design_name': target_info['name'],
            'optimal_parameters': optimal_params,
            'predicted_performance': predicted_perf,
            'target_specifications': target_info['targets'],
            'optimization_loss': loss
        }

        batch_results.append(result)

        print(f"  完成！预测S11: {predicted_perf[0]:.2f}dB, 频率: {predicted_perf[1]:.2f}GHz, 增益: {predicted_perf[2]:.2f}dBi")

    # 保存批量设计结果
    np.save('patch_antenna_results/batch_design_results.npy', batch_results)
    print(f"\n批量设计完成！结果已保存到 batch_design_results.npy")

    return batch_results

def model_comparison_demo():
    """模型比较演示"""
    print("\n" + "=" * 70)
    print("模型性能比较演示")
    print("=" * 70)

    system = PatchAntennaDesignSystem()

    # 加载数据
    X_scaled, y, X_original, y_original = system.generate_synthetic_data(num_samples=6000)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 比较不同模型
    models_to_test = ['mlp', 'resnet', 'cnn']
    comparison_results = {}

    for model_type in models_to_test:
        print(f"\n测试 {model_type.upper()} 模型...")

        model = system.create_model(model_type)
        history = system.train_model(model, X_train, y_train, X_val, y_val, epochs=200)

        # 评估性能
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val).cpu().numpy()

        from sklearn.metrics import r2_score, mean_squared_error
        r2_scores = []
        rmse_scores = []

        for i in range(3):
            r2 = r2_score(y_val.cpu().numpy()[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_val.cpu().numpy()[:, i], y_pred[:, i]))
            r2_scores.append(r2)
            rmse_scores.append(rmse)

        comparison_results[model_type] = {
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores,
            'best_val_loss': min(history['val_loss'])
        }

    # 打印比较结果
    print("\n模型性能比较结果:")
    print("-" * 70)
    print(f"{'模型':<10} {'平均R²':<12} {'平均RMSE':<12} {'最佳损失':<12}")
    print("-" * 70)

    for model_type, results in comparison_results.items():
        avg_r2 = np.mean(results['r2_scores'])
        avg_rmse = np.mean(results['rmse_scores'])
        best_loss = results['best_val_loss']
        print(f"{model_type.upper():<10} {avg_r2:<12.4f} {avg_rmse:<12.4f} {best_loss:<12.4f}")

    return comparison_results

if __name__ == "__main__":
    # 导入必要的库
    import torch

    print("欢迎使用贴片天线设计系统！")
    print("本系统专门用于贴片天线的深度学习设计和优化。")

    # 运行完整工作流程演示
    print("\n" + "=" * 50)
    print("正在运行完整设计流程...")
    print("=" * 50)

    # # 演示1: 完整设计流程
    design_report = complete_workflow_demo()

    # 演示2: 批量设计（可选）
    # print("\n" + "=" * 50)
    # print("正在运行批量设计演示...")
    # print("=" * 50)
    # batch_results = batch_optimization_demo()

    # 演示3: 模型比较（可选）
    # print("\n" + "=" * 50)
    # print("正在运行模型比较演示...")
    # print("=" * 50)
    # comparison_results = model_comparison_demo()

    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n您可以在 patch_antenna_results 目录中查看详细的设计结果和可视化图表。")
    print("如需使用您自己的数据，请修改代码中的CSV文件路径。")