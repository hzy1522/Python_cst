import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def parse_hfss_s_parameters(data_series):
    """
    解析HFSS S参数数据，处理逗号分隔的频率和S参数值

    Args:
        data_series: 包含S参数数据的pandas Series

    Returns:
        numpy数组形式的S参数值
    """
    try:
        s_params = []
        for item in data_series:
            # 处理字符串类型的逗号分隔数据
            if isinstance(item, str) and ',' in item:
                # 分割字符串并取S参数值（通常是第二个值）
                parts = item.split(',')
                if len(parts) >= 2:
                    s_params.append(float(parts[1]))
                else:
                    # 如果只有一个值，直接转换
                    s_params.append(float(parts[0]))
            # 处理已经是数值类型的数据
            elif isinstance(item, (int, float)):
                s_params.append(float(item))
            # 处理其他情况
            else:
                s_params.append(float(str(item)))

        return np.array(s_params)
    except Exception as e:
        print(f"解析S参数数据时出错: {e}")
        # 回退到基本转换方法
        try:
            return data_series.astype(float).values
        except:
            # 如果仍然失败，尝试逐个元素转换
            result = []
            for item in data_series:
                try:
                    if isinstance(item, str) and ',' in item:
                        parts = item.split(',')
                        result.append(float(parts[1]) if len(parts) >= 2 else float(parts[0]))
                    else:
                        result.append(float(item))
                except:
                    result.append(0.0)  # 用0.0作为默认值
            return np.array(result)


def evaluate_s_parameters(predicted_s_params, actual_s_params):
    """
    评估S参数预测准确性

    Args:
        predicted_s_params: 预测的S参数
        actual_s_params: 实际的S参数

    Returns:
        dict: 包含各种评估指标的字典
    """
    if predicted_s_params is None or actual_s_params is None:
        return {}

    try:
        # 确保长度一致
        min_len = min(len(predicted_s_params), len(actual_s_params))
        if min_len <= 0:
            return {}

        pred_s_trimmed = predicted_s_params[:min_len]
        actual_s_trimmed = actual_s_params[:min_len]

        # 计算各种误差指标
        mse = mean_squared_error(actual_s_trimmed, pred_s_trimmed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_s_trimmed, pred_s_trimmed)

        # 计算平均相对误差
        nonzero_mask = actual_s_trimmed != 0
        if np.sum(nonzero_mask) > 0:
            mape = np.mean(np.abs((actual_s_trimmed[nonzero_mask] - pred_s_trimmed[nonzero_mask]) /
                                 actual_s_trimmed[nonzero_mask])) * 100
        else:
            mape = 0

        # 计算SSIM（需要reshape为2D）
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        try:
            size = int(np.sqrt(min_len))
            if size > 0 and size*size <= min_len:
                pred_2d = pred_s_trimmed[:size*size].reshape(size, size)
                actual_2d = actual_s_trimmed[:size*size].reshape(size, size)

                # 归一化到0-1范围
                pred_min, pred_max = np.min(pred_2d), np.max(pred_2d)
                actual_min, actual_max = np.min(actual_2d), np.max(actual_2d)

                if pred_max > pred_min and actual_max > actual_min:
                    pred_norm = (pred_2d - pred_min) / (pred_max - pred_min)
                    actual_norm = (actual_2d - actual_min) / (actual_max - actual_min)

                    # 计算SSIM
                    ssim_value = ssim(pred_norm, actual_norm, data_range=1.0)
                    results['ssim'] = ssim_value
        except Exception as e:
            print(f"  S参数SSIM计算警告: {e}")

        return results
    except Exception as e:
        print(f"S参数评估出错: {e}")
        return {}

def evaluate_far_field_pattern(predicted_far_field, actual_far_field):
    """
    评估远区场方向图预测准确性

    Args:
        predicted_far_field: 预测的远区场数据
        actual_far_field: 实际的远区场数据

    Returns:
        dict: 包含各种评估指标的字典
    """
    if predicted_far_field is None or actual_far_field is None:
        return {}

    try:
        # 确保长度一致
        min_len = min(len(predicted_far_field), len(actual_far_field))
        if min_len <= 0:
            return {}

        pred_ff_trimmed = predicted_far_field[:min_len]
        actual_ff_trimmed = actual_far_field[:min_len]

        # 计算各种误差指标
        mse = mean_squared_error(actual_ff_trimmed, pred_ff_trimmed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_ff_trimmed, pred_ff_trimmed)

        # 计算平均相对误差
        nonzero_mask = actual_ff_trimmed != 0
        if np.sum(nonzero_mask) > 0:
            mape = np.mean(np.abs((actual_ff_trimmed[nonzero_mask] - pred_ff_trimmed[nonzero_mask]) /
                                 actual_ff_trimmed[nonzero_mask])) * 100
        else:
            mape = 0

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        # 计算远区场的SSIM
        try:
            # 假设远区场是37x73的网格
            if min_len >= 37 * 73:
                pred_ff_2d = pred_ff_trimmed[:37*73].reshape(37, 73)
                actual_ff_2d = actual_ff_trimmed[:37*73].reshape(37, 73)

                # 归一化到0-1范围
                pred_ff_min, pred_ff_max = np.min(pred_ff_2d), np.max(pred_ff_2d)
                actual_ff_min, actual_ff_max = np.min(actual_ff_2d), np.max(actual_ff_2d)

                if pred_ff_max > pred_ff_min and actual_ff_max > actual_ff_min:
                    pred_ff_norm = (pred_ff_2d - pred_ff_min) / (pred_ff_max - pred_ff_min)
                    actual_ff_norm = (actual_ff_2d - actual_ff_min) / (actual_ff_max - actual_ff_min)

                    # 计算SSIM
                    ssim_value = ssim(pred_ff_norm, actual_ff_norm, data_range=1.0)
                    results['ssim'] = ssim_value
        except Exception as e:
            print(f"  远区场SSIM计算警告: {e}")

        return results
    except Exception as e:
        print(f"远区场评估出错: {e}")
        return {}

def compare_predictions_with_ground_truth(predicted_s_params_file, actual_s_params_file,
                                       predicted_far_field_file, actual_far_field_file):
    """
    比较预测结果与真实数据

    Args:
        predicted_s_params_file: 预测的S参数文件路径
        actual_s_params_file: 实际的S参数文件路径
        predicted_far_field_file: 预测的远区场文件路径
        actual_far_field_file: 实际的远区场文件路径

    Returns:
        dict: 包含所有评估结果的字典
    """
    results = {}

    # 读取并比较S参数
    try:
        if os.path.exists(predicted_s_params_file) and os.path.exists(actual_s_params_file):
            # 读取预测的S参数
            pred_s_df = pd.read_csv(predicted_s_params_file)
            # 假设S参数在第二列
            if pred_s_df.shape[1] >= 2:
                predicted_s_params = pred_s_df.iloc[:, 1].values
            else:
                predicted_s_params = pred_s_df.iloc[:, 0].values

            # 读取实际的S参数
            actual_s_df = pd.read_csv(actual_s_params_file, skiprows=1, sep='\t')
            if actual_s_df.shape[1] >= 2:
                actual_s_params_raw = actual_s_df.iloc[:, 1]
            else:
                actual_s_params_raw = actual_s_df.iloc[:, 0]

            # 解析实际S参数数据
            actual_s_params = parse_hfss_s_parameters(actual_s_params_raw)

            # 评估S参数
            s_params_eval = evaluate_s_parameters(predicted_s_params, actual_s_params)
            results['s_parameters'] = s_params_eval

            print("📈 S参数评估结果:")
            for metric, value in s_params_eval.items():
                if metric == 'mape':
                    print(f"   {metric.upper()}: {value:.2f}%")
                elif metric == 'ssim':
                    print(f"   {metric.upper()}: {value:.4f}")
                else:
                    print(f"   {metric.upper()}: {value:.6f}")
    except Exception as e:
        print(f"S参数文件读取或评估出错: {e}")

    # 读取并比较远区场数据
    try:
        if os.path.exists(predicted_far_field_file) and os.path.exists(actual_far_field_file):
            # 读取预测的远区场数据
            pred_ff_df = pd.read_csv(predicted_far_field_file)
            predicted_far_field = pred_ff_df['Gain_dB'].values if 'Gain_dB' in pred_ff_df.columns else pred_ff_df.values.flatten()

            # 读取实际的远区场数据
            actual_ff_df = pd.read_csv(actual_far_field_file)
            actual_far_field = actual_ff_df['Gain_dB'].values if 'Gain_dB' in actual_ff_df.columns else actual_ff_df.values.flatten()

            # 评估远区场
            far_field_eval = evaluate_far_field_pattern(predicted_far_field, actual_far_field)
            results['far_field'] = far_field_eval

            print("📊 远区场评估结果:")
            for metric, value in far_field_eval.items():
                if metric == 'mape':
                    print(f"   {metric.upper()}: {value:.2f}%")
                elif metric == 'ssim':
                    print(f"   {metric.upper()}: {value:.4f}")
                else:
                    print(f"   {metric.upper()}: {value:.6f}")
    except Exception as e:
        print(f"远区场文件读取或评估出错: {e}")

    return results
