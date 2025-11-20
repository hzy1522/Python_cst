import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_columns_50_to_250():
    """
    Read CSV files from Train_data folder and plot data from columns 50 to 250
    """
    # Define folder paths
    # data_folder = "Train_data"
    data_folder = r"E:\PythonProject-NNAntenna\Train_data"
    output_folder = "Column_Plots"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Folder '{data_folder}' does not exist.")
        return
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{data_folder}' folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files in '{data_folder}' folder.")
    
    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if file has enough columns
            if len(df.columns) < 250:
                print(f"File {csv_file} has only {len(df.columns)} columns, skipping...")
                continue
            
            # Extract columns 50 to 250 (0-indexed: 49 to 249)
            target_columns = df.columns[49:250]  # Columns 50-250
            data_subset = df[target_columns]
            
            # Create plot
            plt.figure(figsize=(15, 8))
            
            # Plot each row
            for index, row in data_subset.iterrows():
                plt.plot(range(50, 251), row.values, alpha=0.7, linewidth=0.8)
            
            # Configure plot
            plt.title(f"Data from Columns 50-250: {csv_file}")
            plt.xlabel("Column Index")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_filename = f"{os.path.splitext(csv_file)[0]}_cols_50_250.png"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated plot: {csv_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    plot_columns_50_to_250()
