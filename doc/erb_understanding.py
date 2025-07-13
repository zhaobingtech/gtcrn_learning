import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 或 'Agg'（不显示）、'Inline'（Jupyter）
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from gtcrn import GTCRN
# Load the GTCRN model





if __name__ == '__main__':
    run_code = 2

    if run_code == 1:
        model = GTCRN().eval()
        erb_filters = model.erb.erb_filter_banks(65, 64).numpy()

        plt.figure(figsize=(12, 8))
        sns.heatmap(erb_filters, cmap='viridis', cbar=True)
        plt.title("ERB Filter Bank Weight Matrix")
        plt.xlabel("Original Frequency Bins")
        plt.ylabel("ERB Subbands")
        plt.show()
    elif run_code == 2:
        # Save the figure

        model = GTCRN().eval()
        print(model.erb.erb_fc.weight.shape)  # 查看 ERB 投影矩阵大小
        print(model.erb.ierb_fc.weight.shape)  # 查看逆 ERB 投影矩阵大小

        # 可视化 ERB 滤波器
        import matplotlib.pyplot as plt

        plt.imshow(model.erb.erb_filter_banks(65, 64), aspect='auto')
        plt.colorbar()
        plt.title('ERB Filter Bank')
        plt.xlabel('Original Frequency Bins')
        plt.ylabel('ERB Subbands')
        plt.show()
