import subprocess
import time
import os


def monitor_nvidia_smi(interval=5):
    """实时监控nvidia-smi输出，类似于watch命令的功能"""
    try:
        while True:
            # 清空屏幕
            os.system('cls')

            # 使用subprocess调用nvidia-smi命令
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)

            # 打印nvidia-smi的输出
            print(result.stdout)

            # 等待设定的间隔时间（秒）
            time.sleep(interval)
    except KeyboardInterrupt:
        print("监控已停止")


if __name__ == "__main__":
    # 设置监控间隔为5秒，可以根据需要调整
    monitor_nvidia_smi(interval=1)
