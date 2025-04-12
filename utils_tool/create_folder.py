import os


def initialize_folders():
    """初始化visual文件夹结构并清空指定文件内容"""
    # 定义需要创建的文件夹路径
    folders = [
        './visual',
        './visual/visual_val'
    ]

    # 定义需要创建/清空的文件路径
    files = [
        './visual/state_time.txt',
        './visual/state.txt',
        './visual/save_epoch_evaluating_indicator.txt',
        './visual/visual_val/state_time.txt',
        './visual/visual_val/state.txt',
        './visual/visual_val/save_epoch_evaluating_indicator.txt'
    ]

    try:
        # 创建文件夹
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"Created directory: {folder}")

        # 创建/清空文件
        for file in files:
            with open(file, "w", encoding="utf-8") as f:
                f.truncate(0)
            print(f"Initialized file: {file}")

        print("All folders and files have been initialized successfully.")
        return True
    except Exception as e:
        print(f"Error occurred during initialization: {str(e)}")
        return False


def main():
    """主函数"""
    print("Starting folder and file initialization...")

    # 调用初始化函数
    success = initialize_folders()

    if success:
        print("Initialization completed successfully!")
    else:
        print("Initialization failed. Please check error messages.")

    print("Program finished.")


if __name__ == "__main__":
    main()