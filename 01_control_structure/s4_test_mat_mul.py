import time
import torch

matrix_size = 10000 # 積の計算の行列のサイズ
A = torch.randn(matrix_size, matrix_size)
B = torch.randn(matrix_size, matrix_size)

print(f"{matrix_size} ^ 2 サイズの2つの行列の積を求めます")

device = "cuda" if torch.cuda.is_available() else "cpu" # GPU (CUDA) が利用可能か確認
print(f"使用可能なGPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'なし'}")

start_time = time.time()

if device == "cuda": # GPUでの計算
    A_dev, B_dev = A.to("cuda"), B.to("cuda") # データをGPUのメモリに転送
    
else: # CPUでの計算
    A_dev, B_dev = A.cpu(), B.cpu() # 元々がCPU側のデータだけど明示的に転送

result = torch.matmul(A_dev, B_dev) # 行列の積

proc_duration = time.time() - start_time # 計算時間
print(f"処理時間: {proc_duration:.4f} 秒")