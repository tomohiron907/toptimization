"""GPU利用状況の診断スクリプト"""
import sys
import platform

print("=" * 60)
print("toptimization GPU診断レポート")
print("=" * 60)

print(f"\nOS: {platform.system()} {platform.machine()}")
print(f"Python: {sys.version}")

# Taichi のインストール確認
try:
    import taichi as ti
    print(f"Taichi: {ti.__version__}")
except ImportError:
    print("ERROR: Taichi がインストールされていません")
    sys.exit(1)

# 利用可能なバックエンドのテスト
print("\n--- バックエンド対応状況 ---")

backends_to_test = []
if platform.system() == "Darwin":
    backends_to_test = [
        ("Metal (GPU)", ti.metal),
        ("Vulkan (GPU)", ti.vulkan),
        ("CPU", ti.cpu),
    ]
else:
    backends_to_test = [
        ("CUDA (GPU)", ti.cuda),
        ("Vulkan (GPU)", ti.vulkan),
        ("CPU", ti.cpu),
    ]

for name, arch in backends_to_test:
    try:
        ti.init(arch=arch, default_fp=ti.f32, log_level=ti.WARN)
        # 簡単なカーネルでテスト
        test_field = ti.field(dtype=ti.f32, shape=1000)

        @ti.kernel
        def test_kernel():
            for i in test_field:
                test_field[i] = ti.cast(i, ti.f32) * 2.0

        test_kernel()
        result = test_field.to_numpy()
        ok = abs(result[500] - 1000.0) < 0.1
        print(f"  {name}: ✅ 動作OK")
        ti.reset()
    except Exception as e:
        print(f"  {name}: ❌ 利用不可 ({e})")
        try:
            ti.reset()
        except:
            pass

# auto選択時にどのバックエンドが選ばれるかチェック
print("\n--- 自動バックエンド選択 ---")
if platform.system() == "Darwin":
    try:
        ti.init(arch=ti.metal, default_fp=ti.f32, log_level=ti.WARN)
        print(f"  auto → Metal (GPU) が選択されます")
        
        # 実際のランタイム情報を取得
        try:
            impl = ti.lang.impl.get_runtime()
            prog = impl.prog
            if hasattr(prog, 'config'):
                config = prog.config()
                print(f"  Arch: {config.arch}")
        except:
            pass
        ti.reset()
    except:
        print("  auto → CPU (フォールバック)")
        ti.reset()
else:
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32, log_level=ti.WARN)
        print(f"  auto → CUDA (GPU) が選択されます")
        ti.reset()
    except:
        print("  auto → CPU (フォールバック)")
        try:
            ti.reset()
        except:
            pass

# ベンチマーク: GPU vs CPU 比較
print("\n--- ベンチマーク (GPU vs CPU) ---")
import time

def benchmark(arch, arch_name, n=500000):
    """簡易ベンチマーク: 要素数nのmatvec相当の演算"""
    try:
        ti.init(arch=arch, default_fp=ti.f32, log_level=ti.WARN)
        
        a = ti.field(dtype=ti.f32, shape=n)
        b = ti.field(dtype=ti.f32, shape=n)
        c = ti.field(dtype=ti.f32, shape=n)
        
        @ti.kernel
        def compute():
            for i in range(n):
                val = 0.0
                for j in ti.static(range(8)):
                    val += ti.sin(a[i] + ti.cast(j, ti.f32)) * b[i]
                c[i] = val
        
        # ウォームアップ (JITコンパイル)
        a.fill(1.0)
        b.fill(2.0)
        compute()
        ti.sync()
        
        # 計測
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            compute()
            ti.sync()
            times.append(time.perf_counter() - t0)
        
        avg_time = sum(times) / len(times)
        print(f"  {arch_name}: {avg_time*1000:.2f} ms (平均, N={n:,})")
        ti.reset()
        return avg_time
    except Exception as e:
        print(f"  {arch_name}: テスト失敗 ({e})")
        try:
            ti.reset()
        except:
            pass
        return None

gpu_time = None
cpu_time = None

if platform.system() == "Darwin":
    gpu_time = benchmark(ti.metal, "Metal (GPU)")
else:
    gpu_time = benchmark(ti.cuda, "CUDA (GPU)")

cpu_time = benchmark(ti.cpu, "CPU")

if gpu_time and cpu_time:
    speedup = cpu_time / gpu_time
    print(f"\n  GPU速度比: {speedup:.1f}x {'高速 ✅' if speedup > 1.5 else '(GPUの方が遅い ⚠️)'}")
    if speedup < 1.5:
        print("  ⚠️ GPUがCPUより速くない場合、問題サイズが小さすぎる可能性があります")
elif gpu_time is None:
    print("\n  ⚠️ GPUバックエンドが利用できません - CPUで実行されています")

# 現在のプロジェクト設定の確認
print("\n--- プロジェクト設定の確認 ---")
print("  cli.py の _select_backend():")
if platform.system() == "Darwin":
    print("    macOS → ti.metal (GPU) が選択されます")
else:
    print("    Linux/Windows → ti.cuda (GPU) が試行されます")
print("    --backend cpu で強制的にCPU実行も可能")

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)
