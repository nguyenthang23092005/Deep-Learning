import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path

# Đường dẫn
DATA_DIR = "data_GK"
OUTPUT_DIR = "visual"

# Các lệnh cần so sánh
COMMANDS = [
    'tang_nhiet_do',
    'giam_nhiet_do',
    'mo_cua',
    'dong_cua',
    'bat_thong_bao_chay',
    'tat_thong_bao_chay'
]

def load_all_spectrograms(command):
    """Load TẤT CẢ spectrograms của một lệnh"""
    folder = os.path.join(DATA_DIR, command)
    npy_files = list(Path(folder).glob('*.npy'))
    
    if len(npy_files) == 0:
        return []
    
    specs = []
    for npy_file in npy_files:
        spec = np.load(str(npy_file))
        specs.append(spec)
    
    return specs

def compare_all_commands():
    """So sánh spectrograms của tất cả các lệnh - TẠO 10 ẢNH RANDOM"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tạo 10 ảnh với random samples khác nhau
    num_images = 10
    
    for img_num in range(1, num_images + 1):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, command in enumerate(COMMANDS):
            specs = load_all_spectrograms(command)
            
            if len(specs) > 0:
                # Lấy spectrogram NGẪU NHIÊN để hiển thị
                spec = random.choice(specs)
                im = axes[idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                axes[idx].set_title(f"{command}\n{len(specs)} samples", fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Time frames')
                axes[idx].set_ylabel('Mel frequency bins')
                
                # Thêm colorbar
                plt.colorbar(im, ax=axes[idx])
                
                # In thống kê (chỉ lần đầu)
                if img_num == 1:
                    print(f"\n{command}: {len(specs)} samples")
                    print(f"  Shape: {spec.shape}")
                    print(f"  Min: {spec.min():.2f}, Max: {spec.max():.2f}")
                    print(f"  Mean: {spec.mean():.2f}, Std: {spec.std():.2f}")
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[idx].set_title(command)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'comparison_all_commands_{img_num}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu ảnh {img_num}/{num_images}: {output_path}")
    
    print(f"\n✅ Hoàn thành! Đã tạo {num_images} ảnh với random samples")

def compute_intra_inter_class_variance():
    """Tính variance trong cùng class và giữa các class"""
    
    print("\n" + "="*60)
    print("PHÂN TÍCH VARIANCE (INTRA-CLASS vs INTER-CLASS)")
    print("="*60)
    
    all_specs = {}
    
    # Load tất cả spectrograms
    for command in COMMANDS:
        specs = load_all_spectrograms(command)
        if len(specs) > 0:
            all_specs[command] = specs
            print(f"Loaded {len(specs)} samples from {command}")
    
    # Tính INTRA-CLASS variance (variance trong cùng lệnh)
    print("\n📊 INTRA-CLASS VARIANCE (Variance trong cùng lệnh):")
    intra_variances = {}
    
    for command, specs in all_specs.items():
        if len(specs) < 2:
            continue
        
        # Tính distance trung bình giữa các mẫu trong cùng class
        distances = []
        for i in range(len(specs)):
            for j in range(i+1, len(specs)):
                dist = np.linalg.norm(specs[i] - specs[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances) if distances else 0
        intra_variances[command] = avg_dist
        print(f"  {command}: {avg_dist:.2f} (avg distance between {len(distances)} pairs)")
    
    avg_intra = np.mean(list(intra_variances.values()))
    print(f"\n  ⭐ TRUNG BÌNH INTRA-CLASS: {avg_intra:.2f}")
    
    # Tính INTER-CLASS variance (variance giữa các lệnh khác nhau)
    print("\n📊 INTER-CLASS VARIANCE (Variance giữa các lệnh):")
    inter_distances = []
    
    commands_list = list(all_specs.keys())
    for i in range(len(commands_list)):
        for j in range(i+1, len(commands_list)):
            cmd1, cmd2 = commands_list[i], commands_list[j]
            
            # Tính average distance giữa tất cả cặp samples của 2 class
            distances = []
            for spec1 in all_specs[cmd1]:
                for spec2 in all_specs[cmd2]:
                    dist = np.linalg.norm(spec1 - spec2)
                    distances.append(dist)
            
            avg_dist = np.mean(distances)
            inter_distances.append(avg_dist)
            print(f"  {cmd1} vs {cmd2}: {avg_dist:.2f}")
    
    avg_inter = np.mean(inter_distances)
    print(f"\n  ⭐ TRUNG BÌNH INTER-CLASS: {avg_inter:.2f}")
    
    # Tính separability ratio
    separability = avg_inter / avg_intra if avg_intra > 0 else 0
    
    print("\n" + "="*60)
    print("KẾT LUẬN:")
    print("="*60)
    print(f"📈 Intra-class distance (càng nhỏ càng tốt): {avg_intra:.2f}")
    print(f"📈 Inter-class distance (càng lớn càng tốt): {avg_inter:.2f}")
    print(f"📈 Separability ratio (Inter/Intra): {separability:.2f}")
    print()
    
    if separability > 2.0:
        print("✅ XUẤT SẮC! Classes rất phân biệt rõ ràng, model sẽ học tốt!")
    elif separability > 1.5:
        print("✅ TỐT! Classes có sự phân biệt đủ rõ ràng.")
    elif separability > 1.0:
        print("⚠️  TRUNG BÌNH. Classes có overlap, cần cải thiện preprocessing.")
    else:
        print("❌ KÉM! Classes quá overlap, cần thay đổi feature extraction.")
    
    return intra_variances, inter_distances, separability

def compute_difference_matrix():
    """Tính ma trận khác biệt giữa các lệnh (dùng tất cả samples)"""
    all_specs = {}
    
    # Load tất cả spectrograms
    for command in COMMANDS:
        specs = load_all_spectrograms(command)
        if len(specs) > 0:
            all_specs[command] = specs
    
    # Tính ma trận difference (average distance giữa các class)
    n = len(COMMANDS)
    diff_matrix = np.zeros((n, n))
    
    for i, cmd1 in enumerate(COMMANDS):
        for j, cmd2 in enumerate(COMMANDS):
            if cmd1 in all_specs and cmd2 in all_specs:
                # Tính average distance giữa tất cả cặp samples
                distances = []
                for spec1 in all_specs[cmd1][:10]:  # Lấy 10 samples đầu để nhanh
                    for spec2 in all_specs[cmd2][:10]:
                        dist = np.linalg.norm(spec1 - spec2)
                        distances.append(dist)
                diff_matrix[i, j] = np.mean(distances)
    
    # Visualize difference matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(diff_matrix, cmap='hot', aspect='auto')
    plt.colorbar(im, label='Average Euclidean Distance')
    plt.xticks(range(n), COMMANDS, rotation=45, ha='right')
    plt.yticks(range(n), COMMANDS)
    plt.title('Average Distance Matrix Between Commands\n(Computed from all samples)', fontsize=14, fontweight='bold')
    
    # Thêm giá trị vào ô
    for i in range(n):
        for j in range(n):
            text = plt.text(j, i, f'{diff_matrix[i, j]:.0f}',
                          ha="center", va="center", color="white" if diff_matrix[i, j] > diff_matrix.max()/2 else "black",
                          fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'difference_matrix.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Đã lưu ma trận khác biệt: {output_path}")
    plt.show()
    
    # In phân tích
    print("\n" + "="*60)
    print("PHÂN TÍCH MA TRẬN KHÁC BIỆT:")
    print("="*60)
    
    # Tìm cặp giống nhất và khác nhất
    mask = np.ones_like(diff_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    
    most_similar_idx = np.unravel_index(np.argmin(diff_matrix + np.diag([1e10]*n)), diff_matrix.shape)
    most_different_idx = np.unravel_index(np.argmax(diff_matrix * mask), diff_matrix.shape)
    
    print(f"\n✅ Cặp GIỐNG NHẤT:")
    print(f"   {COMMANDS[most_similar_idx[0]]} vs {COMMANDS[most_similar_idx[1]]}")
    print(f"   Distance: {diff_matrix[most_similar_idx]:.2f}")
    
    print(f"\n✅ Cặp KHÁC NHAU NHẤT:")
    print(f"   {COMMANDS[most_different_idx[0]]} vs {COMMANDS[most_different_idx[1]]}")
    print(f"   Distance: {diff_matrix[most_different_idx]:.2f}")
    
    print(f"\n📊 Trung bình distance: {diff_matrix[mask].mean():.2f}")
    print(f"📊 Std distance: {diff_matrix[mask].std():.2f}")

if __name__ == "__main__":
    print("="*60)
    print("SO SÁNH SPECTROGRAMS - TOÀN BỘ DATASET")
    print("="*60)
    
    # So sánh visual
    compare_all_commands()
    
    # Tính intra-class và inter-class variance
    compute_intra_inter_class_variance()
    
    # Tính ma trận khác biệt
    compute_difference_matrix()
    
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
