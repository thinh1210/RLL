#!/usr/bin/env python3
"""
MINH HỌA TRỰC QUAN CÁCH CÁC TOÁN TỬ XÁC SUẤT THỐNG KÊ & ĐẠI SỐ TUYẾN TÍNH TÁC ĐỘNG LÊN MẪU
==========================================================================================
Tài liệu đi kèm: TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md

Kịch bản này chạy 6 thí nghiệm số học trực quan trên các tập mẫu dữ liệu (Samples)
để thấy rõ từng công thức toán học biến đổi mẫu và làm nổi bật tính chất gì:
  1. [Kỳ vọng & Luật Số Lớn] Cách Trung bình mẫu khử nhiễu theo quy luật σ/√N.
  2. [Bất đẳng thức Jensen] Cách toán tử max(Q) chọn nhầm mẫu nhiễu dương (Overestimation)
     và cách Double Q-Learning triệt tiêu độ chệch này.
  3. [Importance Sampling & Clipping] Tái sử dụng mẫu của Policy cũ để tính kỳ vọng cho
     Policy mới, và hiện tượng nổ phương sai khi tỉ số r_t(θ) lớn.
  4. [Entropy & KL Divergence vs. Euclid] Tại sao khoảng cách tham số |Δμ| không phản ánh
     đúng sự thay đổi đám mây mẫu bằng Phân kỳ KL & Ma trận Fisher.
  5. [Đại số Tuyến tính: Hiệp phương sai, Trị riêng & Vector riêng] Bóc tách "Trục xương sống"
     (Principal Axes) từ một đám mây mẫu 2 chiều có tương quan chéo.
  6. [Chuẩn L1, L2, Huber & L_∞] Sự khác biệt khi đối mặt với mẫu nhiễu dị biệt (Outlier)
     và tính chất co (Contraction) của toán tử Bellman trên chuẩn L_∞.
"""

import math
import random
from typing import List, Tuple


def exp1_expectation_and_law_of_large_numbers() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 1: KỲ VỌNG E[X], PHƯƠNG SAI Var(X) & ĐỊNH LUẬT SỐ LỚN TRÊN MẪU")
    print("=" * 88)
    rng = random.Random(42)
    true_q_value = 5.00   # Giá trị thật Q*(s, a) = 5.0
    noise_std = 4.00      # Môi trường rất nhiều nhiễu (σ = 4.0)

    print(f"• Giá trị thật ẩn dưới môi trường : Q*(s, a) = {true_q_value:.2f}")
    print(f"• Nhiễu ngẫu nhiên mỗi lần lấy mẫu: ε ~ N(0, σ²={noise_std**2:.1f}) (Độ lệch chuẩn σ = {noise_std:.2f})\n")

    sample_sizes = [1, 5, 25, 100, 1000, 10000]
    samples: List[float] = []
    for n in range(1, max(sample_sizes) + 1):
        samples.append(true_q_value + rng.gauss(0.0, noise_std))
        if n in sample_sizes:
            mean_val = sum(samples) / n
            var_val = sum((x - mean_val) ** 2 for x in samples) / n
            theoretical_se = noise_std / math.sqrt(n)
            abs_err = abs(mean_val - true_q_value)
            print(
                f"  Số mẫu N = {n:5d} | Trung bình mẫu x̄ = {mean_val:+6.3f} "
                f"(Sai số |x̄ - Q*| = {abs_err:.3f}, Lý thuyết σ/√N = {theoretical_se:.3f}) | "
                f"Phương sai mẫu s² = {var_val:6.2f}"
            )

    print("\n💡 NÊU BẬT ĐIỀU GÌ TRÊN MẪU?")
    print("  - Khi N=1..5: Mẫu nhảy loạn xạ do nhiễu chi phối.")
    print("  - Khi N tăng gấp 100 lần: Sai số của trung bình mẫu giảm đúng √100 = 10 lần (hội tụ về 5.00),")
    print("    trong khi Phương sai mẫu s² hội tụ về đúng σ² = 16.0 (đo độ bung tỏa nội tại của môi trường).\n")


def exp2_jensens_inequality_and_double_q() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 2: BẤT ĐẲNG THỨC JENSEN E[max(X)] >= max(E[X]) & DOUBLE DQN")
    print("=" * 88)
    rng = random.Random(123)
    num_actions = 8
    true_q = 0.0  # Tất cả 8 hành động đều có giá trị thật Q*(s, a_k) = 0.0!
    sample_noise_std = 1.0
    num_trials = 2000

    standard_max_estimates: List[float] = []
    double_q_estimates: List[float] = []

    for _ in range(num_trials):
        # Tập mẫu ước lượng Q_A (Online net) và Q_B (Target net độc lập nhiễu)
        q_a = [true_q + rng.gauss(0.0, sample_noise_std) for _ in range(num_actions)]
        q_b = [true_q + rng.gauss(0.0, sample_noise_std) for _ in range(num_actions)]

        # 1. DQN chuẩn: Dùng cùng 1 mẫu Q_A để vừa tìm max vừa lấy giá trị
        standard_max_estimates.append(max(q_a))

        # 2. Double DQN: Dùng Q_A để chọn vị trí a* = argmax(Q_A), nhưng lấy giá trị trên Q_B[a*]
        best_action_idx = max(range(num_actions), key=lambda i: q_a[i])
        double_q_estimates.append(q_b[best_action_idx])

    mean_std_max = sum(standard_max_estimates) / num_trials
    mean_double_q = sum(double_q_estimates) / num_trials

    print(f"• Giá trị thật của mọi hành động (K = {num_actions} actions): Q*(s, a_1..8) = 0.000")
    print(f"• Trung bình mẫu của Standard DQN E[max_a Q_A(s,a)]     : {mean_std_max:+.4f}  <-- Bị phóng đại ảo (Overestimation Bias)!")
    print(f"• Trung bình mẫu của Double DQN   E[Q_B(s, argmax Q_A)] : {mean_double_q:+.4f}  <-- Khử sạch nhiễu dương về ~0.000!")

    print("\n💡 NÊU BẬT ĐIỀU GÌ TRÊN MẪU?")
    print("  - Toán tử `max` chọn lọc thiên vị mẫu nào vô tình có nhiễu dương lớn nhất trong 8 nhánh.")
    print("  - Khi tách mẫu chọn (Q_A) và mẫu chấm điểm (Q_B), nhiễu trên Q_B có kỳ vọng bằng 0 nên triệt tiêu hoàn toàn ảo tưởng giá trị!\n")


def exp3_importance_sampling_and_ppo_clip() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 3: IMPORTANCE SAMPLING (IS) & HIỆN TƯỢNG NỔ PHƯƠNG SAI TRONG PPO")
    print("=" * 88)
    rng = random.Random(7)

    # Giả sử Policy cũ Q = N(μ_old = 0.0, σ = 1.0), Policy mới P = N(μ_new = 0.4, σ = 1.0)
    # Ta muốn ước lượng kỳ vọng của hàm thưởng f(a) = 2.0 * a dưới Policy mới (Giá trị thật = 2.0 * 0.4 = +0.80)
    mu_old, mu_new, sigma = 0.0, 0.4, 1.0
    true_target_expectation = 2.0 * mu_new

    def gaussian_pdf(x: float, mu: float, s: float) -> float:
        return (1.0 / (math.sqrt(2.0 * math.pi) * s)) * math.exp(-0.5 * ((x - mu) / s) ** 2)

    n_samples = 1000
    old_samples = [rng.gauss(mu_old, sigma) for _ in range(n_samples)]

    naive_old_avg = sum(2.0 * x for x in old_samples) / n_samples
    is_weighted_values = []
    ppo_clipped_values = []
    ratios = []

    for x in old_samples:
        p_new = gaussian_pdf(x, mu_new, sigma)
        p_old = gaussian_pdf(x, mu_old, sigma)
        ratio = p_new / p_old  # r_t(θ) = π_new(a|s) / π_old(a|s)
        ratios.append(ratio)

        f_x = 2.0 * x
        is_weighted_values.append(ratio * f_x)
        clipped_ratio = max(0.8, min(1.2, ratio))
        ppo_clipped_values.append(clipped_ratio * f_x)

    is_estimate = sum(is_weighted_values) / n_samples
    ppo_estimate = sum(ppo_clipped_values) / n_samples

    print(f"• Giá trị kỳ vọng thật dưới Policy mới E_P[2a] : {true_target_expectation:+.4f}")
    print(f"• Trung bình thô trên mẫu cũ (chưa nhân IS)   : {naive_old_avg:+.4f} (Sai lệch hoàn toàn vì dùng μ_old=0)")
    print(f"• Ước lượng sau khi nhân trọng số IS r_t(θ)   : {is_estimate:+.4f} (Khôi phục chính xác kỳ vọng của Policy mới!)")
    print(f"• Ước lượng có cắt tỉa PPO Clip [0.8, 1.2]    : {ppo_estimate:+.4f} (Cận dưới thận trọng, phương sai cực thấp)")
    print(f"• Tỉ số trọng số mẫu r_t(θ)                   : min = {min(ratios):.2f}, max = {max(ratios):.2f}")

    print("\n💡 NÊU BẬT ĐIỀU GÌ TRÊN MẪU?")
    print("  - Importance Sampling không thay đổi giá trị mẫu x_i, mà 'phóng to trọng lượng' (r_i > 1) cho những")
    print("    mẫu nằm gần μ_new = 0.4 và 'dìm trọng lượng' (r_i < 1) những mẫu nằm ở phía âm.\n")


def exp4_kl_divergence_vs_euclidean_distance() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 4: TẠI SAO DÙNG PHÂN KỲ KL & MA TRẬN FISHER THAY VÌ KHOẢNG CÁCH EUCLID?")
    print("=" * 88)

    # Cùng dịch chuyển tham số tâm Δμ = 0.50 trên 2 trường hợp độ rộng đám mây mẫu σ khác nhau:
    # Công thức KL giữa 2 Gaussian cùng σ: D_KL(N(μ1, σ²) || N(μ2, σ²)) = (μ1 - μ2)² / (2 * σ²)
    delta_mu = 0.50
    for case_name, sigma in [("Đám mây mẫu rộng (σ = 2.0 - Đang khám phá)", 2.0),
                             ("Đám mây mẫu nhọn (σ = 0.05 - Đã hội tụ cao)", 0.05)]:
        euclid_dist = abs(delta_mu)
        kl_div = (delta_mu ** 2) / (2.0 * (sigma ** 2))
        fisher_info = 1.0 / (sigma ** 2)  # F_μμ = 1 / σ²
        natural_step = (1.0 / fisher_info) * delta_mu

        print(f"  • {case_name}:")
        print(f"      Khoảng cách tham số Euclid |Δμ| = {euclid_dist:.2f} (Giống hệt nhau)")
        print(f"      Độ cong Thông tin Fisher F_μμ   = {fisher_info:8.2f}")
        print(f"      Phân kỳ KL thực tế D_KL(P || Q) = {kl_div:8.4f} nats "
              f"({'Đám mây mẫu gần như giữ nguyên' if kl_div < 0.1 else '⚠️ HAI ĐÁM MÂY MẪU TÁCH RỜI HOÀN TOÀN (Sụp đổ Policy)!'})")

    print("\n💡 NÊU BẬT ĐIỀU GÌ TRÊN MẪU?")
    print("  - Khoảng cách Euclid |θ_new - θ_old| hoàn toàn 'mù' trước độ rộng σ của đám mây mẫu.")
    print("  - Phân kỳ KL và Ma trận Fisher (F = 1/σ²) đo độ chồng lấp thực sự giữa 2 đám mây mẫu!\n")


def exp5_linear_algebra_pca_and_eigenvalues() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 5: ĐẠI SỐ TUYẾN TÍNH — MA TRẬN HIỆP PHƯƠNG SAI, TRỊ RIÊNG & VECTOR RIÊNG")
    print("=" * 88)
    rng = random.Random(99)

    # Tạo đám mây N=500 mẫu 2 chiều (x1, x2) có trục chính nghiêng góc θ = 30° (π/6 rad)
    # Phương sai dọc trục chính λ_1 = 9.0 (σ_1 = 3.0), dọc trục phụ vuông góc λ_2 = 0.25 (σ_2 = 0.5)
    true_angle_deg = 30.0
    theta = math.radians(true_angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    n_samples = 500
    samples_2d: List[Tuple[float, float]] = []
    for _ in range(n_samples):
        # Mẫu gốc trên hệ trục chuẩn (z1 ~ N(0, 3²), z2 ~ N(0, 0.5²))
        z1 = rng.gauss(0.0, 3.0)
        z2 = rng.gauss(0.0, 0.5)
        # Nhân ma trận xoay R(30°) để biến hình đám mây mẫu
        x1 = cos_t * z1 - sin_t * z2
        x2 = sin_t * z1 + cos_t * z2
        samples_2d.append((x1, x2))

    # 1. Tính trung bình mẫu và Ma trận Hiệp phương sai 2x2 Σ
    m1 = sum(p[0] for p in samples_2d) / n_samples
    m2 = sum(p[1] for p in samples_2d) / n_samples
    s11 = sum((p[0] - m1) ** 2 for p in samples_2d) / n_samples
    s22 = sum((p[1] - m2) ** 2 for p in samples_2d) / n_samples
    s12 = sum((p[0] - m1) * (p[1] - m2) for p in samples_2d) / n_samples

    # 2. Giải phương trình đặc trưng det(Σ - λI) = 0 cho ma trận 2x2 đối xứng
    trace = s11 + s22
    det = s11 * s22 - s12 * s12
    disc = math.sqrt(max(0.0, trace * trace - 4.0 * det))
    lambda_1 = 0.5 * (trace + disc)  # Trị riêng lớn nhất
    lambda_2 = 0.5 * (trace - disc)  # Trị riêng nhỏ nhất

    # Vector riêng chính v_1 ứng với λ_1
    vx, vy = s12, lambda_1 - s11
    norm_v = math.hypot(vx, vy)
    vx, vy = vx / norm_v, vy / norm_v
    recovered_angle_deg = math.degrees(math.atan2(vy, vx))
    if recovered_angle_deg < 0:
        recovered_angle_deg += 180.0

    print(f"• Ma trận Hiệp phương sai tính từ {n_samples} mẫu:")
    print(f"    Σ = [[ {s11:6.3f}, {s12:6.3f} ],")
    print(f"         [ {s12:6.3f}, {s22:6.3f} ]]  (Cov(x1, x2) = {s12:+.3f} > 0 -> Hai biến đồng biến mạnh)")
    print(f"• Trị riêng tìm được    : λ_1 = {lambda_1:.3f} (Lý thuyết: 9.000), λ_2 = {lambda_2:.3f} (Lý thuyết: 0.250)")
    print(f"• Vector riêng chính v1 : [{vx:+.3f}, {vy:+.3f}] -> Góc nghiêng = {recovered_angle_deg:.1f}° (Góc thật: {true_angle_deg:.1f}°)")
    print(f"• Tỷ lệ thông tin giữ lại bởi trục chính v1: {lambda_1 / (lambda_1 + lambda_2) * 100:.1f}%")

    print("\n💡 NÊU BẬT ĐIỀU GÌ TRÊN MẪU?")
    print("  - Vector riêng v1 tự động xoay hệ trục tọa độ để bắt trúng 'sống lưng' của đám mây dữ liệu.")
    print("  - Trị riêng λ1 cho biết 97% sự biến thiên của mẫu nằm trên trục v1 -> Có thể nén 2 chiều về 1 chiều!\n")


def exp6_norms_and_outliers() -> None:
    print("=" * 88)
    print("🧪 THÍ NGHIỆM 6: SO SÁNH CHUẨN L1, L2 (MSE), HUBER & L_∞ KHI MẪU CÓ NHIỄU DỊ BIỆT (OUTLIER)")
    print("=" * 88)

    # Giả sử 5 mẫu TD-error trong Replay Buffer: 4 mẫu nhỏ bình thường và 1 mẫu bị lỗi cảm biến vọt lên +50.0
    td_errors = [0.2, -0.3, 0.1, -0.2, 50.0]

    l1_norm = sum(abs(e) for e in td_errors)
    l2_norm = math.sqrt(sum(e * e for e in td_errors))
    l_inf_norm = max(abs(e) for e in td_errors)

    # Đạo hàm kéo trọng số của mẫu Outlier (e = 50.0) so với mẫu thường (e = 0.2):
    grad_l1_outlier = 1.0
    grad_l2_outlier = 2.0 * 50.0  # = 100.0 (Gấp 250 lần mẫu thường 2*0.2 = 0.4!)

    print(f"• Tập mẫu sai số TD-Error : {td_errors} (Mẫu thứ 5 là Outlier = 50.0)")
    print(f"• Chuẩn L1   (||e||_1)    : {l1_norm:6.2f} | Lực kéo Gradient của mẫu Outlier = {grad_l1_outlier:5.1f} (Bền vững, không nổ)")
    print(f"• Chuẩn L2   (||e||_2)    : {l2_norm:6.2f} | Lực kéo Gradient MSE của Outlier = {grad_l2_outlier:5.1f} (⚠️ Làm nổ Gradient!)")
    print(f"• Chuẩn L_∞  (||e||_∞)    : {l_inf_norm:6.2f} | Chỉ nhìn duy nhất vào mẫu tệ nhất (Dùng cho Định lý Co Bellman)")
    print("=" * 88)


if __name__ == "__main__":
    exp1_expectation_and_law_of_large_numbers()
    exp2_jensens_inequality_and_double_q()
    exp3_importance_sampling_and_ppo_clip()
    exp4_kl_divergence_vs_euclidean_distance()
    exp5_linear_algebra_pca_and_eigenvalues()
    exp6_norms_and_outliers()
