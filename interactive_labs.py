#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_labs.py
===================
Module cung cấp 10 Phòng Thí Nghiệm Mô Phỏng Tương Tác (Interactive HTML5 Canvas Labs)
được nhúng trực tiếp vào các tệp HTML của kho lưu trữ Reinforcement-Learning.

Người học có thể kéo thanh trượt (sliders) thay đổi tham số thực tế và quan sát:
1. Lab Toán Học (XSTK & ĐSTT): Đám mây mẫu 2D, Ma trận Hiệp phương sai Σ, Trị riêng λ_1, λ_2, Vector riêng v_1, v_2 & So sánh chuẩn L1/L2/L∞ khi có mẫu nhiễu dị biệt.
2. Lab Week 1 (Bandits): UCB1 vs ε-Greedy, bán kính tin cậy Hoeffding c*sqrt(ln t / N) và bước học α.
3. Lab Week 2 (MDP & Bellman): GridWorld 4x4, lan truyền giá trị Bellman V(s) và Q-Learning theo γ, α, độ trơn trượt.
4. Lab Week 3 (Deep Q-Networks): Hiện tượng Maximization Bias E[max X] = σ/sqrt(π) (Bất đẳng thức Jensen) và cách Double DQN khử độ chệch về 0.
5. Lab Week 4 (Policy Gradient): Giảm phương sai bằng Reward-to-Go & Baseline b(s)=V(s), và tác động của Entropy Bonus β*H(π).
6. Lab Week 5 (PPO & GAE): Đồ thị hàm mục tiêu kẹp L^CLIP theo ngưỡng ε và Advantage Â_t, cùng chuỗi trọng số (γλ)^l của GAE.
7. Lab Week 6 (Evolution Strategies): Mặt đồng mức 2D, đám mây nhiễu đối xứng ±ε (Mirrored Sampling) khử bậc chẵn Taylor và bước nhảy cập nhật.
8. Lab Week 7 (Model-Based RL & MPC): Sự bùng nổ sai số tích lũy O(L_f^H * ε) theo tầm nhìn H, hằng số Lipschitz L_f và làm trơn Tikhonov σ_ξ.
9. Lab Week 8 (SAC MaxEnt & RLHF/DPO): Phân phối Boltzmann đa đỉnh π*(a|s) ∝ exp(Q/α) theo nhiệt độ α và hàm mất mát DPO theo β_KL.
10. Lab Sim-to-Real Robotics: Động lực học Robot vi sai 2 bánh, so sánh nghiệm số Euler vs RK4 khi tăng Δt & độ cứng K_p, ma sát μ và Lá chắn an toàn LiDAR.
"""

# Ánh xạ từ doc_id trong build_html_docs.py sang tab mặc định của Lab
DOC_DEFAULT_LAB = {
    "home": "ppo_gae",
    "math_symbols": "math_eigen",
    "math_foundations": "math_eigen",
    "param_handbook": "ppo_gae",
    "prereq_roadmap": "math_eigen",
    "weekly_summary": "ppo_gae",
    "week1": "bandit_ucb",
    "week2": "mdp_bellman",
    "week3": "dqn_bias",
    "week4": "pg_variance",
    "week5": "ppo_gae",
    "week6": "es_mirror",
    "week7": "mbrl_mpc",
    "week8": "sac_dpo",
    "sim_guide": "sim2real_physics",
    "sim2real_overview": "sim2real_physics",
    "sim2real_p1": "sim2real_physics",
    "sim2real_p2": "sim2real_physics",
    "sim2real_p3": "sim2real_physics",
    "sim2real_p4": "sim2real_physics",
    "sim2real_p5": "sim2real_physics",
}


def get_interactive_lab_html(doc_id: str) -> str:
    default_tab = DOC_DEFAULT_LAB.get(doc_id, "ppo_gae")
    return INTERACTIVE_LAB_WIDGET.replace("__DEFAULT_LAB_TAB__", default_tab)


INTERACTIVE_LAB_WIDGET = r"""
<section class="sim-lab-wrapper" id="interactive-simulator-lab">
  <div class="sim-lab-header">
    <div class="sim-lab-title-row">
      <div>
        <span class="sim-lab-badge">🎛️ INTERACTIVE SIMULATION LAB</span>
        <h2 class="sim-lab-heading">Phòng Thí Nghiệm Mô Phỏng &amp; Tinh Chỉnh Tham Số Trực Quan</h2>
        <p class="sim-lab-subtitle">
          Kéo các thanh trượt (Sliders) bên dưới để thay đổi tham số toán học — đồ thị, quỹ đạo và phương trình sẽ phản hồi ngay lập tức theo thời gian thực.
        </p>
      </div>
      <button class="btn-action" onclick="toggleSimLabCollapse()" id="sim-collapse-btn">Thu gọn / Mở rộng Lab</button>
    </div>

    <div class="sim-tabs" role="tablist">
      <button class="sim-tab" data-tab="math_eigen" onclick="switchSimTab('math_eigen')">📐 Toán: Σ &amp; Trị riêng</button>
      <button class="sim-tab" data-tab="bandit_ucb" onclick="switchSimTab('bandit_ucb')">🎰 W1: Bandit &amp; UCB1</button>
      <button class="sim-tab" data-tab="mdp_bellman" onclick="switchSimTab('mdp_bellman')">🧭 W2: Bellman &amp; Q-Grid</button>
      <button class="sim-tab" data-tab="dqn_bias" onclick="switchSimTab('dqn_bias')">🎯 W3: Double DQN Bias</button>
      <button class="sim-tab" data-tab="pg_variance" onclick="switchSimTab('pg_variance')">📉 W4: PG Baseline &amp; Entropy</button>
      <button class="sim-tab" data-tab="ppo_gae" onclick="switchSimTab('ppo_gae')">🛡️ W5: PPO Clip &amp; GAE</button>
      <button class="sim-tab" data-tab="es_mirror" onclick="switchSimTab('es_mirror')">🧬 W6: ES Mirrored ±ε</button>
      <button class="sim-tab" data-tab="mbrl_mpc" onclick="switchSimTab('mbrl_mpc')">🔮 W7: Model Error &amp; MPC</button>
      <button class="sim-tab" data-tab="sac_dpo" onclick="switchSimTab('sac_dpo')">🔥 W8: SAC α &amp; DPO β</button>
      <button class="sim-tab" data-tab="sim2real_physics" onclick="switchSimTab('sim2real_physics')">🤖 Sim2Real: Euler vs RK4</button>
    </div>
  </div>

  <div class="sim-lab-body" id="sim-lab-body">
    <!-- ==================== 1. MATH: COVARIANCE & EIGEN ==================== -->
    <div class="sim-panel" id="panel-math_eigen">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>📐 Đám Mây Mẫu 2D, Ma trận Σ &amp; Trị Riêng (λ₁, λ₂)</h3>
          <p class="sim-desc">Quan sát cách độ lệch chuẩn $\sigma_x, \sigma_y$ và góc xoay tương quan $\theta$ biến đổi đám mây mẫu và 2 trục Vector riêng trực giao $\mathbf{v}_1, \mathbf{v}_2$.</p>
          <label class="sim-ctrl">
            <span>Độ trải trục chính $\sigma_1$: <strong id="val-math-s1">1.80</strong></span>
            <input type="range" id="sl-math-s1" min="0.3" max="2.5" step="0.05" value="1.80" oninput="updateMathLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Độ trải trục phụ $\sigma_2$: <strong id="val-math-s2">0.55</strong></span>
            <input type="range" id="sl-math-s2" min="0.15" max="2.0" step="0.05" value="0.55" oninput="updateMathLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Góc xoay tương quan $\theta$: <strong id="val-math-ang">35°</strong></span>
            <input type="range" id="sl-math-ang" min="-85" max="85" step="1" value="35" oninput="updateMathLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Biên độ mẫu nhiễu dị biệt (Outlier $e_N$): <strong id="val-math-out">0.0</strong></span>
            <input type="range" id="sl-math-out" min="0" max="15" step="0.5" value="0" oninput="updateMathLab()" />
          </label>
          <div class="sim-metrics" id="metrics-math"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-math" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 2. WEEK 1: BANDIT & UCB1 ==================== -->
    <div class="sim-panel" id="panel-bandit_ucb">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🎰 Multi-Armed Bandit: UCB1 vs. ε-Greedy</h3>
          <p class="sim-desc">Công thức UCB1: $a_t = \arg\max_a \big[Q_t(a) + c\sqrt{\ln t / N_t(a)}\big]$. Kéo $c$ để xem bán kính tin cậy màu cam thúc đẩy khám phá có định hướng.</p>
          <label class="sim-ctrl">
            <span>Hệ số tự tin UCB $c$: <strong id="val-w1-c">1.41</strong></span>
            <input type="range" id="sl-w1-c" min="0.0" max="4.0" step="0.05" value="1.41" oninput="updateW1Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Tỉ lệ khám phá $\epsilon$ (cho $\epsilon$-greedy): <strong id="val-w1-eps">0.15</strong></span>
            <input type="range" id="sl-w1-eps" min="0.0" max="0.6" step="0.02" value="0.15" oninput="updateW1Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Bước học cập nhật $\alpha$: <strong id="val-w1-alpha">0.15</strong></span>
            <input type="range" id="sl-w1-alpha" min="0.02" max="0.8" step="0.02" value="0.15" oninput="updateW1Lab()" />
          </label>
          <div class="sim-btn-row">
            <button class="btn-sim" onclick="stepW1Bandit(1)">+1 Bước kéo</button>
            <button class="btn-sim" onclick="stepW1Bandit(25)">+25 Bước</button>
            <button class="btn-sim" onclick="stepW1Bandit(200)">+200 Bước</button>
            <button class="btn-sim-sec" onclick="resetW1Bandit()">🔄 Reset</button>
          </div>
          <div class="sim-metrics" id="metrics-w1"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w1" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 3. WEEK 2: MDP BELLMAN & Q-LEARNING ==================== -->
    <div class="sim-panel" id="panel-mdp_bellman">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🧭 FrozenLake 4×4: Toán tử Co Bellman &amp; Hệ số $\gamma$</h3>
          <p class="sim-desc">Phương trình Bellman tối ưu: $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[r + \gamma V^*(s')]$. Kéo $\gamma$ và xác suất trượt băng để xem giá trị lan truyền từ Đích (ô 15, +1) và né Hố băng (H, -1)!</p>
          <label class="sim-ctrl">
            <span>Hệ số chiết khấu $\gamma$: <strong id="val-w2-gamma">0.92</strong></span>
            <input type="range" id="sl-w2-gamma" min="0.10" max="0.99" step="0.01" value="0.92" oninput="updateW2Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Xác suất trượt chân sang bên (`slip_prob`): <strong id="val-w2-slip">0.20</strong></span>
            <input type="range" id="sl-w2-slip" min="0.0" max="0.60" step="0.05" value="0.20" oninput="updateW2Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Số vòng lặp Value Iteration $k$: <strong id="val-w2-iter">15</strong></span>
            <input type="range" id="sl-w2-iter" min="1" max="40" step="1" value="15" oninput="updateW2Lab()" />
          </label>
          <div class="sim-metrics" id="metrics-w2"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w2" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 4. WEEK 3: DOUBLE DQN BIAS ==================== -->
    <div class="sim-panel" id="panel-dqn_bias">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🎯 Thiên Lệch Cực Đại Hóa (Maximization Bias) &amp; Double DQN</h3>
          <p class="sim-desc">Mọi hành động đều có giá trị thật $Q^*(s,a) = 0$, nhưng ước lượng mạng bị nhiễu $\mathcal{N}(0, \sigma^2)$. DQN thường lấy $\max_a Q(s,a;\theta^-)$ nên bị thổi phồng dương, trong khi Double DQN tách chọn &amp; định giá nên khử sạch bias!</p>
          <label class="sim-ctrl">
            <span>Độ lệch chuẩn nhiễu ước lượng $\sigma$: <strong id="val-w3-sigma">1.20</strong></span>
            <input type="range" id="sl-w3-sigma" min="0.1" max="3.0" step="0.1" value="1.20" oninput="updateW3Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Số lượng hành động $|\mathcal{A}|$: <strong id="val-w3-nact">8</strong></span>
            <input type="range" id="sl-w3-nact" min="2" max="30" step="1" value="8" oninput="updateW3Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số chiết khấu $\gamma$ (tích lũy qua $1/(1-\gamma)$): <strong id="val-w3-gamma">0.90</strong></span>
            <input type="range" id="sl-w3-gamma" min="0.50" max="0.98" step="0.02" value="0.90" oninput="updateW3Lab()" />
          </label>
          <div class="sim-btn-row">
            <button class="btn-sim" onclick="updateW3Lab()">🎲 Lấy mẫu Monte Carlo mới (500 lần)</button>
          </div>
          <div class="sim-metrics" id="metrics-w3"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w3" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 5. WEEK 4: POLICY GRADIENT & ENTROPY ==================== -->
    <div class="sim-panel" id="panel-pg_variance">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>📉 Policy Gradient: Baseline $b(s)$ &amp; Entropy Bonus $\beta\mathcal{H}(\pi)$</h3>
          <p class="sim-desc">Quan sát cách trừ Baseline $b(s) = V(s)$ cắt giảm phương sai mẫu của đạo hàm $\nabla_\theta J$, và cách hệ số Entropy $\beta$ chống sụp đổ chính sách sớm (Premature Collapse).</p>
          <label class="sim-ctrl">
            <span>Mức thưởng nền trung bình (Reward Offset): <strong id="val-w4-roff">50</strong></span>
            <input type="range" id="sl-w4-roff" min="0" max="100" step="5" value="50" oninput="updateW4Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số chuẩn hóa Baseline ($0 = \text{Tắt}, 1 = V(s)$): <strong id="val-w4-base">1.00</strong></span>
            <input type="range" id="sl-w4-base" min="0.0" max="1.0" step="0.05" value="1.00" oninput="updateW4Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Trọng số Entropy Bonus $\beta$: <strong id="val-w4-ent">0.15</strong></span>
            <input type="range" id="sl-w4-ent" min="0.01" max="1.50" step="0.02" value="0.15" oninput="updateW4Lab()" />
          </label>
          <div class="sim-metrics" id="metrics-w4"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w4" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 6. WEEK 5: PPO CLIP & GAE ==================== -->
    <div class="sim-panel" id="panel-ppo_gae">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🛡️ PPO-Clip Objective &amp; Trọng Số Suy Giảm GAE($\gamma, \lambda$)</h3>
          <p class="sim-desc">Trái: Hàm mục tiêu $L^{\text{CLIP}}(r_t) = \min\big(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\big)$. Phải: Trọng số $(\gamma\lambda)^l$ của sai số TD $\delta_{t+l}^V$ theo bước tương lai $l$.</p>
          <label class="sim-ctrl">
            <span>Ngưỡng kẹp PPO $\epsilon$ (`CLIP_EPS`): <strong id="val-w5-eps">0.20</strong></span>
            <input type="range" id="sl-w5-eps" min="0.05" max="0.50" step="0.01" value="0.20" oninput="updateW5Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Giá trị Lợi thế $\hat{A}_t$ (Dương = Tốt, Âm = Xấu): <strong id="val-w5-adv">+1.50</strong></span>
            <input type="range" id="sl-w5-adv" min="-2.5" max="2.5" step="0.1" value="1.5" oninput="updateW5Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số GAE $\lambda$ (`GAE_LAMBDA`): <strong id="val-w5-lam">0.95</strong></span>
            <input type="range" id="sl-w5-lam" min="0.0" max="1.0" step="0.01" value="0.95" oninput="updateW5Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số chiết khấu $\gamma$: <strong id="val-w5-gam">0.99</strong></span>
            <input type="range" id="sl-w5-gam" min="0.80" max="0.999" step="0.005" value="0.99" oninput="updateW5Lab()" />
          </label>
          <div class="sim-metrics" id="metrics-w5"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w5" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 7. WEEK 6: EVOLUTION STRATEGIES ==================== -->
    <div class="sim-panel" id="panel-es_mirror">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🧬 Evolution Strategies: Làm Trơn Gaussian &amp; Mirrored Sampling $\pm\varepsilon$</h3>
          <p class="sim-desc">Mũi tên xanh lá là Gradient thật $\nabla F(\theta)$, mũi tên cam là ước lượng ES $\hat{g}$. Bật/tắt lấy mẫu đối xứng $\pm\varepsilon$ để thấy cách nó khử sạch sai số bậc chẵn!</p>
          <label class="sim-ctrl">
            <span>Bán kính nhiễu thăm dò $\sigma$ (`STD_NOISE`): <strong id="val-w6-sigma">0.35</strong></span>
            <input type="range" id="sl-w6-sigma" min="0.05" max="1.00" step="0.05" value="0.35" oninput="updateW6Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Số cặp cá thể quần thể $n$ (`BATCH_SIZE`): <strong id="val-w6-npop">12</strong></span>
            <input type="range" id="sl-w6-npop" min="2" max="50" step="1" value="12" oninput="updateW6Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Chế độ lấy mẫu (0 = 1 phía $+\varepsilon$, 1 = Đối xứng $\pm\varepsilon$): <strong id="val-w6-mirror">Đối xứng ±ε</strong></span>
            <input type="range" id="sl-w6-mirror" min="0" max="1" step="1" value="1" oninput="updateW6Lab()" />
          </label>
          <div class="sim-btn-row">
            <button class="btn-sim" onclick="stepW6ES()">🚀 Bước nhảy cập nhật θ ← θ + α·ĝ</button>
            <button class="btn-sim-sec" onclick="resetW6ES()">🔄 Đặt lại vị trí θ</button>
          </div>
          <div class="sim-metrics" id="metrics-w6"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w6" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 8. WEEK 7: MODEL-BASED RL & MPC ==================== -->
    <div class="sim-panel" id="panel-mbrl_mpc">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🔮 Model-Based RL: Sai Số Tích Lũy $\mathcal{O}(L_f^H \varepsilon)$ &amp; Làm Trơn Tikhonov</h3>
          <p class="sim-desc">Đường xanh là quỹ đạo thật $s_{t+k}$, đường đứt nét đỏ là quỹ đạo tưởng tượng $\hat{s}_{t+k}$ khi cuộn mô hình $H$ bước. Khi hằng số Lipschitz $L_f &gt; 1$, sai số phình to theo cấp số nhân!</p>
          <label class="sim-ctrl">
            <span>Tầm nhìn quy hoạch $H$ (`HORIZON_LENGTH`): <strong id="val-w7-h">12</strong></span>
            <input type="range" id="sl-w7-h" min="2" max="25" step="1" value="12" oninput="updateW7Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Độ dốc Jacobian mô hình $L_f = \|\nabla_s \hat{f}\|$: <strong id="val-w7-lip">1.14</strong></span>
            <input type="range" id="sl-w7-lip" min="0.85" max="1.30" step="0.01" value="1.14" oninput="updateW7Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Nhiễu đầu vào Tikhonov $\sigma_\xi$ (phạt $\|J_f\|_F^2$): <strong id="val-w7-tik">0.00</strong></span>
            <input type="range" id="sl-w7-tik" min="0.00" max="0.25" step="0.01" value="0.00" oninput="updateW7Lab()" />
          </label>
          <div class="sim-metrics" id="metrics-w7"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w7" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 9. WEEK 8: SAC MAXENT & RLHF/DPO ==================== -->
    <div class="sim-panel" id="panel-sac_dpo">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🔥 SAC Soft-Bellman Temperature $\alpha$ &amp; RLHF/DPO $\beta_{KL}$</h3>
          <p class="sim-desc">Trái: Địa hình $Q(s,a)$ có 2 đỉnh cực đại và phân phối Boltzmann $\pi^*(a|s) \propto \exp(Q(s,a)/\alpha)$. Phải: Đường cong xác suất thắng Bradley-Terry / DPO theo $\beta_{KL}$.</p>
          <label class="sim-ctrl">
            <span>Nhiệt độ Entropy SAC $\alpha$: <strong id="val-w8-alpha">0.25</strong></span>
            <input type="range" id="sl-w8-alpha" min="0.03" max="1.50" step="0.02" value="0.25" oninput="updateW8Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Độ chênh lệch đỉnh phụ $\Delta Q_{\text{sub}}$: <strong id="val-w8-dq">0.85</strong></span>
            <input type="range" id="sl-w8-dq" min="0.20" max="1.00" step="0.05" value="0.85" oninput="updateW8Lab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số phạt KL trong RLHF/DPO $\beta_{KL}$: <strong id="val-w8-beta">0.50</strong></span>
            <input type="range" id="sl-w8-beta" min="0.05" max="2.00" step="0.05" value="0.50" oninput="updateW8Lab()" />
          </label>
          <div class="sim-metrics" id="metrics-w8"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-w8" width="600" height="360"></canvas>
        </div>
      </div>
    </div>

    <!-- ==================== 10. SIM-TO-REAL ROBOTICS PHYSICS ==================== -->
    <div class="sim-panel" id="panel-sim2real_physics">
      <div class="sim-grid">
        <div class="sim-controls">
          <h3>🤖 Sim-to-Real: Tích Phân Euler vs. RK4, Độ Cứng $K_p$ &amp; Ma Sát $\mu$</h3>
          <p class="sim-desc">Kiểm chứng tại sao chọn sai bước thời gian $\Delta t$ lớn kết hợp với bộ điều khiển khớp cứng $K_p$ làm Explicit Euler nổ tung (Numerical Explosion), trong khi RK4 giữ vững ổn định vật lý!</p>
          <label class="sim-ctrl">
            <span>Bước thời gian mô phỏng $\Delta t$ (`max_step_size`): <strong id="val-sim-dt">0.025 s</strong></span>
            <input type="range" id="sl-sim-dt" min="0.005" max="0.065" step="0.002" value="0.025" oninput="updateSim2RealLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Độ cứng lò xo ảo PD $K_p$ (`stiffness`): <strong id="val-sim-kp">65</strong></span>
            <input type="range" id="sl-sim-kp" min="15" max="140" step="5" value="65" oninput="updateSim2RealLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số cản nhớt $K_d$ (`damping`): <strong id="val-sim-kd">8.0</strong></span>
            <input type="range" id="sl-sim-kd" min="1.0" max="25.0" step="0.5" value="8.0" oninput="updateSim2RealLab()" />
          </label>
          <label class="sim-ctrl">
            <span>Hệ số ma sát mặt sàn $\mu$ (giới hạn lực bám $\mu mg$): <strong id="val-sim-mu">0.60</strong></span>
            <input type="range" id="sl-sim-mu" min="0.10" max="1.20" step="0.05" value="0.60" oninput="updateSim2RealLab()" />
          </label>
          <div class="sim-metrics" id="metrics-sim2real"></div>
        </div>
        <div class="sim-canvas-box">
          <canvas id="cv-sim2real" width="600" height="360"></canvas>
        </div>
      </div>
    </div>
  </div>
</section>

<style>
  .sim-lab-wrapper {
    background: var(--bg-secondary);
    border: 2px solid var(--accent-primary);
    border-radius: 14px;
    margin: 1.2rem 0 2.2rem;
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .sim-lab-header {
    padding: 1.15rem 1.35rem 0.75rem;
    background: linear-gradient(180deg, rgba(88, 166, 255, 0.10), transparent);
    border-bottom: 1px solid var(--border-color);
  }
  .sim-lab-title-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .sim-lab-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    background: var(--accent-primary);
    color: #fff;
    margin-bottom: 0.35rem;
  }
  .sim-lab-heading {
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
    border: none !important;
    font-size: 1.28rem !important;
    color: var(--text-primary) !important;
  }
  .sim-lab-subtitle {
    margin: 0;
    font-size: 0.86rem;
    color: var(--text-secondary);
  }
  .sim-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.95rem;
  }
  .sim-tab {
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.38rem 0.68rem;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .sim-tab:hover {
    border-color: var(--accent-primary);
    color: var(--text-primary);
  }
  .sim-tab.active {
    background: var(--accent-primary);
    color: #ffffff;
    border-color: var(--accent-primary);
  }
  .sim-lab-body {
    padding: 1.25rem 1.35rem;
  }
  .sim-panel {
    display: none;
  }
  .sim-panel.active {
    display: block;
  }
  .sim-grid {
    display: grid;
    grid-template-columns: 340px minmax(0, 1fr);
    gap: 1.25rem;
    align-items: center;
  }
  .sim-controls h3 {
    margin: 0 0 0.4rem 0 !important;
    font-size: 1.02rem !important;
    color: var(--accent-primary);
  }
  .sim-desc {
    font-size: 0.83rem;
    color: var(--text-secondary);
    margin: 0 0 0.85rem 0;
    line-height: 1.5;
  }
  .sim-ctrl {
    display: block;
    margin-bottom: 0.65rem;
    font-size: 0.82rem;
  }
  .sim-ctrl span {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.2rem;
  }
  .sim-ctrl input[type="range"] {
    width: 100%;
    accent-color: var(--accent-primary);
    cursor: pointer;
  }
  .sim-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.65rem 0;
  }
  .btn-sim {
    background: var(--accent-primary);
    color: #fff;
    border: none;
    border-radius: 7px;
    padding: 0.42rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-sim:hover {
    opacity: 0.9;
  }
  .btn-sim-sec {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: 7px;
    padding: 0.42rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
  }
  .sim-metrics {
    margin-top: 0.75rem;
    padding: 0.65rem 0.8rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.77rem;
    line-height: 1.55;
  }
  .sim-canvas-box {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .sim-canvas-box canvas {
    width: 100%;
    max-width: 600px;
    height: auto;
    border-radius: 6px;
  }
  @media (max-width: 980px) {
    .sim-grid {
      grid-template-columns: 1fr;
    }
  }
</style>

<script>
(function() {
  function getColors() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return {
      bg: isLight ? '#ffffff' : '#0d1117',
      grid: isLight ? '#e5e7eb' : '#21262d',
      text: isLight ? '#1f2328' : '#e6edf3',
      muted: isLight ? '#57606a' : '#8b949e',
      blue: isLight ? '#0969da' : '#58a6ff',
      green: isLight ? '#1a7f37' : '#3fb950',
      purple: isLight ? '#8250df' : '#d2a8ff',
      orange: isLight ? '#bc4c00' : '#f0883e',
      red: isLight ? '#cf222e' : '#ff7b72'
    };
  }

  window.toggleSimLabCollapse = function() {
    const body = document.getElementById('sim-lab-body');
    body.style.display = (body.style.display === 'none') ? 'block' : 'none';
  };

  window.switchSimTab = function(tabId) {
    document.querySelectorAll('.sim-tab').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.sim-panel').forEach(p => {
      p.classList.toggle('active', p.id === 'panel-' + tabId);
    });
    renderActiveTab(tabId);
  };

  function renderActiveTab(tabId) {
    if (tabId === 'math_eigen') updateMathLab();
    else if (tabId === 'bandit_ucb') updateW1Lab();
    else if (tabId === 'mdp_bellman') updateW2Lab();
    else if (tabId === 'dqn_bias') updateW3Lab();
    else if (tabId === 'pg_variance') updateW4Lab();
    else if (tabId === 'ppo_gae') updateW5Lab();
    else if (tabId === 'es_mirror') updateW6Lab();
    else if (tabId === 'mbrl_mpc') updateW7Lab();
    else if (tabId === 'sac_dpo') updateW8Lab();
    else if (tabId === 'sim2real_physics') updateSim2RealLab();
  }

  // ==================== 1. MATH LAB: 2D CLOUD & EIGEN ====================
  const basePts = [];
  for (let i = 0; i < 120; i++) {
    // Box-Muller
    const u1 = (i + 1) / 121.0;
    const u2 = ((i * 73) % 119 + 1) / 120.0;
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const th = 2.0 * Math.PI * u2;
    basePts.push([r * Math.cos(th), r * Math.sin(th)]);
  }

  window.updateMathLab = function() {
    const s1 = parseFloat(document.getElementById('sl-math-s1').value);
    const s2 = parseFloat(document.getElementById('sl-math-s2').value);
    const angDeg = parseFloat(document.getElementById('sl-math-ang').value);
    const outVal = parseFloat(document.getElementById('sl-math-out').value);
    document.getElementById('val-math-s1').textContent = s1.toFixed(2);
    document.getElementById('val-math-s2').textContent = s2.toFixed(2);
    document.getElementById('val-math-ang').textContent = angDeg + '°';
    document.getElementById('val-math-out').textContent = outVal.toFixed(1);

    const rad = angDeg * Math.PI / 180.0;
    const c = Math.cos(rad), s = Math.sin(rad);
    const l1 = s1 * s1, l2 = s2 * s2;
    const sigXX = l1 * c * c + l2 * s * s;
    const sigYY = l1 * s * s + l2 * c * c;
    const sigXY = (l1 - l2) * c * s;

    // Norms with outlier
    const baseErrs = [0.2, -0.3, 0.4, -0.1, 0.25];
    const errs = baseErrs.concat([outVal]);
    const l1Norm = errs.reduce((a, b) => a + Math.abs(b), 0) / errs.length;
    const l2Norm = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length);
    const lInfNorm = Math.max(...errs.map(Math.abs));

    document.getElementById('metrics-math').innerHTML =
      `Ma trận Hiệp phương sai Σ = [[${sigXX.toFixed(2)}, ${sigXY.toFixed(2)}], [${sigXY.toFixed(2)}, ${sigYY.toFixed(2)}]]<br>` +
      `Trị riêng: <strong>λ₁ = ${l1.toFixed(2)}</strong>, <strong>λ₂ = ${l2.toFixed(2)}</strong> | Số điều kiện κ = ${(Math.max(l1,l2)/Math.min(l1,l2)).toFixed(1)}<br>` +
      `Chuẩn sai số khi Outlier=${outVal.toFixed(1)}: L₁=${l1Norm.toFixed(2)} | L₂=${l2Norm.toFixed(2)} | L∞=${lInfNorm.toFixed(2)}`;

    const cv = document.getElementById('cv-math');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    const cx = 210, cy = 180, scale = 48;
    // Grid
    ctx.strokeStyle = col.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, cy); ctx.lineTo(400, cy);
    ctx.moveTo(cx, 20); ctx.lineTo(cx, 340);
    ctx.stroke();

    // Cloud points
    ctx.fillStyle = col.blue;
    ctx.globalAlpha = 0.65;
    basePts.forEach(([z1, z2]) => {
      const x0 = z1 * s1, y0 = z2 * s2;
      const x = x0 * c - y0 * s;
      const y = x0 * s + y0 * c;
      ctx.beginPath();
      ctx.arc(cx + x * scale, cy - y * scale, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.globalAlpha = 1.0;

    // Eigenvector 1 & 2 arrows
    function drawArrow(x1, y1, x2, y2, color, label) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      const ang = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 9 * Math.cos(ang - 0.4), y2 - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(x2 - 9 * Math.cos(ang + 0.4), y2 - 9 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillText(label, x2 + 6, y2 - 4);
    }
    drawArrow(cx, cy, cx + c * s1 * 1.6 * scale, cy - s * s1 * 1.6 * scale, col.orange, `v₁ (√λ₁=${s1.toFixed(2)})`);
    drawArrow(cx, cy, cx - s * s2 * 1.6 * scale, cy - c * s2 * 1.6 * scale, col.green, `v₂ (√λ₂=${s2.toFixed(2)})`);

    // Right side: Norm comparison bar chart
    const bx = 435, bw = 36, baseB = 315;
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Độ nhạy Chuẩn vs Outlier', 425, 38);
    const norms = [
      { name: 'L₁ (Bền)', val: l1Norm, color: col.green },
      { name: 'L₂ (MSE)', val: l2Norm, color: col.blue },
      { name: 'L∞ (Max)', val: lInfNorm, color: col.red }
    ];
    norms.forEach((n, idx) => {
      const x = bx + idx * 52;
      const h = Math.min(240, n.val * 16);
      ctx.fillStyle = n.color;
      ctx.fillRect(x, baseB - h, bw, h);
      ctx.fillStyle = col.text;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(n.val.toFixed(1), x + 4, baseB - h - 6);
      ctx.fillText(n.name, x - 2, baseB + 16);
    });
  };

  // ==================== 2. WEEK 1: BANDIT & UCB1 ====================
  const trueMeans = [0.2, 0.55, 0.85, 0.40, 0.65];
  let w1State = {
    t: 5,
    N: [1, 1, 1, 1, 1],
    Q: [0.2, 0.5, 0.7, 0.3, 0.6],
    regretUCB: [0],
    regretEps: [0],
    cumUCB: 0,
    cumEps: 0
  };

  window.resetW1Bandit = function() {
    w1State = {
      t: 5,
      N: [1, 1, 1, 1, 1],
      Q: [0.2, 0.5, 0.7, 0.3, 0.6],
      regretUCB: [0],
      regretEps: [0],
      cumUCB: 0,
      cumEps: 0
    };
    updateW1Lab();
  };

  window.stepW1Bandit = function(steps) {
    const c = parseFloat(document.getElementById('sl-w1-c').value);
    const eps = parseFloat(document.getElementById('sl-w1-eps').value);
    const alpha = parseFloat(document.getElementById('sl-w1-alpha').value);
    const bestMean = 0.85;

    for (let s = 0; s < steps; s++) {
      w1State.t += 1;
      // UCB selection
      let bestArm = 0, bestScore = -1e9;
      for (let a = 0; a < 5; a++) {
        const ucb = w1State.Q[a] + c * Math.sqrt(Math.log(w1State.t) / w1State.N[a]);
        if (ucb > bestScore) { bestScore = ucb; bestArm = a; }
      }
      const reward = trueMeans[bestArm] + (Math.random() - 0.5) * 0.4;
      w1State.N[bestArm] += 1;
      w1State.Q[bestArm] += alpha * (reward - w1State.Q[bestArm]);
      w1State.cumUCB += (bestMean - trueMeans[bestArm]);
      w1State.regretUCB.push(w1State.cumUCB);

      // Eps-greedy regret simulation
      let epsArm = 2;
      if (Math.random() < eps) epsArm = Math.floor(Math.random() * 5);
      w1State.cumEps += (bestMean - trueMeans[epsArm]);
      w1State.regretEps.push(w1State.cumEps);
    }
    updateW1Lab();
  };

  window.updateW1Lab = function() {
    const c = parseFloat(document.getElementById('sl-w1-c').value);
    const eps = parseFloat(document.getElementById('sl-w1-eps').value);
    const alpha = parseFloat(document.getElementById('sl-w1-alpha').value);
    document.getElementById('val-w1-c').textContent = c.toFixed(2);
    document.getElementById('val-w1-eps').textContent = eps.toFixed(2);
    document.getElementById('val-w1-alpha').textContent = alpha.toFixed(2);

    let nextArm = 0, maxU = -1e9;
    for (let a = 0; a < 5; a++) {
      const score = w1State.Q[a] + c * Math.sqrt(Math.log(w1State.t) / w1State.N[a]);
      if (score > maxU) { maxU = score; nextArm = a; }
    }

    document.getElementById('metrics-w1').innerHTML =
      `Tổng số bước t = <strong>${w1State.t}</strong> | Cánh tay tối ưu thật: <strong>Arm #3 (q*=0.85)</strong><br>` +
      `Hành động UCB chọn kế tiếp: <strong>Arm #${nextArm + 1}</strong> (Điểm UCB = ${maxU.toFixed(2)})<br>` +
      `Hối tiếc tích lũy (Regret): UCB1 = <strong>${w1State.cumUCB.toFixed(2)}</strong> vs ε-Greedy = <strong>${w1State.cumEps.toFixed(2)}</strong>`;

    const cv = document.getElementById('cv-w1');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Left: 5 Arms Q_t(a) + Bonus U_t(a)
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Ước lượng Q_t(a) (Xanh) + Bán kính UCB (Cam)', 20, 26);

    const baseY = 310, scaleH = 110;
    for (let a = 0; a < 5; a++) {
      const x = 30 + a * 58;
      const qH = Math.max(5, w1State.Q[a] * scaleH);
      const bonus = c * Math.sqrt(Math.log(w1State.t) / w1State.N[a]);
      const bH = Math.min(160, bonus * scaleH);

      ctx.fillStyle = col.blue;
      ctx.fillRect(x, baseY - qH, 38, qH);
      ctx.fillStyle = col.orange;
      ctx.fillRect(x, baseY - qH - bH, 38, bH);

      // True mean marker
      const trueY = baseY - trueMeans[a] * scaleH;
      ctx.strokeStyle = col.green;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x - 4, trueY); ctx.lineTo(x + 42, trueY);
      ctx.stroke();

      ctx.fillStyle = col.text;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(`Arm ${a + 1}`, x + 3, baseY + 16);
      ctx.fillStyle = col.muted;
      ctx.fillText(`N=${w1State.N[a]}`, x + 4, baseY + 30);
    }

    // Right: Regret curves
    const rx = 345, ry = 45, rw = 235, rh = 265;
    ctx.strokeStyle = col.grid;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Hối tiếc tích lũy L_T (Thấp = Tốt)', rx + 10, 26);

    function drawCurve(arr, color) {
      if (arr.length < 2) return;
      const maxVal = Math.max(5, w1State.cumEps, w1State.cumUCB);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      arr.forEach((v, i) => {
        const px = rx + (i / (arr.length - 1)) * rw;
        const py = ry + rh - (v / maxVal) * (rh - 15);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
    drawCurve(w1State.regretEps, col.red);
    drawCurve(w1State.regretUCB, col.green);
    ctx.fillStyle = col.green; ctx.fillText('▬ UCB1 O(ln T)', rx + 12, ry + 22);
    ctx.fillStyle = col.red; ctx.fillText('▬ ε-Greedy O(T)', rx + 12, ry + 40);
  };

  // ==================== 3. WEEK 2: MDP BELLMAN GRID ====================
  const holes = new Set([5, 7, 11, 12]);
  const goalCell = 15;

  window.updateW2Lab = function() {
    const gamma = parseFloat(document.getElementById('sl-w2-gamma').value);
    const slip = parseFloat(document.getElementById('sl-w2-slip').value);
    const iters = parseInt(document.getElementById('sl-w2-iter').value, 10);
    document.getElementById('val-w2-gamma').textContent = gamma.toFixed(2);
    document.getElementById('val-w2-slip').textContent = slip.toFixed(2);
    document.getElementById('val-w2-iter').textContent = iters;

    let V = new Array(16).fill(0);
    let policy = new Array(16).fill(0);
    const dirs = [
      [0, -1], // Left
      [1, 0],  // Down
      [0, 1],  // Right
      [-1, 0]  // Up
    ];
    const arrows = ['←', '↓', '→', '↑'];

    function stepState(s, a) {
      const r = Math.floor(s / 4), c = s % 4;
      const nr = Math.min(3, Math.max(0, r + dirs[a][0]));
      const nc = Math.min(3, Math.max(0, c + dirs[a][1]));
      return nr * 4 + nc;
    }

    for (let k = 0; k < iters; k++) {
      const nextV = V.slice();
      for (let s = 0; s < 16; s++) {
        if (s === goalCell) { nextV[s] = 1.0; continue; }
        if (holes.has(s)) { nextV[s] = -1.0; continue; }
        let bestQ = -1e9, bestA = 0;
        for (let a = 0; a < 4; a++) {
          const pMain = 1.0 - slip;
          const pSide = slip / 2.0;
          const aLeft = (a + 3) % 4;
          const aRight = (a + 1) % 4;
          const transitions = [
            [pMain, stepState(s, a)],
            [pSide, stepState(s, aLeft)],
            [pSide, stepState(s, aRight)]
          ];
          let q = 0;
          transitions.forEach(([prob, sNext]) => {
            const rew = (sNext === goalCell) ? 1.0 : (holes.has(sNext) ? -0.5 : -0.01);
            const vNext = (sNext === goalCell || holes.has(sNext)) ? 0.0 : V[sNext];
            q += prob * (rew + gamma * vNext);
          });
          if (q > bestQ) { bestQ = q; bestA = a; }
        }
        nextV[s] = bestQ;
        policy[s] = bestA;
      }
      V = nextV;
    }

    document.getElementById('metrics-w2').innerHTML =
      `Giá trị tại Ô xuất phát V*(0) = <strong>${V[0].toFixed(3)}</strong> | Hệ số co Banach γ^k = <strong>${Math.pow(gamma, iters).toFixed(4)}</strong><br>` +
      `Hướng đi tối ưu tại Ô 0: <strong>${arrows[policy[0]]}</strong> | Tại Ô 14 (sát đích): <strong>V*(14) = ${V[14].toFixed(3)} (${arrows[policy[14]]})</strong>`;

    const cv = document.getElementById('cv-w2');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    const cellS = 76, ox = 145, oy = 28;
    for (let s = 0; s < 16; s++) {
      const r = Math.floor(s / 4), c = s % 4;
      const x = ox + c * cellS, y = oy + r * cellS;
      if (s === goalCell) {
        ctx.fillStyle = 'rgba(63, 185, 80, 0.35)';
      } else if (holes.has(s)) {
        ctx.fillStyle = 'rgba(255, 123, 114, 0.32)';
      } else {
        const norm = Math.max(0, Math.min(1, (V[s] + 0.3) / 1.3));
        ctx.fillStyle = `rgba(88, 166, 255, ${(0.08 + norm * 0.45).toFixed(2)})`;
      }
      ctx.fillRect(x, y, cellS - 4, cellS - 4);
      ctx.strokeStyle = col.grid;
      ctx.strokeRect(x, y, cellS - 4, cellS - 4);

      ctx.fillStyle = col.text;
      ctx.font = 'bold 12px Inter, sans-serif';
      if (s === goalCell) {
        ctx.fillText('🏆 GOAL', x + 10, y + 32);
        ctx.fillText('+1.00', x + 18, y + 52);
      } else if (holes.has(s)) {
        ctx.fillText('🕳️ HOLE', x + 10, y + 32);
        ctx.fillText('-0.50', x + 18, y + 52);
      } else {
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.fillStyle = col.orange;
        ctx.fillText(arrows[policy[s]], x + 28, y + 34);
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.fillStyle = col.text;
        ctx.fillText(V[s].toFixed(2), x + 18, y + 58);
      }
    }
  };

  // ==================== 4. WEEK 3: DOUBLE DQN BIAS ====================
  window.updateW3Lab = function() {
    const sigma = parseFloat(document.getElementById('sl-w3-sigma').value);
    const nAct = parseInt(document.getElementById('sl-w3-nact').value, 10);
    const gamma = parseFloat(document.getElementById('sl-w3-gamma').value);
    document.getElementById('val-w3-sigma').textContent = sigma.toFixed(2);
    document.getElementById('val-w3-nact').textContent = nAct;
    document.getElementById('val-w3-gamma').textContent = gamma.toFixed(2);

    const trials = 400;
    let sumDQN = 0, sumDDQN = 0;
    const actionSampleQ1 = new Array(nAct).fill(0);
    const actionSampleQ2 = new Array(nAct).fill(0);

    function randn() {
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    for (let t = 0; t < trials; t++) {
      let maxQ1 = -1e9, argMax1 = 0;
      const q1 = [], q2 = [];
      for (let a = 0; a < nAct; a++) {
        const v1 = randn() * sigma;
        const v2 = randn() * sigma;
        q1.push(v1); q2.push(v2);
        if (v1 > maxQ1) { maxQ1 = v1; argMax1 = a; }
        if (t === 0) { actionSampleQ1[a] = v1; actionSampleQ2[a] = v2; }
      }
      sumDQN += maxQ1;
      sumDDQN += q2[argMax1];
    }

    const avgDQN = sumDQN / trials;
    const avgDDQN = sumDDQN / trials;
    const asympDQN = avgDQN / (1.0 - gamma);

    document.getElementById('metrics-w3').innerHTML =
      `Giá trị thật Q*(s,a) = <strong>0.00</strong> | Kỳ vọng DQN thường E[max Q] = <strong style="color:#ff7b72">+${avgDQN.toFixed(3)}</strong> (Bị thổi phồng!)<br>` +
      `Kỳ vọng Double DQN E[Q₂(argmax Q₁)] = <strong style="color:#3fb950">${avgDDQN.toFixed(3)}</strong> (Không chệch ≈ 0)<br>` +
      `Sai số tích lũy qua chuỗi Bellman 1/(1-γ): DQN nổ lên tới <strong>+${asympDQN.toFixed(2)}</strong>`;

    const cv = document.getElementById('cv-w3');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Draw comparison bars & single sample action noises
    const midY = 195;
    ctx.strokeStyle = col.grid;
    ctx.beginPath();
    ctx.moveTo(25, midY); ctx.lineTo(360, midY);
    ctx.stroke();

    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText(`Mẫu nhiễu Q(s, a; θ) trên ${nAct} hành động (1 lượt)`, 25, 28);

    const wBar = Math.max(6, Math.floor(310 / nAct) - 4);
    let maxIdx = 0;
    for (let a = 1; a < nAct; a++) if (actionSampleQ1[a] > actionSampleQ1[maxIdx]) maxIdx = a;

    for (let a = 0; a < nAct; a++) {
      const x = 30 + a * (wBar + 4);
      const h = actionSampleQ1[a] * 38;
      ctx.fillStyle = (a === maxIdx) ? col.red : col.blue;
      if (h >= 0) ctx.fillRect(x, midY - h, wBar, h);
      else ctx.fillRect(x, midY, wBar, -h);
    }

    // Right: Summary bias bars
    ctx.fillStyle = col.text;
    ctx.fillText('Độ chệch trung bình (400 mẫu)', 390, 28);
    const b1H = Math.min(140, avgDQN * 55);
    ctx.fillStyle = col.red;
    ctx.fillRect(415, midY - b1H, 60, b1H);
    ctx.fillStyle = col.text;
    ctx.fillText(`DQN: +${avgDQN.toFixed(2)}`, 405, midY - b1H - 8);

    const b2H = avgDDQN * 55;
    ctx.fillStyle = col.green;
    ctx.fillRect(505, midY - Math.max(3, Math.abs(b2H)), 60, Math.max(4, Math.abs(b2H)));
    ctx.fillStyle = col.text;
    ctx.fillText(`DDQN: ${avgDDQN.toFixed(2)}`, 498, midY - 12);
  };

  // ==================== 5. WEEK 4: POLICY GRADIENT & ENTROPY ====================
  window.updateW4Lab = function() {
    const roff = parseFloat(document.getElementById('sl-w4-roff').value);
    const baseFrac = parseFloat(document.getElementById('sl-w4-base').value);
    const beta = parseFloat(document.getElementById('sl-w4-ent').value);
    document.getElementById('val-w4-roff').textContent = roff.toFixed(0);
    document.getElementById('val-w4-base').textContent = baseFrac.toFixed(2);
    document.getElementById('val-w4-ent').textContent = beta.toFixed(2);

    // Raw returns around roff + [-4, +2, -1, +5]
    const advs = [-4.0, 2.0, -1.5, 5.0];
    const baseline = baseFrac * roff;
    const signals = advs.map(a => (roff + a) - baseline);
    const meanSig = signals.reduce((a, b) => a + b, 0) / signals.length;
    const variance = signals.reduce((a, b) => a + Math.pow(b, 2), 0) / signals.length;

    // Softmax policy with temperature / entropy weight beta
    const logits = [1.8, 0.9, 0.2, -0.4].map(l => l / Math.max(0.08, beta * 3.5));
    const maxL = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxL));
    const sumE = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumE);
    const entropy = -probs.reduce((acc, p) => acc + (p > 1e-9 ? p * Math.log(p) : 0), 0);

    document.getElementById('metrics-w4').innerHTML =
      `Bình phương độ lớn Gradient E[(G_t - b)²]: <strong>${variance.toFixed(1)}</strong> (Khi b=V(s), giảm từ ${(roff*roff + 11.5).toFixed(0)} về 11.6!)<br>` +
      `Shannon Entropy H(π) = <strong>${entropy.toFixed(3)} nats</strong> (Max = ln(4) ≈ 1.386)<br>` +
      `Phân phối π(a|s): [${probs.map(p => (p*100).toFixed(1)+'%').join(', ')}]`;

    const cv = document.getElementById('cv-w4');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Left: Signals G_t - b(s) for 4 sampled actions
    const midY = 190;
    ctx.strokeStyle = col.grid;
    ctx.beginPath(); ctx.moveTo(20, midY); ctx.lineTo(300, midY); ctx.stroke();
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Tín hiệu đẩy Gradient (G_t - b) trên 4 mẫu', 20, 28);

    signals.forEach((sig, i) => {
      const x = 35 + i * 64;
      const h = Math.max(-135, Math.min(135, sig * 2.2));
      ctx.fillStyle = sig >= 0 ? col.green : col.red;
      if (h >= 0) ctx.fillRect(x, midY - h, 42, h);
      else ctx.fillRect(x, midY, 42, -h);
      ctx.fillStyle = col.text;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText((sig >= 0 ? '+' : '') + sig.toFixed(1), x + 4, h >= 0 ? midY - h - 6 : midY - h + 14);
      ctx.fillText(`Mẫu ${i+1}`, x + 4, 340);
    });

    // Right: Policy distribution pi(a|s) under Entropy Bonus
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText(`Phân phối π(a|s) (Entropy H = ${entropy.toFixed(2)})`, 335, 28);
    const baseB = 315;
    probs.forEach((p, i) => {
      const x = 350 + i * 58;
      const h = p * 230;
      ctx.fillStyle = col.purple;
      ctx.fillRect(x, baseB - h, 40, h);
      ctx.fillStyle = col.text;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText((p * 100).toFixed(0) + '%', x + 6, baseB - h - 6);
      ctx.fillText(`a_${i+1}`, x + 12, baseB + 18);
    });
  };

  // ==================== 6. WEEK 5: PPO CLIP & GAE ====================
  window.updateW5Lab = function() {
    const eps = parseFloat(document.getElementById('sl-w5-eps').value);
    const adv = parseFloat(document.getElementById('sl-w5-adv').value);
    const lam = parseFloat(document.getElementById('sl-w5-lam').value);
    const gam = parseFloat(document.getElementById('sl-w5-gam').value);

    document.getElementById('val-w5-eps').textContent = eps.toFixed(2);
    document.getElementById('val-w5-adv').textContent = (adv >= 0 ? '+' : '') + adv.toFixed(2);
    document.getElementById('val-w5-lam').textContent = lam.toFixed(2);
    document.getElementById('val-w5-gam').textContent = gam.toFixed(3);

    const gaeFactor = gam * lam;
    const effHorizon = 1.0 / Math.max(1e-4, 1.0 - gaeFactor);

    document.getElementById('metrics-w5').innerHTML =
      `Vùng tin cậy tỷ số r_t(θ): <strong>[${(1 - eps).toFixed(2)}, ${(1 + eps).toFixed(2)}]</strong> | Khi vượt ngưỡng: <strong>∇_θ L^CLIP = 0</strong><br>` +
      `Hệ số suy giảm GAE (γλ) = <strong>${gaeFactor.toFixed(4)}</strong> | Tầm nhìn hiệu dụng H_adv ≈ <strong>${effHorizon.toFixed(1)} bước</strong>`;

    const cv = document.getElementById('cv-w5');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Left Plot: PPO Clip Objective vs r_t in [0.2, 2.0]
    const lx = 35, ly = 42, lw = 250, lh = 265;
    ctx.strokeStyle = col.grid;
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Đồ thị L^CLIP(r_t) theo tỷ số r_t', lx, 26);

    const midY = ly + lh / 2;
    ctx.beginPath(); ctx.moveTo(lx, midY); ctx.lineTo(lx + lw, midY); ctx.stroke();

    // Vertical lines at 1-eps, 1, 1+eps
    function rToX(r) { return lx + ((r - 0.2) / 1.8) * lw; }
    function valToY(v) { return midY - v * 28; }

    [1 - eps, 1.0, 1 + eps].forEach((rVal, idx) => {
      const px = rToX(rVal);
      ctx.strokeStyle = idx === 1 ? col.muted : col.orange;
      ctx.setLineDash(idx === 1 ? [3, 3] : [4, 4]);
      ctx.beginPath(); ctx.moveTo(px, ly); ctx.lineTo(px, ly + lh); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col.muted;
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(rVal.toFixed(2), px - 12, ly + lh + 14);
    });

    // Unclipped r*A (dashed blue)
    ctx.strokeStyle = col.blue;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let r = 0.2; r <= 2.0; r += 0.03) {
      const px = rToX(r), py = Math.max(ly, Math.min(ly + lh, valToY(r * adv)));
      if (r === 0.2) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Clipped min(r*A, clip(r)*A) (solid green)
    ctx.strokeStyle = col.green;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    for (let r = 0.2; r <= 2.0; r += 0.02) {
      const rClip = Math.max(1 - eps, Math.min(1 + eps, r));
      const obj = Math.min(r * adv, rClip * adv);
      const px = rToX(r), py = Math.max(ly, Math.min(ly + lh, valToY(obj)));
      if (r === 0.2) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Right Plot: GAE weights (gamma*lambda)^l for l=0..24
    const rx = 325, ry = 42, rw = 250, rh = 265;
    ctx.strokeStyle = col.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = col.text;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Trọng số (γλ)^l của sai số TD δ_{t+l}', rx, 26);

    for (let l = 0; l < 20; l++) {
      const w = Math.pow(gaeFactor, l);
      const bx = rx + 8 + l * 12;
      const bh = w * (rh - 25);
      ctx.fillStyle = col.purple;
      ctx.fillRect(bx, ry + rh - bh, 8, bh);
    }
    ctx.fillStyle = col.muted;
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Bước tương lai l = 0 → 20', rx + 55, ry + rh + 16);
  };

  // ==================== 7. WEEK 6: EVOLUTION STRATEGIES ====================
  let esTheta = [-1.4, -1.1];
  window.resetW6ES = function() {
    esTheta = [-1.4, -1.1];
    updateW6Lab();
  };

  window.stepW6ES = function() {
    const gEst = window._lastESGrad || [0, 0];
    esTheta[0] += 0.22 * gEst[0];
    esTheta[1] += 0.22 * gEst[1];
    updateW6Lab();
  };

  window.updateW6Lab = function() {
    const sigma = parseFloat(document.getElementById('sl-w6-sigma').value);
    const nPop = parseInt(document.getElementById('sl-w6-npop').value, 10);
    const mirrored = parseInt(document.getElementById('sl-w6-mirror').value, 10) === 1;
    document.getElementById('val-w6-sigma').textContent = sigma.toFixed(2);
    document.getElementById('val-w6-npop').textContent = nPop;
    document.getElementById('val-w6-mirror').textContent = mirrored ? 'Đối xứng ±ε' : '1 phía +ε';

    // Objective F(x,y) = - (x^2 + 2*y^2) - 0.3*(x^3 + y^2*x) (has curvature so 1-sided has bias/noise)
    function F(x, y) {
      return -(x * x + 1.8 * y * y) + 0.25 * (x * x * x + x * y * y);
    }
    const trueGrad = [
      -2 * esTheta[0] + 0.75 * esTheta[0] * esTheta[0] + 0.25 * esTheta[1] * esTheta[1],
      -3.6 * esTheta[1] + 0.5 * esTheta[0] * esTheta[1]
    ];

    let gx = 0, gy = 0;
    const samples = [];
    for (let i = 0; i < nPop; i++) {
      const ang = (2 * Math.PI * i) / nPop + 0.3;
      const ex = Math.cos(ang) * 1.15, ey = Math.sin(ang) * 0.9;
      const fPos = F(esTheta[0] + sigma * ex, esTheta[1] + sigma * ey);
      samples.push([esTheta[0] + sigma * ex, esTheta[1] + sigma * ey, true]);
      if (mirrored) {
        const fNeg = F(esTheta[0] - sigma * ex, esTheta[1] - sigma * ey);
        samples.push([esTheta[0] - sigma * ex, esTheta[1] - sigma * ey, false]);
        gx += ((fPos - fNeg) / (2 * sigma)) * ex;
        gy += ((fPos - fNeg) / (2 * sigma)) * ey;
      } else {
        // Without baseline, F(theta) term adds huge noise
        gx += (fPos / sigma) * ex;
        gy += (fPos / sigma) * ey;
      }
    }
    gx /= nPop; gy /= nPop;
    window._lastESGrad = [gx, gy];

    const normTrue = Math.hypot(trueGrad[0], trueGrad[1]) + 1e-8;
    const normEst = Math.hypot(gx, gy) + 1e-8;
    const cosSim = (trueGrad[0] * gx + trueGrad[1] * gy) / (normTrue * normEst);

    document.getElementById('metrics-w6').innerHTML =
      `Tâm tham số hiện tại θ = [${esTheta[0].toFixed(2)}, ${esTheta[1].toFixed(2)}] | Đích tối ưu: [0.00, 0.00]<br>` +
      `Độ tương đồng Cosine(∇F thật, ĝ ES) = <strong>${cosSim.toFixed(4)}</strong> (Mirrored ±ε khử sạch số hạng bậc chẵn F(θ) và Hessian!)`;

    const cv = document.getElementById('cv-w6');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    const cx = 300, cy = 180, scale = 95;
    // Draw contour ellipses around (0,0)
    for (let r = 0.35; r <= 2.2; r += 0.35) {
      ctx.strokeStyle = col.grid;
      ctx.beginPath();
      ctx.ellipse(cx, cy, r * scale, (r / Math.sqrt(1.8)) * scale, 0, 0, 2 * Math.PI);
      ctx.stroke();
    }
    // Optimum star
    ctx.fillStyle = col.green;
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.fillText('★ Đích θ*=(0,0)', cx + 6, cy - 6);

    // Samples
    samples.forEach(([sx, sy, isPos]) => {
      ctx.fillStyle = isPos ? col.blue : col.purple;
      ctx.beginPath();
      ctx.arc(cx + sx * scale, cy - sy * scale, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Current theta
    const tx = cx + esTheta[0] * scale, ty = cy - esTheta[1] * scale;
    ctx.fillStyle = col.orange;
    ctx.beginPath(); ctx.arc(tx, ty, 6, 0, 2 * Math.PI); ctx.fill();

    // True gradient vs ES gradient arrows
    function drawVec(vx, vy, color, label) {
      const len = Math.hypot(vx, vy);
      if (len < 1e-4) return;
      const ux = (vx / len) * 55, uy = (vy / len) * 55;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx + ux, ty - uy); ctx.stroke();
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillText(label, tx + ux + 5, ty - uy);
    }
    drawVec(trueGrad[0], trueGrad[1], col.green, '∇F thật');
    drawVec(gx, gy, col.orange, 'ĝ ES');
  };

  // ==================== 8. WEEK 7: MODEL-BASED RL & MPC ====================
  window.updateW7Lab = function() {
    const H = parseInt(document.getElementById('sl-w7-h').value, 10);
    const rawLip = parseFloat(document.getElementById('sl-w7-lip').value);
    const tik = parseFloat(document.getElementById('sl-w7-tik').value);
    document.getElementById('val-w7-h').textContent = H;
    document.getElementById('val-w7-lip').textContent = rawLip.toFixed(2);
    document.getElementById('val-w7-tik').textContent = tik.toFixed(2);

    // Tikhonov input noise regularizes the effective Lipschitz constant
    const effLip = Math.max(0.85, rawLip - tik * 0.9);
    const eps1 = 0.04;
    const trueTraj = [], predTraj = [], errBars = [];

    let errAcc = 0;
    for (let k = 0; k <= H; k++) {
      const sTrue = Math.sin(k * 0.32);
      trueTraj.push(sTrue);
      if (k > 0) errAcc = errAcc * effLip + eps1;
      predTraj.push(sTrue + errAcc * (k % 2 === 0 ? 1 : 0.85));
      errBars.push(errAcc);
    }

    document.getElementById('metrics-w7').innerHTML =
      `Độ dốc hiệu dụng sau làm trơn Tikhonov: <strong>L_eff = ${effLip.toFixed(3)}</strong><br>` +
      `Sai số dự đoán ở bước cuối H=${H}: <strong>||ŝ_{t+H} - s_{t+H}|| = ${errAcc.toFixed(3)}</strong> (Gấp ${(errAcc / eps1).toFixed(1)} lần sai số 1 bước!)`;

    const cv = document.getElementById('cv-w7');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    const ox = 45, oy = 30, w = 525, h = 290, midY = oy + h / 2;
    ctx.strokeStyle = col.grid;
    ctx.strokeRect(ox, oy, w, h);

    function stepToX(k) { return ox + (k / H) * w; }
    function sToY(val) { return midY - val * 65; }

    // Error fan shaded area
    ctx.fillStyle = 'rgba(255, 123, 114, 0.16)';
    ctx.beginPath();
    for (let k = 0; k <= H; k++) {
      const x = stepToX(k), y = sToY(trueTraj[k] + errBars[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let k = H; k >= 0; k--) {
      const x = stepToX(k), y = sToY(trueTraj[k] - errBars[k]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();

    // True trajectory (green)
    ctx.strokeStyle = col.green; ctx.lineWidth = 3;
    ctx.beginPath();
    trueTraj.forEach((v, k) => {
      const x = stepToX(k), y = sToY(v);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Hallucinated rollout (red dashed)
    ctx.strokeStyle = col.red; ctx.lineWidth = 2.5; ctx.setLineDash([5, 4]);
    ctx.beginPath();
    predTraj.forEach((v, k) => {
      const x = stepToX(k), y = Math.max(oy + 5, Math.min(oy + h - 5, sToY(v)));
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = col.green; ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('▬ Quỹ đạo thật s_{t+k}', ox + 12, oy + 22);
    ctx.fillStyle = col.red;
    ctx.fillText('--- Mô hình tưởng tượng ŝ_{t+k} (Vùng đỏ = Bùng nổ sai số O(L^H ε))', ox + 175, oy + 22);
  };

  // ==================== 9. WEEK 8: SAC & DPO ====================
  window.updateW8Lab = function() {
    const alpha = parseFloat(document.getElementById('sl-w8-alpha').value);
    const dq = parseFloat(document.getElementById('sl-w8-dq').value);
    const beta = parseFloat(document.getElementById('sl-w8-beta').value);
    document.getElementById('val-w8-alpha').textContent = alpha.toFixed(2);
    document.getElementById('val-w8-dq').textContent = dq.toFixed(2);
    document.getElementById('val-w8-beta').textContent = beta.toFixed(2);

    const N = 80;
    const actions = [], qVals = [], probs = [];
    let maxScaled = -1e9;
    for (let i = 0; i < N; i++) {
      const a = -1.0 + (2.0 * i) / (N - 1);
      actions.push(a);
      // Bimodal Q landscape: peak 1 at a=-0.45 (height 1.0), peak 2 at a=+0.50 (height dq)
      const q = Math.exp(-Math.pow((a + 0.45) / 0.25, 2)) + dq * Math.exp(-Math.pow((a - 0.50) / 0.25, 2));
      qVals.push(q);
      if (q / alpha > maxScaled) maxScaled = q / alpha;
    }
    let sumExp = 0;
    const rawExps = qVals.map(q => {
      const e = Math.exp(q / alpha - maxScaled);
      sumExp += e;
      return e;
    });
    rawExps.forEach(e => probs.push(e / sumExp));

    document.getElementById('metrics-w8').innerHTML =
      `Khi α = ${alpha.toFixed(2)}: Policy SAC ${alpha < 0.08 ? 'co cụm thành Greedy (chỉ chọn đỉnh 1)' : (alpha > 0.8 ? 'phẳng dần về Uniform' : 'khám phá Đa Mốt (giữ cả 2 đỉnh!)')}<br>` +
      `Trong DPO (β_KL = ${beta.toFixed(2)}): Độ dốc Sigmoid σ(β·Δlog(π/π_ref)) tỉ lệ thuận với β_KL.`;

    const cv = document.getElementById('cv-w8');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Left: Bimodal Q(s,a) and Boltzmann pi*(a|s)
    const lx = 30, ly = 40, lw = 255, lh = 270;
    ctx.strokeStyle = col.grid; ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = col.text; ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Q(s,a) 2 đỉnh (Xanh) & π*(a|s) (Cam)', lx, 26);

    // Q curve
    ctx.strokeStyle = col.blue; ctx.lineWidth = 2.5;
    ctx.beginPath();
    qVals.forEach((q, i) => {
      const px = lx + (i / (N - 1)) * lw;
      const py = ly + lh - (q / 1.3) * (lh - 20);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Policy density curve
    const maxP = Math.max(...probs) + 1e-6;
    ctx.strokeStyle = col.orange; ctx.lineWidth = 3;
    ctx.beginPath();
    probs.forEach((p, i) => {
      const px = lx + (i / (N - 1)) * lw;
      const py = ly + lh - (p / maxP) * (lh - 30);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Right: DPO Sigmoid Loss & Preference Probability
    const rx = 320, ry = 40, rw = 255, rh = 270;
    ctx.strokeStyle = col.grid; ctx.lineWidth = 1; ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = col.text;
    ctx.fillText('DPO: Xác suất P(y_w ≻ y_l) = σ(β·Δ)', rx, 26);

    ctx.strokeStyle = col.purple; ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const delta = -4.0 + (8.0 * i) / 80;
      const prob = 1.0 / (1.0 + Math.exp(-beta * delta));
      const px = rx + (i / 80) * rw;
      const py = ry + lh - prob * (lh - 20) - 10;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };

  // ==================== 10. SIM-TO-REAL PHYSICS LAB ====================
  window.updateSim2RealLab = function() {
    const dt = parseFloat(document.getElementById('sl-sim-dt').value);
    const kp = parseFloat(document.getElementById('sl-sim-kp').value);
    const kd = parseFloat(document.getElementById('sl-sim-kd').value);
    const mu = parseFloat(document.getElementById('sl-sim-mu').value);
    document.getElementById('val-sim-dt').textContent = dt.toFixed(3) + ' s';
    document.getElementById('val-sim-kp').textContent = kp.toFixed(0);
    document.getElementById('val-sim-kd').textContent = kd.toFixed(1);
    document.getElementById('val-sim-mu').textContent = mu.toFixed(2);

    const mass = 1.0, g = 9.81, targetX = 1.0;
    const maxForce = mu * mass * g; // Friction traction limit

    function accel(x, v) {
      const rawF = kp * (targetX - x) - kd * v;
      const clampedF = Math.max(-maxForce, Math.min(maxForce, rawF));
      return clampedF / mass;
    }

    const totalTime = 1.6;
    const steps = Math.max(8, Math.floor(totalTime / dt));
    const eulerTraj = [[0, 0]], rk4Traj = [[0, 0]];
    let xe = 0, ve = 0, xr = 0, vr = 0;
    let eulerExploded = false;

    for (let i = 1; i <= steps; i++) {
      const t = i * dt;
      // Explicit Euler
      const ae = accel(xe, ve);
      xe = xe + dt * ve;
      ve = ve + dt * ae;
      if (Math.abs(xe) > 3.5) eulerExploded = true;
      eulerTraj.push([t, Math.max(-2.5, Math.min(2.8, xe))]);

      // RK4
      const k1x = vr, k1v = accel(xr, vr);
      const k2x = vr + 0.5 * dt * k1v, k2v = accel(xr + 0.5 * dt * k1x, vr + 0.5 * dt * k1v);
      const k3x = vr + 0.5 * dt * k2v, k3v = accel(xr + 0.5 * dt * k2x, vr + 0.5 * dt * k2v);
      const k4x = vr + dt * k3v, k4v = accel(xr + dt * k3x, vr + dt * k3v);
      xr += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x);
      vr += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v);
      rk4Traj.push([t, Math.max(-2.5, Math.min(2.8, xr))]);
    }

    document.getElementById('metrics-sim2real').innerHTML =
      `Giới hạn lực bám ma sát F_max = μ·m·g = <strong>${maxForce.toFixed(2)} N</strong> | Tích số ổn định Δt·K_p = <strong>${(dt * kp).toFixed(2)}</strong><br>` +
      `Trạng thái Explicit Euler: <strong style="color:${eulerExploded ? '#ff7b72' : '#3fb950'}">${eulerExploded ? '💥 MẤT ỔN ĐỊNH SỐ (NỔ QUỸ ĐẠO!)' : '✅ Hội tụ ổn định'}</strong> | RK4: <strong style="color:#3fb950">✅ Luôn ổn định vật lý</strong>`;

    const cv = document.getElementById('cv-sim2real');
    const ctx = cv.getContext('2d');
    const col = getColors();
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, cv.width, cv.height);

    const ox = 45, oy = 32, w = 525, h = 290;
    ctx.strokeStyle = col.grid;
    ctx.strokeRect(ox, oy, w, h);

    function tToX(t) { return ox + (t / totalTime) * w; }
    function xToY(xVal) { return oy + h * 0.68 - xVal * 85; }

    // Target line x = 1.0
    const ty = xToY(1.0);
    ctx.strokeStyle = col.muted; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ox, ty); ctx.lineTo(ox + w, ty); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col.muted; ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Đích x_target = 1.0 m', ox + w - 125, ty - 6);

    // Draw Euler (red)
    ctx.strokeStyle = col.red; ctx.lineWidth = 2.5;
    ctx.beginPath();
    eulerTraj.forEach(([t, xVal], idx) => {
      const px = tToX(t), py = Math.max(oy + 4, Math.min(oy + h - 4, xToY(xVal)));
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Draw RK4 (green)
    ctx.strokeStyle = col.green; ctx.lineWidth = 3;
    ctx.beginPath();
    rk4Traj.forEach(([t, xVal], idx) => {
      const px = tToX(t), py = Math.max(oy + 4, Math.min(oy + h - 4, xToY(xVal)));
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.fillStyle = col.green; ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('▬ Runge-Kutta bậc 4 (RK4 - Chuẩn xác)', ox + 12, oy + 22);
    ctx.fillStyle = col.red;
    ctx.fillText('▬ Explicit Euler (Dễ nổ khi Δt hoặc K_p lớn)', ox + 265, oy + 22);
  };

  document.addEventListener('DOMContentLoaded', () => {
    switchSimTab('__DEFAULT_LAB_TAB__');
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        setTimeout(() => {
          const activeBtn = document.querySelector('.sim-tab.active');
          if (activeBtn) renderActiveTab(activeBtn.getAttribute('data-tab'));
        }, 20);
      });
    }
  });
})();
</script>
"""
