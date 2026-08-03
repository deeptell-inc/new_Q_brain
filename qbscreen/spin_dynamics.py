#!/usr/bin/env python3
"""3レイヤー量子脳仮説 — 具体的シミュレーション

MAO-Aの計算パラメータに基づく4つのシミュレーション:
  Sim 1: RPMスピンダイナミクス (一重項収率の磁場依存性)
  Sim 2: 結合e⁻-³¹P系のデコヒーレンス (Lindblad)
  Sim 3: 量子リザーバ計算 (QRC)
  Sim 4: タイムクリスタル核スピンメモリ

依存: NumPy, SciPy, matplotlib
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

OUT_DIR = Path("simulation_results")
OUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 物理定数
# ═══════════════════════════════════════════════════════════════════
HBAR = 1.0545718e-34        # J·s
MU_B = 9.2740100783e-24     # J/T
G_E = 2.002319              # 電子g因子
GAMMA_P = 17.235e6          # ³¹P 磁気回転比 (Hz/T)
GAMMA_H = 42.577e6          # ¹H 磁気回転比 (Hz/T)

# MAO-A パラメータ (本スクリーニング実測値)
A_P31_MHZ = 200.0           # ³¹P HFC (MHz)
A_H1_MHZ = 2.7              # ¹H HFC (MHz)
R_EP_A = 11.6               # e⁻-P 距離 (Å)
SOC_CM = 63.3               # SOC (cm⁻¹)
F_OSC = 0.517               # 振動子強度
T2E_BRAIN_NS = 1.10         # 電子T₂ᵉ @脳内 (ns)
T2N_BRAIN_US = 3249.0       # 核T₂(³¹P) @脳内 (μs)
T1E_US = 100.0              # 電子T₁ᵉ (μs)
T1N_S = 4.6                 # 核T₁(³¹P) (s)


# ═══════════════════════════════════════════════════════════════════
# スピン演算子ユーティリティ
# ═══════════════════════════════════════════════════════════════════

# パウリ行列
SX = np.array([[0, 1], [1, 0]], dtype=complex) / 2
SY = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
SZ = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)
SP = SX + 1j * SY  # 上昇演算子
SM = SX - 1j * SY  # 下降演算子


def spin_op(op, i, n):
    """n-スピン系のi番目スピンに演算子opを適用 (0-indexed)."""
    mats = [I2] * n
    mats[i] = op
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def singlet_projector(i, j, n):
    """スピンi,jの一重項射影演算子 P_S = 1/4 - S_i·S_j."""
    # np.errstate guards a spurious 'divide by zero in matmul' warning emitted by
    # some NumPy 2.x BLAS builds on Apple ARM; the result is numerically exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        SiSj = (spin_op(SX, i, n) @ spin_op(SX, j, n)
                + spin_op(SY, i, n) @ spin_op(SY, j, n)
                + spin_op(SZ, i, n) @ spin_op(SZ, j, n))
    dim = 2 ** n
    return 0.25 * np.eye(dim, dtype=complex) - SiSj


def build_rpm_hamiltonian(B_tesla, n_H=6, J_mhz=0.0):
    """MAO-AのRPスピンハミルトニアンを構築。

    スピン配置: [e1, e2, P1, P2, H1, H2, ..., H_nH]
    n_total = 2 + 2 + n_H

    Parameters
    ----------
    B_tesla : float
        外部磁場 (T)
    n_H : int
        ¹H核スピンの数 (計算コスト: 2^(4+n_H))
    J_mhz : float
        交換相互作用 (MHz), 通常 ~0 for separated RP
    """
    n = 4 + n_H
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    # MHz → rad/s の変換係数 (ℏ=1 の単位系で MHz を使用)
    # H は MHz 単位で構築し、時間は μs 単位にする
    # (1 MHz × 1 μs = 2π → 自然な単位)

    # --- 電子Zeeman ---
    omega_e = G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)  # MHz
    H += omega_e * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))

    # --- 核Zeeman ---
    omega_P = GAMMA_P * B_tesla * 1e-6  # MHz
    omega_H = GAMMA_H * B_tesla * 1e-6  # MHz
    H += omega_P * (spin_op(SZ, 2, n) + spin_op(SZ, 3, n))
    for k in range(n_H):
        H += omega_H * spin_op(SZ, 4 + k, n)

    # np.errstate guards spurious matmul warnings on some NumPy 2.x ARM BLAS
    # builds; the operator products are numerically exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # --- 超微細結合 (isotropic) ---
        # e1 に P1, P2 が結合
        for p_idx in [2, 3]:
            for op in [SX, SY, SZ]:
                H += A_P31_MHZ * spin_op(op, 0, n) @ spin_op(op, p_idx, n)

        # e1 に H1..H_nH が結合 (等分配)
        for h_idx in range(n_H):
            for op in [SX, SY, SZ]:
                H += A_H1_MHZ * spin_op(op, 0, n) @ spin_op(op, 4 + h_idx, n)

        # e2 には弱いHFCのみ (基質ラジカル側, ¹H のみ)
        A_substrate_H = 5.0  # MHz (基質側, やや大きい)
        for h_idx in range(min(3, n_H)):
            for op in [SX, SY, SZ]:
                H += A_substrate_H * spin_op(op, 1, n) @ spin_op(op, 4 + h_idx, n)

        # --- 交換相互作用 ---
        if abs(J_mhz) > 1e-10:
            S1S2 = sum(spin_op(op, 0, n) @ spin_op(op, 1, n) for op in [SX, SY, SZ])
            H += J_mhz * S1S2

    return H


# ═══════════════════════════════════════════════════════════════════
# Sim 1: RPMスピンダイナミクス — 一重項収率の磁場依存性
# ═══════════════════════════════════════════════════════════════════

def sim1_rpm_dynamics(n_H=4, n_B_points=50, t_max_us=1.0, n_t=500,
                      k_recomb_mhz=1.0):
    """一重項収率 Φ_S(B) を計算（対角化ベース高速版）。

    Φ_S = k ∫₀^∞ Tr[P_S ρ(t)] exp(-kt) dt
    対角化 H = V Λ V† によりρ(t)を解析的に計算。
    """
    print("=" * 60)
    print("Sim 1: RPMスピンダイナミクス (MAO-A)")
    print(f"  スピン数: {4+n_H} (2e + 2P + {n_H}H), 次元: {2**(4+n_H)}")
    print("=" * 60)

    n = 4 + n_H
    dim = 2 ** n
    dt = t_max_us / n_t

    # 一重項射影演算子 (電子スピン 0,1)
    P_S = singlet_projector(0, 1, n)

    # 初期状態: 電子一重項 × 核スピン最大混合
    rho0 = P_S / np.trace(P_S)

    B_fields = np.concatenate([
        np.linspace(0, 0.0005, 15),    # 0 - 0.5 mT
        np.linspace(0.0005, 0.005, 25)  # 0.5 - 5 mT
    ])

    singlet_yields = []

    for ib, B in enumerate(B_fields):
        if (ib + 1) % 10 == 0:
            print(f"  磁場 {ib+1}/{len(B_fields)}: B = {B*1e3:.3f} mT")

        H = build_rpm_hamiltonian(B, n_H=n_H)

        # 対角化: H = V diag(E) V†
        E, V = np.linalg.eigh(H)
        Vd = V.conj().T

        # P_S を固有基底で表現
        P_S_eig = Vd @ P_S @ V
        rho0_eig = Vd @ rho0 @ V

        # Φ_S = k Σ_mn |P_S_eig[m,n]|² × |rho0_eig[m,n]|²...
        # 正確には: Φ_S = k ∫ Σ_{m,n,p,q} V†P_S V_{mn} V†ρ0 V_{pq}
        #           × exp(-2πi(E_m-E_n)t) δ_{mp}δ_{nq} exp(-kt) dt
        # 簡略化: Φ_S = k Σ_{m,n} P_S_eig[m,n] × rho0_eig[n,m]
        #           × 1/(k + 2πi(E_m - E_n))
        k_rate = k_recomb_mhz  # MHz

        # ベクトル化: dE[m,n] = E[m] - E[n]
        dE = E[:, None] - E[None, :]  # (dim, dim)
        G = 1.0 / (k_rate + 1j * dE)  # Green関数
        phi_S = k_rate * np.real(np.sum(P_S_eig * rho0_eig.T * G))
        singlet_yields.append(phi_S)

    singlet_yields = np.array(singlet_yields)
    B_mT = B_fields * 1e3

    # --- 磁場効果 (MFE) ---
    phi_zero = singlet_yields[0]
    MFE = (singlet_yields - phi_zero) / phi_zero * 100  # %

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(B_mT, singlet_yields, "b-", lw=2)
    ax1.set_xlabel("Magnetic field B (mT)", fontsize=12)
    ax1.set_ylabel("Singlet yield Φ_S", fontsize=12)
    ax1.set_title("Sim 1: RPM Singlet Yield — MAO-A\n"
                   f"({4+n_H} spins: 2e + 2×³¹P + {n_H}×¹H)", fontsize=12)
    ax1.axhline(phi_zero, color="gray", ls="--", alpha=0.5, label=f"B=0: {phi_zero:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(B_mT, MFE, "r-", lw=2)
    ax2.set_xlabel("Magnetic field B (mT)", fontsize=12)
    ax2.set_ylabel("MFE (%)", fontsize=12)
    ax2.set_title("Magnetic Field Effect on 5-HT Oxidation", fontsize=12)
    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sim1_rpm_dynamics.png", dpi=150)
    plt.close()
    print(f"  Φ_S(B=0) = {phi_zero:.4f}")
    print(f"  Φ_S(B=5mT) = {singlet_yields[-1]:.4f}")
    print(f"  MFE max = {np.max(np.abs(MFE)):.2f}%")

    return {"B_mT": B_mT.tolist(), "singlet_yield": singlet_yields.tolist(),
            "MFE_percent": MFE.tolist(), "phi_S_zero": float(phi_zero)}


# ═══════════════════════════════════════════════════════════════════
# Sim 2: 結合e⁻-³¹P系のデコヒーレンス (Lindblad)
# ═══════════════════════════════════════════════════════════════════

def sim2_decoherence():
    """4準位系 (e⁻-³¹P) のLindblad方程式によるデコヒーレンス。"""
    print("\n" + "=" * 60)
    print("Sim 2: 結合e⁻-³¹P系のデコヒーレンス")
    print("=" * 60)

    dim = 4  # 2-spin system
    n = 2

    # ハミルトニアン: H = ω_S S_z + ω_I I_z + A S·I (@地磁気)
    B = 50e-6  # 50 μT
    omega_S = G_E * MU_B * B / (HBAR * 2 * np.pi * 1e6)  # MHz
    omega_I = GAMMA_P * B * 1e-6  # MHz
    A = A_P31_MHZ  # MHz

    H = (omega_S * spin_op(SZ, 0, n) + omega_I * spin_op(SZ, 1, n)
         + A * sum(spin_op(op, 0, n) @ spin_op(op, 1, n) for op in [SX, SY, SZ]))

    # デコヒーレンス速度 (MHz単位)
    gamma_T2e = 1.0 / (T2E_BRAIN_NS * 1e-3)   # 1/T₂ᵉ in MHz
    gamma_T1e = 1.0 / (T1E_US)                  # 1/T₁ᵉ in MHz
    gamma_T2n = 1.0 / (T2N_BRAIN_US)            # 1/T₂ⁿ in MHz
    gamma_T1n = 1.0 / (T1N_S * 1e6)             # 1/T₁ⁿ in MHz

    # Lindblad演算子
    L_ops = [
        np.sqrt(gamma_T2e) * spin_op(SZ, 0, n),   # 電子脱位相
        np.sqrt(gamma_T1e) * spin_op(SM, 0, n),    # 電子緩和
        np.sqrt(gamma_T2n) * spin_op(SZ, 1, n),    # 核脱位相
        np.sqrt(gamma_T1n) * spin_op(SM, 1, n),    # 核緩和
    ]

    # 初期状態: 電子↑核↑ + 電子↓核↓ の重ね合わせ (最大エンタングルメント)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0 / np.sqrt(2)  # |↑↑⟩
    psi0[3] = 1.0 / np.sqrt(2)  # |↓↓⟩
    rho0 = np.outer(psi0, psi0.conj())

    # Lindblad super-operator
    def lindblad_rhs(t_us, rho_flat):
        rho = rho_flat.reshape(dim, dim)
        drho = -2j * np.pi * (H @ rho - rho @ H)
        for L in L_ops:
            Ld = L.conj().T
            LdL = Ld @ L
            drho += 2 * np.pi * (L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL))
        return drho.flatten()

    # 時間発展 — 3つのタイムスケールを段階的に
    results = {}
    time_ranges = [
        ("electron", np.linspace(0, 0.01, 500)),    # 0-10 ns (電子T₂)
        ("nuclear", np.linspace(0, 10000, 1000)),     # 0-10 ms (核T₂)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (label, t_span) in enumerate(time_ranges):
        sol = solve_ivp(lindblad_rhs, [t_span[0], t_span[-1]],
                        rho0.flatten(), t_eval=t_span, method="RK45",
                        rtol=1e-8, atol=1e-10)

        # コヒーレンス |ρ₀₃| (電子+核エンタングルメント)
        rho_03 = np.array([sol.y[:, k].reshape(dim, dim)[0, 3] for k in range(len(sol.t))])
        # 電子コヒーレンス |ρ₀₁| (電子部分のみ)
        rho_01 = np.array([sol.y[:, k].reshape(dim, dim)[0, 1] for k in range(len(sol.t))])
        # 核コヒーレンス |ρ₀₂|
        rho_02 = np.array([sol.y[:, k].reshape(dim, dim)[0, 2] for k in range(len(sol.t))])
        # 対角 (population)
        pop_00 = np.array([np.real(sol.y[:, k].reshape(dim, dim)[0, 0]) for k in range(len(sol.t))])
        pop_33 = np.array([np.real(sol.y[:, k].reshape(dim, dim)[3, 3]) for k in range(len(sol.t))])

        ax = axes[idx]
        if label == "electron":
            t_ns = sol.t * 1e3  # μs → ns
            ax.plot(t_ns, np.abs(rho_03), "b-", lw=2, label="|ρ₀₃| e⁻-³¹P entanglement")
            ax.plot(t_ns, np.abs(rho_01), "r--", lw=1.5, label="|ρ₀₁| electron coherence")
            ax.plot(t_ns, np.abs(rho_02), "g--", lw=1.5, label="|ρ₀₂| nuclear coherence")
            ax.set_xlabel("Time (ns)")
            ax.set_title("Electron decoherence\n(T₂ᵉ ≈ 1 ns)", fontsize=11)
            ax.axvline(T2E_BRAIN_NS, color="gray", ls=":", alpha=0.5, label=f"T₂ᵉ={T2E_BRAIN_NS}ns")
        else:
            t_ms = sol.t * 1e-3  # μs → ms
            ax.plot(t_ms, np.abs(rho_03), "b-", lw=2, label="|ρ₀₃| entanglement")
            ax.plot(t_ms, np.abs(rho_02), "g-", lw=2, label="|ρ₀₂| nuclear coherence")
            ax.plot(t_ms, pop_00, "k--", lw=1, alpha=0.5, label="P(|↑↑⟩)")
            ax.plot(t_ms, pop_33, "k:", lw=1, alpha=0.5, label="P(|↓↓⟩)")
            ax.set_xlabel("Time (ms)")
            ax.set_title(f"Nuclear decoherence\n(T₂(³¹P) ≈ {T2N_BRAIN_US/1e3:.1f} ms)",
                         fontsize=11)
            ax.axvline(T2N_BRAIN_US / 1e3, color="gray", ls=":", alpha=0.5,
                       label=f"T₂(³¹P)={T2N_BRAIN_US/1e3:.1f}ms")

        ax.set_ylabel("Coherence / Population")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sim 2: Decoherence of Coupled e⁻-³¹P System (MAO-A, B=50μT)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sim2_decoherence.png", dpi=150)
    plt.close()
    print(f"  電子T₂ᵉ = {T2E_BRAIN_NS} ns")
    print(f"  核T₂(³¹P) = {T2N_BRAIN_US} μs = {T2N_BRAIN_US/1e3:.1f} ms")
    print(f"  コヒーレンス寿命比: 核/電子 = {T2N_BRAIN_US*1e3/T2E_BRAIN_NS:.0f}倍")

    return {"T2e_ns": T2E_BRAIN_NS, "T2n_us": T2N_BRAIN_US,
            "ratio": T2N_BRAIN_US * 1e3 / T2E_BRAIN_NS}


# ═══════════════════════════════════════════════════════════════════
# Sim 3: 量子リザーバ計算 (QRC)
# ═══════════════════════════════════════════════════════════════════

def _build_input_kick_cache(input_spins, n, n_levels=2):
    """入力キック演算子のキャッシュを構築。u ∈ {-1, +1} の2値のみ。"""
    cache = {}
    for u in [-1.0, 1.0]:
        # テンソル積で直接構築 (各スピンの回転を逐次kron)
        mats = []
        for k in range(n):
            if k in input_spins:
                coupling = 0.8 if k < 2 else 0.3
                theta = u * np.pi / 2 * coupling
                c, s = np.cos(theta), -1j * np.sin(theta)
                mats.append(np.array([[c, s], [s, c]], dtype=complex))
            else:
                mats.append(I2)
        U = mats[0]
        for m in mats[1:]:
            U = np.kron(U, m)
        cache[u] = U
    return cache


def _run_qrc_single(n_spins, n_H, n_samples=300, n_steps=40, dt=0.02):
    """単一リザーバサイズでQRCを実行。

    方式: 状態ベクトル + 事前対角化 + 入力キック (高速版)
    密度行列ρではなく状態ベクトル|ψ⟩で計算: O(N) vs O(N²)
    """
    n = 2 + 2 + n_H
    dim = 2 ** n

    # --- 固定ハミルトニアン事前対角化 ---
    B0 = 50e-6
    H0 = build_rpm_hamiltonian(B0, n_H=n_H)
    E, V = np.linalg.eigh(H0)
    Vd = V.conj().T
    phase_vec = np.exp(-2j * np.pi * E * dt)
    U_free = V @ np.diag(phase_vec) @ Vd

    P_S = singlet_projector(0, 1, n)

    # 観測演算子
    obs_ops = [spin_op(SZ, k, n) for k in range(n)]
    obs_ops.append(P_S)

    # 入力方式: HFC変調 — 生物学的に正しいメカニズム
    # 基質結合によりHFCが変調 → 異なるスピンダイナミクス
    # 各クラスで異なるHFC摂動ハミルトニアンを事前計算

    # 摂動ハミルトニアン: δH = δA × (S₁·I_P1 + S₁·I_P2 + S₂·I_H...)
    # 全HFCチャネルを入力で変調 — リザーバ全体に情報伝搬
    dA = 80.0  # MHz (基質結合による変化 ~40% of A_P31)
    H_perturb = np.zeros((dim, dim), dtype=complex)
    # 電子1 ↔ 全核スピン
    for nuc_idx in range(2, n):
        coupling_weight = 1.0 if nuc_idx < 4 else 0.5  # P:強, H:弱
        for op in [SX, SY, SZ]:
            H_perturb += coupling_weight * spin_op(op, 0, n) @ spin_op(op, nuc_idx, n)
    # 電子2 ↔ 部分核スピン (非対称性が情報を生む)
    for nuc_idx in range(3, min(n, 6)):
        for op in [SX, SY, SZ]:
            H_perturb += 0.3 * spin_op(op, 1, n) @ spin_op(op, nuc_idx, n)
    U_pert_plus = expm(-2j * np.pi * dA * H_perturb * dt)
    U_pert_minus = expm(+2j * np.pi * dA * H_perturb * dt)
    pert_cache = {1.0: U_pert_plus, -1.0: U_pert_minus}

    # 初期状態: 一重項の固有ベクトル
    evals, evecs = np.linalg.eigh(P_S)
    psi0 = evecs[:, -1]

    np.random.seed(42)
    all_features = []
    all_labels = []

    for _ in range(n_samples):
        u_seq = np.random.choice([-1.0, 1.0], size=n_steps)
        psi = psi0.copy()

        obs_trajectory = []
        for step in range(n_steps):
            # 1. 入力: HFC摂動キック (基質結合の模擬)
            psi = pert_cache[u_seq[step]] @ psi
            # 2. 自由発展
            psi = U_free @ psi
            # 3. 観測
            obs = [np.real(psi.conj() @ (o @ psi)) for o in obs_ops]
            obs_trajectory.append(obs)

        n_read = min(20, n_steps)
        linear_feat = np.array(obs_trajectory[-n_read:]).flatten()
        # 2次特徴 (最後5ステップの観測量の積)
        last5 = np.array(obs_trajectory[-5:]).flatten()
        quad_feat = []
        for i in range(0, len(last5), 2):
            for j in range(i, min(i+4, len(last5))):
                quad_feat.append(last5[i] * last5[j])
        feat = np.concatenate([linear_feat, np.array(quad_feat)])
        all_features.append(feat)

        # ターゲット: 最後3入力のパリティ
        parity = int(np.prod(u_seq[-3:]) > 0)
        all_labels.append(parity)

    X = np.array(all_features)
    y = np.array(all_labels)

    # Train/test分割
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Ridge回帰
    alpha = 0.1
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    W = np.linalg.solve(XtX, X_train.T @ y_train)

    y_pred_test = (X_test @ W > 0.5).astype(int)
    acc_test = np.mean(y_pred_test == y_test)

    return acc_test, X.shape[1], X, y


def sim3_quantum_reservoir():
    """スケーラブルQRC: リザーバサイズを4→10スピンまで拡張して精度を比較。"""
    print("\n" + "=" * 60)
    print("Sim 3: 量子リザーバ計算 (QRC) — スケーリング解析")
    print("=" * 60)

    # --- サイズを変えて精度を比較 ---
    configs = [
        (4,  0, "4 spins\n(2e+2P)"),
        (6,  2, "6 spins\n(2e+2P+2H)"),
        (8,  4, "8 spins\n(2e+2P+4H)"),
        (10, 6, "10 spins\n(2e+2P+6H)"),
    ]

    results = {}
    accs = []
    labels_list = []
    dims = []

    for n_total, n_H, label in configs:
        dim = 2 ** n_total
        print(f"\n  {label.replace(chr(10),' ')}: dim={dim}")
        t0 = time.time()
        acc, n_feat, X_last, y_last = _run_qrc_single(
            n_total, n_H, n_samples=300, n_steps=40, dt=0.02
        )
        elapsed = time.time() - t0
        print(f"    精度: {acc*100:.1f}%  特徴数: {n_feat}  時間: {elapsed:.1f}s")
        accs.append(acc * 100)
        labels_list.append(label)
        dims.append(dim)
        results[n_total] = {"accuracy": float(acc), "dim": dim,
                            "n_features": n_feat, "time_s": elapsed}

    # --- ランダムベースライン ---
    n_feat_last = results[max(results.keys())]["n_features"]
    np.random.seed(99)
    X_random = np.random.randn(300, n_feat_last)
    n_train = 210
    alpha = 0.1
    XtX = X_random[:n_train].T @ X_random[:n_train] + alpha * np.eye(n_feat_last)
    W_r = np.linalg.solve(XtX, X_random[:n_train].T @ y_last[:n_train])
    acc_random = np.mean((X_random[n_train:] @ W_r > 0.5).astype(int) == y_last[n_train:])
    print(f"\n  ランダムベースライン: {acc_random*100:.1f}%")

    # --- プロット ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 精度 vs リザーバサイズ
    ax = axes[0]
    colors_bar = ["#a8d8ea", "#7ec8e3", "#3a86ff", "#023e8a"]
    bars = ax.bar(range(len(accs)), accs, color=colors_bar, edgecolor="black")
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("(a) QRC Accuracy vs Reservoir Size\nTemporal Parity Task", fontsize=11)
    ax.axhline(50, color="red", ls="--", alpha=0.4, label="Chance")
    ax.axhline(acc_random * 100, color="gray", ls=":", alpha=0.5, label="Random")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) 精度 vs Hilbert空間次元 (対数)
    ax = axes[1]
    ax.semilogx(dims, accs, "ko-", lw=2, ms=8)
    ax.axhline(50, color="red", ls="--", alpha=0.4)
    ax.set_xlabel("Hilbert space dimension", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("(b) Scaling: Accuracy vs dim(H)", fontsize=11)
    ax.grid(True, alpha=0.3)
    for d, a in zip(dims, accs):
        ax.annotate(f"{a:.1f}%", (d, a), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    # (c) PCA可視化 (最大リザーバ)
    ax = axes[2]
    from numpy.linalg import svd
    X_centered = X_last - X_last.mean(axis=0)
    U_svd, s, Vt = svd(X_centered, full_matrices=False)
    proj = U_svd[:, :2] * s[:2]
    for c, lbl, col in [(0, "Even parity", "blue"), (1, "Odd parity", "red")]:
        mask = y_last == c
        ax.scatter(proj[mask, 0], proj[mask, 1], c=col, alpha=0.4, s=20, label=lbl)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(f"(c) Feature Space ({max(dims)} dim reservoir)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sim3_qrc.png", dpi=150)
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════
# Sim 4: タイムクリスタル核スピンメモリ
# ═══════════════════════════════════════════════════════════════════

def sim4_time_crystal():
    """Floquet駆動下の³¹P核スピンのタイムクリスタル挙動。

    QTCC (Wakaura & Suksmono 2025) のEq.(1)に基づく:
    H(t) = { Σ 0.5(1-d)X_j   (0 ≤ t mod 2T ≤ T)
           { H_1              (T < t mod 2T ≤ 2T)

    H_1 = Ising-like interaction (³¹P-³¹P dipolar coupling)
    """
    print("\n" + "=" * 60)
    print("Sim 4: タイムクリスタル核スピンメモリ")
    print("=" * 60)

    n = 4  # 4個の³¹P核スピン
    dim = 2 ** n

    # --- Ising型相互作用 (H₁) ---
    # H₁ = Σ J_ij Z_i Z_j + h_i Z_i  (³¹P-³¹P双極子結合)
    J_coupling = 0.01  # MHz (³¹P-³¹P双極子, ~10 Å距離)
    h_disorder = 0.005  # MHz (不均一性/disorder → MBL)

    np.random.seed(123)
    h_fields = h_disorder * np.random.randn(n)

    H1 = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        H1 += h_fields[i] * spin_op(SZ, i, n)
        for j in range(i + 1, n):
            H1 += J_coupling * spin_op(SZ, i, n) @ spin_op(SZ, j, n)

    # --- Floquetキック (X回転) ---
    def build_kick(d_noise):
        """H_kick = Σ 0.5(1-d) X_j"""
        H_kick = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            H_kick += 0.5 * (1 - d_noise) * 2 * spin_op(SX, i, n)  # X = 2*SX
        return H_kick

    # --- 初期状態: |++++⟩ (X方向偏極 — TC振動が明瞭) ---
    # |+⟩ = (|↑⟩ + |↓⟩)/√2 なので |++++⟩ = 全状態の均等重ね合わせ
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # --- ノイズレベルを変えてTC挙動を比較 ---
    T_period = 1.0  # μs (Floquet周期)
    n_periods = 100
    d_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    coherence_lifetimes = {}

    for idx, d in enumerate(d_values):
        H_kick = build_kick(d)
        # Floquet演算子: U_F = exp(-iH₁T) × exp(-iH_kick T)
        U_kick = expm(-2j * np.pi * H_kick * T_period)
        U_int = expm(-2j * np.pi * H1 * T_period)
        U_F = U_int @ U_kick  # 1周期のFloquet演算子

        psi = psi0.copy()
        magnetizations = []

        for period in range(n_periods):
            # 磁化 ⟨M_z⟩ = Σ ⟨σ_z^i⟩
            Mz = sum(np.real(psi.conj() @ spin_op(SZ, i, n) @ psi) for i in range(n))
            magnetizations.append(Mz / n)  # 正規化
            psi = U_F @ psi

        magnetizations = np.array(magnetizations)
        stroboscopic_t = np.arange(n_periods) * 2 * T_period  # 2T周期

        # TC検出: 偶数/奇数周期の磁化の差
        even_mag = magnetizations[::2]
        odd_mag = magnetizations[1::2]

        # DTC order parameter: |⟨M_z(even)⟩ - ⟨M_z(odd)⟩| / 2
        if len(even_mag) > 10 and len(odd_mag) > 10:
            dtc_order = np.abs(np.mean(even_mag[-20:]) - np.mean(odd_mag[-20:])) / 2
        else:
            dtc_order = 0.0

        # コヒーレンス寿命: 磁化の振幅が1/eに減衰する時間
        amplitudes = np.abs(magnetizations - np.mean(magnetizations))
        if amplitudes[0] > 0.01:
            decay_idx = np.argmax(amplitudes < amplitudes[0] / np.e)
            if decay_idx > 0:
                coh_lifetime = decay_idx * 2 * T_period  # μs
            else:
                coh_lifetime = n_periods * 2 * T_period  # 減衰しない
        else:
            coh_lifetime = 0.0

        coherence_lifetimes[d] = coh_lifetime

        ax = axes[idx]
        ax.plot(stroboscopic_t, magnetizations, "b-", lw=0.8, alpha=0.7)
        ax.plot(stroboscopic_t[::2], magnetizations[::2], "ro", ms=2, alpha=0.5, label="even")
        ax.plot(stroboscopic_t[1::2], magnetizations[1::2], "bs", ms=2, alpha=0.5, label="odd")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("⟨Mz⟩/N")
        ax.set_title(f"d = {d} (noise)\nDTC order = {dtc_order:.3f}, τ = {coh_lifetime:.0f} μs",
                     fontsize=10)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Sim 4: Time Crystal ³¹P Nuclear Spin Memory\n"
                 "(4 spins, Floquet-driven Ising model, T=1μs)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sim4_time_crystal.png", dpi=150)
    plt.close()

    # --- コヒーレンス寿命 vs ノイズのプロット ---
    fig2, ax = plt.subplots(figsize=(8, 5))
    d_vals = list(coherence_lifetimes.keys())
    lifetimes = list(coherence_lifetimes.values())
    ax.semilogy([d if d > 0 else 1e-4 for d in d_vals], lifetimes, "ko-", lw=2, ms=8)
    ax.set_xlabel("Noise parameter d", fontsize=12)
    ax.set_ylabel("Coherence lifetime (μs)", fontsize=12)
    ax.set_title("TC Protection: Coherence Lifetime vs Noise", fontsize=12)
    ax.axhline(T2N_BRAIN_US, color="red", ls="--", alpha=0.5,
               label=f"T₂(³¹P) no TC = {T2N_BRAIN_US:.0f} μs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sim4_tc_lifetime.png", dpi=150)
    plt.close()

    for d, lt in coherence_lifetimes.items():
        print(f"  d = {d}: τ_coherence = {lt:.0f} μs")

    return {"coherence_lifetimes": {str(k): v for k, v in coherence_lifetimes.items()}}


# ═══════════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("3レイヤー量子脳仮説 — シミュレーション")
    print("MAO-Aパラメータに基づく具体的計算")
    print("=" * 60)

    all_results = {}
    t0 = time.time()

    # Sim 1: RPMスピンダイナミクス (n_H=4 → 256次元, 対角化で高速)
    r1 = sim1_rpm_dynamics(n_H=4, n_B_points=30)
    all_results["sim1_rpm"] = r1

    # Sim 2: デコヒーレンス
    r2 = sim2_decoherence()
    all_results["sim2_decoherence"] = r2

    # Sim 3: QRC
    r3 = sim3_quantum_reservoir()
    all_results["sim3_qrc"] = r3

    # Sim 4: タイムクリスタル
    r4 = sim4_time_crystal()
    all_results["sim4_time_crystal"] = r4

    elapsed = time.time() - t0
    print(f"\n全シミュレーション完了: {elapsed:.1f} 秒")

    # JSON保存
    with open(OUT_DIR / "simulation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"結果: {OUT_DIR}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--qrc-only":
        print("=" * 60)
        print("QRC スケーリング解析のみ実行")
        print("=" * 60)
        OUT_DIR.mkdir(exist_ok=True)
        t0 = time.time()
        r3 = sim3_quantum_reservoir()
        elapsed = time.time() - t0
        print(f"\n完了: {elapsed:.1f} 秒")
        with open(OUT_DIR / "sim3_qrc_results.json", "w") as f:
            json.dump(r3, f, indent=2, default=str)
    else:
        main()
