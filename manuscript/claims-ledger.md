# claims-ledger — 全主張とその論拠

対象: `main.tex` / `supplementary.tex` / `cover_letter.tex` / `data_availability.tex` / `README.md`
凍結基準: `FREEZE_MANIFEST.txt`（102ファイル、SHA-256）
試験: 290 passed（`pytest exit=0`）

## 論拠の種別（この区別が本台帳の要点）

| 種別 | 意味 |
|---|---|
| **COMPUTED** | 本パッケージの計算。JSON に保存され、多くは試験で束縛 |
| **MEASURED-LIT** | 他者の**測定値**。一次文献を引用 |
| **DERIVED-LIT** | 測定された定数＋標準理論からの**導出**。測定ではない |
| **ASSUMED** | 仮定。論拠なし、または継続性のために保持 |
| **OPEN** | 未解決と本文が明記しているもの |

**DERIVED-LIT と MEASURED-LIT の混同が、この論文が最初に犯した誤りの型である**
（$T_{1n}\sim1$ s を「測定されている」と書いたが、それは高磁場の測定であり
動作点の値ではなかった）。以下ではこの二つを厳密に分ける。

---

## A. 方法論・推定器

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| A1 | 単一電子リセットは IPC $=0.000$ を与える人工物 | SI S3, cover letter | COMPUTED | `corrected_injection.json:original_e1_reset` | `test_injection_row` |
| A2 | 相関ペア誕生では IPC $=2.941\pm0.108$ | SI S3 | COMPUTED | `corrected_injection.json:corrected_ST_birth` | `test_injection_row` |
| A3 | 分離対では $\langle P_S\rangle$ が入力に依らず $1/4$ に固定される | SI S3 | 解析 | $\langle P_S\rangle=\frac14-\langle\bm S_1\!\cdot\!\bm S_2\rangle$、$\langle\bm S_2\rangle=0$ | — |
| A4 | ソルバーはコヒーレント極限を $10^{-6}$ 以内で再現 | SI S1 | COMPUTED | `test_master_equation.py` | 同左（11件） |
| A5 | 容量は out-of-sample（前半で学習・後半で採点） | SI S2, Methods | 実装 | `reservoir.py:_capacity` | — |
| A6 | null floor $=+0.010$、クリップ無しでは $-1.49$ | SI S2 | COMPUTED | `reanalysis.json:R1` | 未束縛 |
| A7 | Dambre 境界は有限標本で破れる：$1.013\pm0.009$、29/30 | SI S2 | COMPUTED | `s2_dambre_control.json` | `test_dambre_bound_is_violated_in_finite_samples`, `test_dambre_printed_values` |
| A8 | 容量は $L\gtrsim2400$ で収束 | Results, SI S2 | COMPUTED | `final_numbers.json:F1` | `test_length_convergence_row` |
| A9 | 時間格子は $n_t=96$ で収束 | SI S4 | COMPUTED | `grid_convergence.json` | `test_grid_row` |
| A10 | MC の格子誤差は $0.17\%$ | Results | COMPUTED | `c2_mc_grid.json` | `test_clock_effect_exceeds_grid_error` |

**注 A7**: 投稿版は「$1.011$、30/30」と記載していたが、その実行の出力は保存されておらず
**再現できなかった**。再生成値を採用し経緯を SI に明記。**これは本パッケージ唯一の
「以前報告した値が再現しなかった」実例**であり、mtime 来歴の議論に対する反例でもある。

---

## B. 結果1 — 化学で読める

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| B1 | 7つの読み出し経路の IPC（1.02〜4.65） | Results 表, SI S4 | COMPUTED | `readout_routes.json` | `test_readout_route_row`（7行） |
| B2 | 床は経路固有（1.02 / 2.01 / 1.96） | Results 表, Fig.2 | COMPUTED | `m7_route_floors.json`, `m7_hetero_floor.json` | `test_readout_floor_and_excess_column`, `test_hetero_route_floor_column` |
| B3 | CIDNP の excess は $2.64$（床の4.6倍ではない） | Results | COMPUTED（導出） | $4.650-2.005$ | `test_route_specific_floors` |
| B4 | accumulated pool は古典遅延線（$q=1$ で MC $=1.937$） | Results | COMPUTED | `m7_route_floors.json:YS_t_accum.floor_MC` | `test_route_specific_floors` |
| B5 | 読み出しはチャネル数で飽和（5→10 で $+0.003$） | Results | COMPUTED | `readout_routes.json:YS_t` vs `SandT_t` | `test_readout_route_row` |
| B6 | 再結合を切ると CIDNP 点の容量は床（$1.02$） | Results | COMPUTED | `final_numbers.json:F5` | 未束縛 |
| B7 | ゆえに CIDNP は読み出しでなく**書き込み**機構 | Results, SI S4 | 解析＋B6 | $J=0$ で一電子還元状態は最大混合 | — |
| B8 | 生成物レジスタは survivor より高容量 | Results, SI S4 | COMPUTED | `c1_product_carryover.json` | `test_product_register_beats_survivor`, `test_product_register_row` |

---

## C. 結果2 — 量子優位性なし

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| C1 | 工学点：量子 5.59 対 ESN 9.41 / 10.37 | Results 表, SI S6 | COMPUTED | `final_numbers.json:F3` | `test_engineered_classical_row`（4行＋SD） |
| C2 | クリプトクロム点：古典が 8ch で $+49\%$、5ch で $+123\%$ | Results 表 | COMPUTED | `m6_cry_classical.json` | `test_cryptochrome_classical_row`, `test_classical_beats_quantum_on_both_channel_matchings` |
| C3 | コヒーレンス比 $0.816$（全スピン）／$0.817$（電子のみ） | Results, SI S5 | COMPUTED | `final_numbers.json:F2`, `c6_coherence_control.json` | `test_coherence_fraction_values`, `test_coherence_control_row`（6行＋SD） |
| C4 | 半古典参照：古典 $0.764$、その床 $0.768$、**excess $\approx0$** | SI S8 | COMPUTED | `open4_semiclassical{,_floor}.json` | `test_semiclassical_values`, `test_classical_reservoir_has_no_memory_above_its_own_floor` |
| C5 | ゆえに $0.82$ は保守的（半古典比は $0.859$） | Results, SI S8 | COMPUTED | 同上 | `test_semiclassical_values` |

**注 C3**: 「電子のみでも 0.817」は、パネルが「核レジスタを消しただけの同語反復」と
主張したのを**反証**した対照。パネル指摘が計算で覆った2例のうちの1つ。

---

## D. 結果3 — 時計と基準

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| D1 | 5桁のターンオーバー間隔で MC は $1.6\%$ しか変化しない | Abstract, Results | COMPUTED | `c2_clock_paired.json`（12シード対応差、$t=4.23$） | `test_clock_effect_is_resolved_and_is_not_0p2_percent`, `test_paired_clock_generator_is_reproducible` |
| D2 | その変化は格子誤差（$0.17\%$）を上回る | Results | COMPUTED | `c2_mc_grid.json` | `test_clock_effect_exceeds_grid_error` |
| D3 | horizon $=(\mathrm{MC}-C_0)\times T$；$C_0\simeq1$ は記憶を運ばない | Results, Fig.1/3 | COMPUTED | `c3_delay_kernel.json` | `test_delay_kernel_and_horizon_definition` |
| D4 | レジスタは**占有数**を蓄える（純位相減衰では容量不変） | Results, Fig.3(b), SI S7 | COMPUTED | `c4_nuclear_channel.json` | `test_register_stores_populations_not_coherence`, `test_nuclear_channel_row`（6行×2） |
| D5 | **ただし動作点では $T_1=T_2$ 厳密**、ゆえに D4 は数値を変えない | Results, SI S7/S12 | DERIVED-LIT | extreme narrowing で $J(0)=J(\omega)=J(2\omega)$ | `test_T1_equals_T2_at_the_operating_field` |
| D6 | 不利な結果は product register でも成立 | SI S4 表 | COMPUTED | `audit3_{nuclide,desync,anisotropy}_product.json` | `test_adverse_results_survive_the_product_register`, `test_product_audit_table` |

---

## E. 制約（S12）— 論拠種別が最も重要な節

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| E1 | $\tau_c(60$ kDa$)=15.2$ ns | SI S12 | DERIVED-LIT | Stokes–Einstein–Debye、$\bar v=0.73$、水和 1.3 | `test_predicted_T1_row`  **rank-2**（$1/(6D_r)$、NMR が要する側。rank-1 は3倍の 45.7 ns）。球形近似・水和 1.3 は ASSUMED |
| E2 | 検算器の検証：$^2$H/D$_2$O 比 $1.08$、$^{14}$N 比 $0.46$、$^1$H$_2$O 比 $1.54$ | SI S12 表 | COMPUTED vs MEASURED-LIT | `open5_relaxation_estimate.json:validation` | `test_validation_row` |
| E3 | 蛋白質結合プロトン：$50~\mu$T で $2.4$ ms、$9.4$ T で $9.0$ s | SI S12 表 | **DERIVED-LIT** | BPP 双極子＋幾何（1.78 Å） | `test_predicted_T1_row`, `test_register_relaxation_is_a_low_field_problem`  **分子内双極子のみの上界。**浴込みは $1.6$ ms（浴比 $f=0.54$ は本パッケージ自身の水検算行から） |
| E4 | フラビン $^{14}$N：$0.14$–$0.34~\mu$s | SI S12 表 | **DERIVED-LIT** | 四極子式＋**測定 QCC**（Martínez 2025, N5 3.6 / N10 4.8 MHz） | `test_predicted_T1_row` |
| E5 | 光駆動：日光下 $7.7$ s、頭蓋内 $\sim250$ 年 | SI S12, Limitations | **DERIVED-LIT** | 測定吸光係数 $\varepsilon_{450}=11{,}300$＋光子流束仮定＋量子収率 $0.1$ | `test_light_driven_row`, `test_light_cannot_drive_the_cycle_in_a_brain` |
| E6 | 蛋白質結合レジスタの horizon 上限は $6.1$ ms（帯域外） | Results, Limitations, SI S12 | COMPUTED＋DERIVED-LIT | `open5_turnover_estimate.json:feasible_region` | `test_protein_bound_register_cannot_reach_the_neural_band`, `test_feasible_region_row`  同上、浴込みの上限は $4.0$ ms。**マスター方程式に依存**（MC($q$) 曲線経由） |
| E7 | 実行可能なのは $\tau_c\lesssim6$ ns、窓は $\tau_c$ 依存（5 ns で 9.8–26.9 ms） | criterion box, SI S12 | COMPUTED | 同上 | `test_feasible_window_string`  根拠を**格子最終 true 点から二分法へ変更**。Trp H$\beta$ は浴なし $9.37$ ns（$1.63\times$）、浴込み $6.08$ ns（$2.51\times$）。**マスター方程式に依存** |
| E8 | 分子間プロトン浴 $f=0.542$（$\sum_{\rm ext}$ 経由）。浴込みで $T_1=1.58$ ms、天井 $3.99$ ms、境界 $6.08$ ns | SI S12 | **COMPUTED（較正は本パッケージ自身の水検算行）** | `open5_relaxation_estimate.json` の validation 行、`open5_turnover_estimate.json` の `critical_tau_c` | `test_the_intermolecular_bath_is_what_makes_the_requirement_threefold` |
| E9 | 頑健性: 動機づけ可能な全変動で天井は帯域外 | SI S18 | COMPUTED | `open5_turnover_estimate.json` の `robustness` | `test_robustness_row`（11行）, `test_no_motivated_variation_reaches_the_band` |

**この節の要点**: E3–E5 はすべて **DERIVED-LIT**——測定された定数と標準理論からの導出であり、
**測定値そのものではない**。かつ**ラジカル対マスター方程式を必要としない**。
本文はこの帰属を明記している（「the constants that decide whether the horizon reaches
the neural band … would have followed without the master equation」）。

**E5 の弱点**: 光子流束（日光の青色成分 $3\times10^{16}$ cm$^{-2}$s$^{-1}$、
組織透過 $10^{-9}$）と量子収率 0.1 は**ASSUMED**であり、吸光係数のみが MEASURED-LIT。
結論（10桁不足）は仮定の桁が数桁動いても覆らないが、数値そのものは仮定依存である。

---

## F. 核種・アンサンブル・異方性

| # | 主張 | 記載 | 論拠種別 | 論拠 | 試験 |
|---|---|---|---|---|---|
| F1 | プロトンのみのレジスタは excess の $29$–$30\%$ を保持 | Abstract, Results, SI S9 | COMPUTED | `open2_nuclide_register.json`, `audit3_nuclide_product.json` | `test_proton_only_register_survives_but_costs_most_of_the_capacity`, `test_nuclide_row` |
| F2 | kinetics 読み出しは核種制限で死ぬ（excess $+0.000$） | Results, SI S9 | COMPUTED | 同上 | 同上 |
| F3 | ターンオーバー位相ジッタは kinetics を殺す | Limitations, SI S10 | COMPUTED | `open1_ensemble_desync_floor.json`, `audit3_desync_product.json` | `test_desynchronisation_kills_kinetics_but_not_cidnp`, `test_jitter_row_{survivor,product}` |
| F4 | 大ジッタでの raw IPC 上昇はプーリング由来の人工物 | SI S10 | COMPUTED | $q=1$ 対照 | `test_desynchronisation_kills_kinetics_but_not_cidnp` |
| F5 | 不均一性は脅威でない（20% 散布でも 8ch は 4.77） | Limitations, SI S10 | COMPUTED | `open1_ensemble_heterogeneity.json`（16コピー・3シード） | `test_heterogeneity_row`, `test_heterogeneity_is_not_a_threat` |
| F5b | **不均一 $\tau_c$**（16倍散布）でもプール後 excess は $+2.04$（一様プール $+1.96$）。旧 OPEN 項目 I4 を解消 | Limitations (ii), SI S10 | COMPUTED | `open6_tau_c_heterogeneity.json`（6コピー層化分位点・3シード・$T_d=10$ ms） | `test_tau_c_heterogeneity_row`, `test_tau_c_spread_is_a_mean_field_effect` |
| F5c | その上昇は小さい（$\sigma=1$ で $+4.5\%$）が、**平均場効果ではない**。平均 $\bar q$ の低下が説明するのは上昇 $+0.088$ のうち $+0.030$ のみで、残り $+0.059$（**66%**）は散布そのもの。$\sigma=0$ で対照は $8\times10^{-13}$ まで一致 | SI S10 | COMPUTED | 同上（`spread_effect_8ch` 列） | `test_tau_c_spread_is_small_but_not_mean_field`, `test_tau_c_meanfield_is_exact_at_zero_spread` |
| F6 | 異方性 $\eta=0.5$ で容量の $2/3$ を失う | Limitations, SI S11 | COMPUTED | `open3_anisotropy.json`, `audit3_anisotropy_product.json` | `test_anisotropy_row`, `test_anisotropy_costs_capacity_and_vanishes_at_the_ising_point` |
| F7 | $\eta=1$（Ising 点）で excess は厳密ゼロ（survivor $7\times10^{-11}$） | SI S11 | COMPUTED | 同上 | 同上 |
| F8 | $^{14}$N を spin-1 にすると excess は $+28\%$ | Limitations, SI S11 | COMPUTED | `open3_spin1.json` | `test_spin1_row`, `test_spin_half_approximation_for_14N_is_conservative` |

---

## G. 文献に依拠する主張

| # | 主張 | 記載 | 論拠種別 | 一次出典 |
|---|---|---|---|---|
| G1 | 蛋白質結合 $^1$H の $T_1$ は秒スケール（**高磁場**） | Limitations, SI S9 | MEASURED-LIT | Tjandra et al., JACS **117**, 12562 (1995) — 600 MHz |
| G2 | 蛋白質中の $^{14}$N は高速緩和シンク | Limitations, SI S9 | MEASURED-LIT | Sunde & Halle, JMR **203**, 257 (2010); Goddard et al., JMR **199**, 68 (2009) |
| G3 | フラビンセミキノンの $^{14}$N 四極子テンソル | SI S9/S12 | MEASURED-LIT | Martínez et al., Magn. Reson. **6**, 183 (2025) |
| G4 | in vivo $^{31}$P は秒スケール（小分子・高磁場） | Limitations | MEASURED-LIT | Bogner et al., MRM **62**, 574 (2009) |
| G5 | 光サイクル間の核偏極蓄積は in vivo 条件で消える | Discussion | MEASURED-LIT（理論） | Wong, Solov'yov, Hore & Kattnig, JCP **154**, 035102 (2021) |
| G6 | Fisher の長寿命核コヒーレンスは推定であり測定でない | Discussion | MEASURED-LIT（反証） | Player & Hore, JRSI **15**, 20180494 (2018); Korenchan et al., PCCP **23**, 19465 (2021) |
| G7 | Tegmark はイオン位置・tubulin を扱い核スピンを扱わない | Discussion | 文献の**射程**の指摘 | Tegmark, PRE **61**, 4194 (2000) |
| G8 | フラビン蛋白質で photo-CIDNP は観測、クリプトクロムでは未観測 | Discussion | MEASURED-LIT | Thamarath et al., JACS **132**, 15542 (2010) |
| G9 | 神経系 CRY1/CRY2 は主に概日転写制御因子 | Intro, Limitations | MEASURED-LIT | Partch, Green & Takahashi, TCB **24**, 90 (2014) |

**G1 の扱いが本論文で最も注意を要する**: 秒スケールは**測定されている**が
**高磁場での測定**であり、動作点（地磁気）では E3 により $2.4$ ms。
本文・SI・カバーレターの全所でこの限定を付す修正を、セット3〜5で実施した。

---

## H. ASSUMED（仮定であることを明記すべきもの）

| # | 仮定 | 記載 | 状態 |
|---|---|---|---|
| H1 | $T_{1n}=1$ s（旧クロック走査のパラメータ） | SI S7 | 継続性のため保持と明記、S12 で下方修正すると明記 |
| H2 | 光子流束・量子収率（E5） | SI S12 | 数値は仮定、結論は桁に頑健 |
| H3 | 工学点のパラメータ（$J=2$ MHz, $B=1$ mT, $\tau=T_2^e=50$ ns） | Methods | 生物学的前提の**外**であることを明記 |
| H4 | 超微細の大きさ（few–40 MHz 帯） | Methods | 「精密値には依存しない」と明記。ただし異方性には依存する（F6） |
| H5 | リッジ $\lambda=10^{-6}$、特徴非標準化 | SI S2/S12 | 報告値は保守的側と明記 |

---

## I. OPEN（未解決と本文が明記）

| # | 未解決事項 | 記載 | 備考 |
|---|---|---|---|
| I1 | 神経系クリプトクロムのターンオーバー間隔が未測定 | Limitations (iv), Conclusion | in vitro 経路を提案 |
| I2 | クリプトクロム光生成物の核 $T_1$ の直接測定が皆無 | Limitations (iv), SI S12 | いかなる磁場でも存在しない |
| I3 | 異方性は公表テンソルでなく**走査** | Limitations, SI S11 | 制限は縮小、除去せず |
| ~~I4~~ | ~~不均一 $\tau_c$ でのプーリングは未計算~~ | — | **セット7で解消**（F5b/F5c）。行番号は履歴のため残す |
| ~~I5~~ | ~~mtime 来歴（JSON が生成モジュールより古い）~~ | — | **セット9で完全解消**。空の `simulation_results/` から全21コマンドを実行し、**追跡42件のうち41件が byte 一致**（DIFFER 0、未書き込み 0、欠落 0、21/21 exit 0）。除外1件は `open5_feasible_region.json`（現行生成器が出す `feasible` フィールドを欠く superseded スナップショット）。この過程で `cryptochrome_reality.json` が**現行コードで再現しない**ことが判明し（生成器は決定的、出荷ファイルが旧版）、再生成で置換した。セット7の「67/67」はこのファイルを検証していなかった |

---

## J. 本台帳が明らかにした帰属の要点

1. **生き残った基準の数値は、ほぼすべて DERIVED-LIT である。**
   $2.4$ ms（E3）と光子収支（E5）は
   緩和理論・光物理と測定定数からの導出であり、**マスター方程式を要さない**。
   **ただし $6.1$ ms（E6）と $\tau_c\lesssim6$ ns（E7）は要する** —— どちらも
   $q=1-\exp(-T_{\rm d}/T_{1n})$ を、マスター方程式の出力である
   `register_reuse.json` の $\mathrm{MC}(q)$ 曲線に通して得るものである。
   当初この4件すべてを非依存としたのは誤りだった（`new_Q_volition` パネル指摘）。
2. **リザバー計算が固有に寄与しているのは** $\mathrm{MC}-C_0\simeq1.9$（D3）、
   経路固有の床（B2）、書き込み機構としての CIDNP（B7）、および
   「占有数レジスタである」という構造的知見（D4）である。
   ただし D4 は動作点で数値を変えない（D5）。
3. **パネル指摘が計算で覆った例が2件**（C3 の電子のみ対照、D4/D5 の脱位相対照）。
   どちらもパネル自身が挙げた反証条件を実行した結果である。
4. **本パッケージ唯一の再現失敗は A7**（Dambre 1.011→1.013）。原因は元出力の未保存。

---

## K. 台帳自身の限界

- 本台帳は凍結時点（`FREEZE_MANIFEST.txt`）の状態を記述する。
- 「試験」欄が空欄の項目（A5, A6, B6, B7, A3）は**現時点で未束縛**である。
  うち B6（再結合オフで容量が床）は結果1の機構論（B7）の唯一の数値的支柱であり、
  束縛すべき優先度が高い。
- 試験名 `test_kinetic_readout_decays_smoothly_with_jitter` は、
  セット5で判明した floor の跳ね（$1.009\to1.856$）を踏まえると**名称が不正確**。
  凍結中のため改名は次ラウンドに送る。
