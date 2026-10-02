# Graph Report - multiscale-grn-inference  (2026-10-02)

## Corpus Check
- 130 files · ~436,788 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 21 file(s) not represented in the graph (top: .csv 12, (none) 4, .npz 4)

## Summary
- 905 nodes · 2133 edges · 41 communities (29 shown, 12 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.89)
- Token cost: 60,260 input · 0 output

## Community Hubs (Navigation)
- Non-stationary Data Generation
- Full Dynamics Comparison
- Edge Recovery Metrics
- FP Inference Pipeline
- Diagnostic Findings (Docs)
- OU/FP/Consistency Losses
- Bayesian Non-stationary GRN
- JKOnet* Baseline
- Bayesian Destructive GRN
- Archived Loss Diagnostics
- Algorithm 1 Main/Optimize
- Test Ground Truth Helpers
- Wasserstein Comparison
- Run Directory Housekeeping
- OU Loss Checks
- FP Cell Population (JKO)
- Knockout Loss Ablation
- ComputeLoss Implementation
- ComputeLoss Tests
- GRN Graph Plotting
- AnnData & Dataset Plots
- Exact OU Transition
- Intervention Fixtures
- OU Expression Tests
- Single-gene Knockout Sim
- CMA-ES Comparison
- Knockout Array Runner
- Loss-at-Truth Check
- Dataset Report
- Loss Combination Comparison
- Loss Landscape Plot
- Stationary Simulator
- JKO Convergence Tests
- BIP Initialize
- CI Workflow
- JKOnet* Setup
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `Theta` - 44 edges
2. `preprocessing()` - 31 edges
3. `NetworkSimulatorNonStationaryMu` - 23 edges
4. `ou_gene_expression()` - 21 edges
5. `extract_snapshots()` - 18 edges
6. `plot_embeddings()` - 17 edges
7. `_sliced_w2()` - 16 edges
8. `make_single_gene_knockout_sim()` - 15 edges
9. `fit_combination()` - 15 edges
10. `fit_combination_cma()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `get_true_params()` --uses--> `NetworkSimulatorPerGeneMu`  [INFERRED]
  deprecated/destructive_ou_loss_check.py → src/multsc_grn_inference/datagen/run_destructive_measurements.py
- `generate_and_save()` --uses--> `NetworkSimulatorNonStationaryMu`  [INFERRED]
  deprecated/loss_diagnostic.py → src/multsc_grn_inference/datagen/non_stationary_sim.py
- `draw_cross_section()` --uses--> `NetworkSimulatorNonStationaryMu`  [INFERRED]
  tests/algorithm/_ground_truth.py → src/multsc_grn_inference/datagen/non_stationary_sim.py
- `make_snapshots()` --uses--> `NetworkSimulatorNonStationaryMu`  [INFERRED]
  tests/algorithm/_ground_truth.py → src/multsc_grn_inference/datagen/non_stationary_sim.py
- `make_stationary_sim()` --uses--> `NetworkSimulatorNonStationaryMu`  [INFERRED]
  tests/algorithm/_ground_truth.py → src/multsc_grn_inference/datagen/non_stationary_sim.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Baselines testing value of known mu_k against Algorithm 1** — tests_diagnostics_jko_testing_readme_algorithm_1, tests_diagnostics_jko_testing_readme_known_intervention_means, tests_diagnostics_jko_testing_readme_jkonet_star, tests_diagnostics_sch_bridge_test_readme_ot_conditional_flow_matching, tests_diagnostics_jko_testing_readme_sliced_w2_shared_metric [EXTRACTED 1.00]
- **Optimizer fixes chain: simplex fix -> CMA-ES -> moment preconditioner** — tests_diagnostics_cma_es_test_readme_nelder_mead_zero_simplex_bug, tests_diagnostics_cma_es_test_readme_custom_initial_simplex, tests_diagnostics_cma_es_test_readme_cma_es, tests_diagnostics_moment_preconditioner_test_readme_covariance_trajectory_preconditioner [INFERRED 0.85]
- **Non-interventional run series** — tests_diagnostics_non_interventional_runs_21_09_6_14_34_8_readme_doc, tests_diagnostics_non_interventional_runs_21_09_6_14_59_8_readme_doc, tests_diagnostics_non_interventional_runs_21_09_6_16_40_8_readme_doc, tests_diagnostics_non_interventional_runs_22_09_6_11_28_8_readme_doc, tests_diagnostics_non_interventional_runs_21_09_6_14_34_8_readme_stationary_ground_truth [EXTRACTED 1.00]

## Communities (41 total, 12 thin omitted)

### Community 0 - "Non-stationary Data Generation"
Cohesion: 0.06
Nodes (40): main(), mu_tanh(), NetworkSimulatorPerGeneMu, plot_snapshot_counts(), _save_timepoints(), _sim_to_dataframes_destructive(), fit_and_sample(), main() (+32 more)

### Community 1 - "Full Dynamics Comparison"
Cohesion: 0.06
Nodes (31): main(), _plot(), full_fp_trajectory_w2(), full_ou_trajectory_w2(), spectral_recovery(), _edge_metrics(), main(), _plot() (+23 more)

### Community 2 - "Edge Recovery Metrics"
Cohesion: 0.07
Nodes (23): auprc(), auroc(), edge_labels(), precision_at_k(), recall_at_k(), score(), _edge_metrics(), fit_one() (+15 more)

### Community 3 - "FP Inference Pipeline"
Cohesion: 0.07
Nodes (22): build_loss_fn(), loss(), compute_per_timepoint_w2(), load_observations(), load_true_grns(), main(), make_simulate_fn(), simulate() (+14 more)

### Community 4 - "Diagnostic Findings (Docs)"
Cohesion: 0.08
Nodes (33): compare_cma_es.py script, Nelder-Mead vs CMA-ES diagnostic, Exact OU transition vs Euler-Maruyama, _exact_combinations.py (loss_ou/loss_fp/loss_cons with exact transition), exact_ou.py (exact_ou_transition, exact_fp_cell_population), loss-identifiability diagnostic (check_loss_at_truth n_substeps sweep), Exact OU transition via Van Loan (1978) block-matrix exponential, Algorithm 1 (OU/FP/Cons GRN inference with known mu_k) (+25 more)

### Community 5 - "OU/FP/Consistency Losses"
Cohesion: 0.09
Nodes (19): _cons_step(), consistency_loss(), fp_loss(), _fp_step(), _kde_sample(), _ou_euler_maruyama(), ou_fp_loss(), ou_loss() (+11 more)

### Community 6 - "Bayesian Non-stationary GRN"
Cohesion: 0.08
Nodes (17): build_model(), build_transition_batches(), main(), plot_data_overview(), plot_posterior_traces(), plot_recovery(), NetworkSimulatorNonStationaryMu, test_theta0_closer_to_true_params_than_a_naive_generic_start() (+9 more)

### Community 7 - "JKOnet* Baseline"
Cohesion: 0.08
Nodes (15): build_scenario(), _edge_metrics(), main(), one_step_ahead_w2_jko(), _plot(), _plot_matrices(), _require_jko_setup(), run_jko_data_generator() (+7 more)

### Community 8 - "Bayesian Destructive GRN"
Cohesion: 0.11
Nodes (14): build_model(), build_nn_transitions(), _build_nx_graph(), get_reference_A(), load_expression_by_time(), load_true_grns(), main(), plot_data_overview() (+6 more)

### Community 9 - "Archived Loss Diagnostics"
Cohesion: 0.09
Nodes (13): _decode(), generate_and_save(), main(), optimise_loss(), objective(), plot_loss_values(), plot_recovery(), plot_snapshot_scatter() (+5 more)

### Community 10 - "Algorithm 1 Main/Optimize"
Cohesion: 0.13
Nodes (14): main(), optimize(), _fake_mu(), _fake_snapshots(), test_main_initialises_theta_via_bip_before_optimizing(), test_main_preprocesses_every_snapshot(), test_main_returns_optimizes_result(), _chi() (+6 more)

### Community 11 - "Test Ground Truth Helpers"
Cohesion: 0.12
Nodes (13): draw_cross_section(), _euler_maruyama_step(), make_snapshots(), make_stationary_sim(), test_evolution_over_time_matches_euler_maruyama_ensemble_moments(), test_nu_star_mean_matches_ou_gene_expression_ensemble_mean(), _theta(), _source_cloud() (+5 more)

### Community 12 - "Wasserstein Comparison"
Cohesion: 0.16
Nodes (13): destructive_sample(), gaussian_sample(), kde_sample(), main(), marginal_w(), mu_tanh(), _plot_density_shapes(), _plot_kde_vs_gaussian() (+5 more)

### Community 13 - "Run Directory Housekeeping"
Cohesion: 0.09
Nodes (4): git_commit(), make_run_dir(), Readme, run_stamp()

### Community 14 - "OU Loss Checks"
Cohesion: 0.09
Nodes (9): get_true_params(), load_snapshots(), _mean_trajectory(), plot_A_heatmap(), plot_comparison(), plot_convergence(), plot_phase_portrait(), plot_recovery() (+1 more)

### Community 15 - "FP Cell Population (JKO)"
Cohesion: 0.14
Nodes (10): fp_cell_population(), fp_particles(), ou_gene_expression(), preprocessing(), Theta, test_output_is_nonnegative(), test_uses_mu_at_k_not_a_fixed_interval(), rollout() (+2 more)

### Community 16 - "Knockout Loss Ablation"
Cohesion: 0.16
Nodes (10): main(), grn_networks(), metrics_and_heatmaps(), _ordered(), build_dataset(), Dataset, run_one(), summary_line() (+2 more)

### Community 17 - "ComputeLoss Implementation"
Cohesion: 0.16
Nodes (12): _fp_propagate(), loss_cons(), loss_fp(), loss_ou(), _propagate(), _sliced_w2(), exact_fp_particles(), exact_ou_transition() (+4 more)

### Community 18 - "ComputeLoss Tests"
Cohesion: 0.25
Nodes (15): compute_loss(), _chi(), _snapshots(), test_compute_loss_all_components_nonneg_finite(), test_compute_loss_deterministic_given_same_seed(), test_compute_loss_returns_dict_with_expected_keys(), test_compute_loss_total_equals_sum_of_components(), test_loss_cons_is_nonneg_finite() (+7 more)

### Community 19 - "GRN Graph Plotting"
Cohesion: 0.17
Nodes (7): draw_grn(), off_diag_mask(), shared_layout(), to_digraph(), true_edge_set(), grn_networks(), metrics_and_heatmaps()

### Community 22 - "Intervention Fixtures"
Cohesion: 0.14
Nodes (4): effective_sigma(), make_constant_sim(), make_knockout_sim(), make_uncoupled_twin()

### Community 23 - "OU Expression Tests"
Cohesion: 0.29
Nodes (6): _analytic_mean(), test_converges_to_analytic_solution_as_substeps_increase(), test_ensemble_mean_tracks_analytic_mean_under_noise(), test_input_array_is_not_mutated(), test_output_shape_matches_input(), _theta()

### Community 24 - "Single-gene Knockout Sim"
Cohesion: 0.22
Nodes (4): main(), extract_snapshots(), make_single_gene_knockout_sim(), build_scenario()

### Community 25 - "CMA-ES Comparison"
Cohesion: 0.33
Nodes (4): _edge_metrics(), main(), _plot(), _plot_matrices()

### Community 26 - "Knockout Array Runner"
Cohesion: 0.22
Nodes (8): MKL_NUM_THREADS, MPLBACKEND, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, PATH, run_knockout_array.sh script, VECLIB_MAXIMUM_THREADS

### Community 27 - "Loss-at-Truth Check"
Cohesion: 0.33
Nodes (4): loss_at(), loss_ou_cons_at(), main(), _plot()

### Community 29 - "Loss Combination Comparison"
Cohesion: 0.38
Nodes (3): decode(), drift_field(), encode()

### Community 30 - "Loss Landscape Plot"
Cohesion: 0.52
Nodes (5): _l_fp(), _l_ou(), _l_ou_fp(), _objective(), _theta()

### Community 34 - "CI Workflow"
Cohesion: 0.50
Nodes (4): Codecov coverage upload, CI Workflow, CI test matrix (ubuntu/macos/windows x py3.11/3.13), uv run pytest with coverage on multsc_grn_inference

## Knowledge Gaps
- **17 isolated node(s):** `multsc-grn-inference`, `run_knockout_array.sh script`, `PATH`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 336 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Theta` connect `FP Cell Population (JKO)` to `BIP Initialize`, `Full Dynamics Comparison`, `Edge Recovery Metrics`, `Algorithm 1 Main/Optimize`, `Test Ground Truth Helpers`, `Knockout Loss Ablation`, `ComputeLoss Implementation`, `ComputeLoss Tests`, `Exact OU Transition`, `OU Expression Tests`, `Loss Combination Comparison`, `Loss Landscape Plot`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `Readme` connect `Run Directory Housekeeping` to `Edge Recovery Metrics`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `Exact OU transition vs Euler-Maruyama` connect `Diagnostic Findings (Docs)` to `Full Dynamics Comparison`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `Theta` (e.g. with `bip_initialize()` and `compute_loss()`) actually correct?**
  _`Theta` has 35 INFERRED edges - model-reasoned connections that need verification._
- **What connects `multsc-grn-inference`, `run_knockout_array.sh script`, `PATH` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Non-stationary Data Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.055482456140350876 - nodes in this community are weakly interconnected._
- **Should `Full Dynamics Comparison` be split into smaller, more focused modules?**
  _Cohesion score 0.06349206349206349 - nodes in this community are weakly interconnected._