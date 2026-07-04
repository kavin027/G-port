PYTHON ?= python

TCP_BASELINE_ROOT ?= results/external_baselines_tcp_plain_full_best_safe
TCP_STRESS_ROOT ?= results/external_baselines_network_stress_full_best_safe
K3S_MAIN_ROOT ?= results/server_k3s_20260702/coded_k3s_external_full
K3S_BEST_SAFE_ROOT ?= results/server_k3s_20260702/coded_k3s_best_safe_full
K3S_STRESS_ROOT ?= results/server_k3s_20260702/coded_k3s_stress_full
K3S_RECOVERY_ROOT ?= results/server_k3s_20260702/coded_k3s_recovery_stress
PAPER_REPRO_ROOT ?= results/paper_reproduction
ARTIFACT_ROOT ?= submission
PAPER_DIR ?= paper/socc26

.PHONY: reproduce-paper-assets reproduce-k3s-main reproduce-majorrev-k3s reproduce-external-baselines reproduce-best-safe reproduce-stress reproduce-tables reproduce-figure1 reproduce-figure2 reproduce-figure3 reproduce-table1 reproduce-table2 reproduce-table3 reproduce-table4 reproduce-table5 reproduce-table6 reproduce-feature-ablation reproduce-guard-prediction compile-paper audit-anonymity

reproduce-paper-assets: reproduce-figure1 reproduce-figure2 reproduce-figure3 reproduce-table1 reproduce-table2 reproduce-table3 reproduce-table4 reproduce-table5

reproduce-figure1: compile-paper

reproduce-figure2:
	$(PYTHON) scripts/reproduce_paper_artifacts.py figure2

reproduce-figure3:
	$(PYTHON) scripts/reproduce_paper_artifacts.py figure3 --k3s-main-root $(K3S_MAIN_ROOT) --paper-dir $(PAPER_DIR)

reproduce-table1:
	$(PYTHON) scripts/reproduce_paper_artifacts.py table1 --tcp-root $(TCP_BASELINE_ROOT) --stress-root $(TCP_STRESS_ROOT) --k3s-main-root $(K3S_MAIN_ROOT) --k3s-best-safe-root $(K3S_BEST_SAFE_ROOT) --paper-repro-root $(PAPER_REPRO_ROOT)

reproduce-table2: reproduce-table1

reproduce-table3: reproduce-figure3

reproduce-table4:
	$(PYTHON) scripts/reproduce_paper_artifacts.py table4 --k3s-stress-root $(K3S_STRESS_ROOT) --k3s-recovery-root $(K3S_RECOVERY_ROOT)

reproduce-table5:
	$(PYTHON) scripts/reproduce_paper_artifacts.py table5

reproduce-k3s-main: reproduce-table1 reproduce-table3 reproduce-table4

reproduce-majorrev-k3s:
	$(PYTHON) analyze_majorrev_k8s.py --root $(K3S_MAIN_ROOT)
	$(PYTHON) analyze_majorrev_k8s.py --root $(K3S_BEST_SAFE_ROOT)

reproduce-feature-ablation:
	$(PYTHON) analyze_feature_ablation.py --root $(K3S_MAIN_ROOT) --out $(K3S_MAIN_ROOT)/feature_ablation_rebuild --label static-fallback-k3s

reproduce-external-baselines: reproduce-table1

reproduce-best-safe:
	$(PYTHON) analyze_majorrev_k8s.py --root $(K3S_BEST_SAFE_ROOT)

reproduce-table6: reproduce-table4

reproduce-stress: reproduce-table4

reproduce-guard-prediction: reproduce-figure3

reproduce-tables: reproduce-table1 reproduce-table2 reproduce-table3 reproduce-table4 reproduce-table5

compile-paper:
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=main_k3s_best main.tex

audit-anonymity:
	$(PYTHON) scripts/check_artifact_anonymity.py $(ARTIFACT_ROOT)
