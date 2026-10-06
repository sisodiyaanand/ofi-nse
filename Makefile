DAY ?= 2026-09-01
.PHONY: test bars ablation markout sweep final stress all
test:     ; pytest -q
bars:     ; python -m src.signals.ofi $(DAY)
ablation: ; python -m src.ablation
markout:  ; python -m src.markout $(DAY)
sweep:    ; python -m src.sim.sweep
final:    ; python -m src.sim.final
stress:   ; python -m src.sim.stress
all: test ablation sweep final stress
latency: ; python -m src.sim.latency
