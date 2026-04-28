"""Population management: init, selection (tournament w/ elitism), evolution loop,
hall of fame, niching quotas, and parallel eval orchestration.
"""
from __future__ import annotations

import concurrent.futures as cf
import random
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .exceptions import FitnessError
from .fitness import FitnessResult, evaluate_individual_in_process, evaluate_individual_subprocess
from .fitness_cache import FitnessCache
from .genome import (
    Genome,
    crossover_uniform,
    mutate_genome,
    random_genome,
    sample_params,
)
from .logger import GALogger
from .registry import all_keys


# ---------- selection ----------

def tournament_select(
    ranked: list[tuple[Genome, float]],
    k: int,
    rng: random.Random,
) -> Genome:
    """k-way tournament selection. Higher fitness wins."""
    sample = rng.sample(ranked, min(k, len(ranked)))
    sample.sort(key=lambda x: x[1], reverse=True)
    return sample[0][0]


# ---------- initial population ----------

def initial_population(cfg: Config, rng: random.Random) -> list[Genome]:
    pool = cfg.ga.combo_pool or all_keys()
    # If stage2, lock to single combo
    if cfg.ga.fixed_combo is not None:
        fixed = cfg.ga.fixed_combo
        return [Genome(fixed, sample_params(fixed, rng))
                for _ in range(cfg.ga.population_size)]
    # Stage1: spread initial population across combo_pool
    pop: list[Genome] = []
    per_combo = max(1, cfg.ga.population_size // len(pool))
    for key in pool:
        for _ in range(per_combo):
            pop.append(Genome(key, sample_params(key, rng)))
            if len(pop) >= cfg.ga.population_size:
                return pop
    # Fill remainder with random
    while len(pop) < cfg.ga.population_size:
        pop.append(random_genome(rng, pool))
    return pop[:cfg.ga.population_size]


# ---------- evolution loop ----------

class GARunner:
    def __init__(
        self,
        cfg: Config,
        *,
        logger: GALogger,
        cache: FitnessCache,
        use_subprocess: bool = True,
    ):
        self.cfg = cfg
        self.logger = logger
        self.cache = cache
        self.use_subprocess = use_subprocess
        self.rng = random.Random(cfg.ga.seed)

        self.output_dir = cfg.runtime.output_dir
        self.generations_dir = self.output_dir / "generations"
        self.ckpt_path = self.output_dir / "ga_checkpoint.pkl"

        self.hall_of_fame: list[tuple[Genome, float, dict[str, float]]] = []

    # ---------- eval ----------
    def _work_dir_for(self, gen: int, idx: int, g: Genome) -> Path:
        return self.generations_dir / f"gen_{gen:03d}" / f"ind_{idx:03d}_{g.combo_key}_{g.stable_hash()}"

    def _pick_gpu(self, worker_idx: int) -> Optional[int]:
        rt = self.cfg.runtime
        if rt.per_worker_gpu:
            return rt.per_worker_gpu[worker_idx % len(rt.per_worker_gpu)]
        if rt.n_gpus > 0:
            return worker_idx % rt.n_gpus
        return None

    def _eval_one(self, gen: int, idx: int, genome: Genome, worker_idx: int = 0) -> FitnessResult:
        gh = genome.stable_hash()
        if gh in self.cache:
            rec = self.cache.get(gh)
            assert rec is not None
            self.logger.info(f"cache hit  {genome.short_label()} → fitness={rec['fitness']:.4f}")
            return FitnessResult(
                fitness=rec["fitness"], metrics=rec["metrics"],
                duration_s=rec.get("duration_s", 0.0),
                status="cached",
            )

        wd = self._work_dir_for(gen, idx, genome)
        if self.cfg.runtime.dry_run:
            fake = {"iou": self.rng.uniform(0.4, 0.9), "boundary_f1": self.rng.uniform(0.3, 0.8), "tv_penalty": 0.0}
            from .fitness import score_genome
            return FitnessResult(
                fitness=score_genome(genome, fake, self.cfg),
                metrics=fake, duration_s=0.1, status="ok",
                error=None, artifacts_dir=wd,
            )

        gpu = self._pick_gpu(worker_idx)
        if self.use_subprocess:
            result = evaluate_individual_subprocess(
                genome, self.cfg, work_dir=wd, gpu_id=gpu,
                timeout_minutes=self.cfg.runtime.timeout_per_individual_minutes,
            )
        else:
            result = evaluate_individual_in_process(genome, self.cfg, work_dir=wd, gpu_id=gpu)

        # Retry crashed individuals if budget allows
        retries = self.cfg.ga.max_retries_per_individual
        while result.status in ("crashed", "timeout") and retries > 0:
            self.logger.warn(f"retrying {genome.short_label()} (remaining={retries})")
            retries -= 1
            if self.use_subprocess:
                result = evaluate_individual_subprocess(
                    genome, self.cfg, work_dir=wd, gpu_id=gpu,
                    timeout_minutes=self.cfg.runtime.timeout_per_individual_minutes,
                )
            else:
                result = evaluate_individual_in_process(genome, self.cfg, work_dir=wd, gpu_id=gpu)

        # cache only successful runs (never cache failures)
        if result.status == "ok" and result.fitness > float("-inf"):
            self.cache.put(gh, {
                "combo_key": genome.combo_key,
                "fitness": result.fitness,
                "metrics": result.metrics,
                "duration_s": result.duration_s,
                "params": genome.params,
            })
        return result

    def _eval_population(self, gen: int, pop: list[Genome]) -> list[FitnessResult]:
        results: list[Optional[FitnessResult]] = [None] * len(pop)
        workers = max(1, self.cfg.runtime.parallel_workers)
        if workers == 1:
            for i, g in enumerate(pop):
                self.logger.info(f"gen {gen} ind {i}/{len(pop)}  eval  {g.short_label()}")
                r = self._eval_one(gen, i, g, worker_idx=0)
                results[i] = r
                self.logger.log_individual(
                    generation=gen, idx=i,
                    combo_key=g.combo_key, genome_hash=g.stable_hash(),
                    fitness=r.fitness, metrics=r.metrics,
                    duration_s=r.duration_s, status=r.status,
                )
        else:
            self.logger.info(f"gen {gen} parallel eval with {workers} workers")
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(self._eval_one, gen, i, g, i % workers): i
                    for i, g in enumerate(pop)
                }
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = FitnessResult(
                            fitness=float("-inf"), metrics={},
                            duration_s=0.0, status="crashed", error=str(e),
                        )
                    results[i] = r
                    g = pop[i]
                    self.logger.log_individual(
                        generation=gen, idx=i,
                        combo_key=g.combo_key, genome_hash=g.stable_hash(),
                        fitness=r.fitness, metrics=r.metrics,
                        duration_s=r.duration_s, status=r.status,
                    )
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    # ---------- evolution ----------
    def _evolve(self, ranked: list[tuple[Genome, float]]) -> list[Genome]:
        ga = self.cfg.ga
        rng = self.rng
        elites = [g for g, _ in ranked[:ga.elite_count]]
        children: list[Genome] = list(elites)
        while len(children) < ga.population_size:
            p1 = tournament_select(ranked, ga.tournament_k, rng)
            p2 = tournament_select(ranked, ga.tournament_k, rng)
            if rng.random() < ga.crossover_prob:
                c1, c2 = crossover_uniform(p1, p2, rng)
            else:
                c1, c2 = Genome(p1.combo_key, dict(p1.params)), Genome(p2.combo_key, dict(p2.params))
            if rng.random() < ga.mutation_prob:
                c1 = mutate_genome(c1, rng,
                                   combo_mutation_rate=ga.combo_mutation_rate if ga.fixed_combo is None else 0.0,
                                   param_mutation_rate=ga.param_mutation_rate,
                                   combo_pool=ga.combo_pool)
            if rng.random() < ga.mutation_prob:
                c2 = mutate_genome(c2, rng,
                                   combo_mutation_rate=ga.combo_mutation_rate if ga.fixed_combo is None else 0.0,
                                   param_mutation_rate=ga.param_mutation_rate,
                                   combo_pool=ga.combo_pool)
            children.append(c1)
            if len(children) < ga.population_size:
                children.append(c2)

        # Enforce niching quotas (preserve architectural diversity in stage1)
        if ga.niching_quota and ga.fixed_combo is None:
            children = self._enforce_quotas(children, ga.niching_quota)
        return children[:ga.population_size]

    def _enforce_quotas(self, children: list[Genome], quota: dict[str, int]) -> list[Genome]:
        """If a combo is over-represented, replace extras with random individuals
        from under-represented combos."""
        counts: Counter[str] = Counter(g.combo_key for g in children)
        pool = self.cfg.ga.combo_pool or all_keys()
        out: list[Genome] = []
        for g in children:
            limit = quota.get(g.combo_key, self.cfg.ga.population_size)
            if counts[g.combo_key] > limit:
                # demote: replace with random from under-quota combo
                under = [k for k in pool if counts[k] < quota.get(k, self.cfg.ga.population_size)]
                if under:
                    new_key = self.rng.choice(under)
                    g = Genome(new_key, sample_params(new_key, self.rng))
                    counts[new_key] += 1
                    counts[g.combo_key] = counts.get(g.combo_key, 0)
                else:
                    counts[g.combo_key] -= 1  # still over but nothing to swap to
            out.append(g)
        return out

    # ---------- main loop ----------
    def run(self) -> None:
        cfg = self.cfg
        # Resume?
        start_gen = 0
        if cfg.runtime.resume_from and cfg.runtime.resume_from.exists():
            self.logger.info(f"Resuming from {cfg.runtime.resume_from}")
            state = load_checkpoint(cfg.runtime.resume_from)
            self.rng.setstate(state["rng_state"])
            start_gen = state["generation"] + 1
            population_with_fit: list[tuple[Genome, float]] = state["population"]
            self.hall_of_fame = state["hall_of_fame"]
            pop = [g for g, _ in population_with_fit]
        else:
            pop = initial_population(cfg, self.rng)
            population_with_fit = []

        for gen in range(start_gen, cfg.ga.generations):
            self.logger.info(f"=== Generation {gen}/{cfg.ga.generations} ===")
            t0 = time.time()
            results = self._eval_population(gen, pop)
            fitnesses = [r.fitness for r in results]
            ranked: list[tuple[Genome, float]] = sorted(
                zip(pop, fitnesses), key=lambda x: x[1], reverse=True,
            )

            # Update HoF with top-3 of generation
            for r, (g, fit) in zip(
                sorted(zip(results, pop, fitnesses), key=lambda x: x[2], reverse=True)[:3],
                ranked[:3],
            ):
                res_obj, _, _ = r
                self.hall_of_fame.append((g, fit, res_obj.metrics))
            self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
            self.hall_of_fame = self.hall_of_fame[:20]

            best_g, best_fit = ranked[0]
            self.logger.log_generation_summary(gen, fitnesses, best_g)
            self.logger.info(f"gen {gen} best: {best_g.short_label()} fitness={best_fit:.4f} duration={time.time()-t0:.0f}s")

            # checkpoint
            if (gen + 1) % cfg.runtime.checkpoint_every_gen == 0:
                save_checkpoint(
                    self.ckpt_path,
                    generation=gen,
                    population=ranked,
                    rng=self.rng,
                    config_dict=cfg.model_dump(mode="json"),
                    hall_of_fame=self.hall_of_fame,
                )
                self.logger.info(f"saved checkpoint → {self.ckpt_path}")

            # Last gen: stop
            if gen + 1 >= cfg.ga.generations:
                break

            # Evolve
            pop = self._evolve(ranked)
            population_with_fit = ranked

        # Final report
        self._report_final()

    def _report_final(self) -> None:
        if not self.hall_of_fame:
            self.logger.warn("Hall of Fame is empty — no successful individuals.")
            return
        self.logger.info("=== Final Hall of Fame (top 10) ===")
        for i, (g, fit, met) in enumerate(self.hall_of_fame[:10]):
            self.logger.info(
                f"  {i+1:2d}. {g.combo_key:<32} fitness={fit:.4f}  "
                f"iou={met.get('iou', 0):.4f}  bf1={met.get('boundary_f1', 0):.4f}  "
                f"hash={g.stable_hash()}"
            )
        # persist as JSON
        import json
        out = self.output_dir / "hall_of_fame.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump([
                {"rank": i+1, "combo_key": g.combo_key, "params": g.params,
                 "fitness": fit, "metrics": met, "hash": g.stable_hash()}
                for i, (g, fit, met) in enumerate(self.hall_of_fame)
            ], f, indent=2)
        self.logger.info(f"Wrote {out}")
