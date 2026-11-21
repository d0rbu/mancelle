"""Reserve framed stdout before importing the persistent final evaluator."""

from scripts.eval.axbench_adv.vendored.bootstrap import serve_module

if __name__ == "__main__":
    raise SystemExit(serve_module("scripts.eval.axbench_adv.vendored.final_worker"))
