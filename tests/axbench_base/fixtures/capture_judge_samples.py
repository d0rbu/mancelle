"""Capture real judge responses for the corpus-driven parser tests.

Run once (requires ``OPENROUTER_API_KEY``) to regenerate
``tests/fixtures/judge_response_samples.jsonl``. The script is committed
so the corpus can be rebuilt if the judge model / API changes. Do NOT
run as part of the regular test suite — it burns OpenRouter credits and
is slow.

Usage::

    OPENROUTER_API_KEY=sk-or-... uv run python \\
        tests/axbench_base/fixtures/capture_judge_samples.py

Output format (one JSON object per line)::

    {
      "id": "examples_001",
      "template": "judge_concept_examples",
      "description": "...",
      "examples": "...",
      "concept": null,
      "sentence": null,
      "raw_response": "...",
      "model": "openai/gpt-5-nano"
    }

``concept`` / ``sentence`` / ``description`` / ``examples`` are populated
only for the fields used by the corresponding template; the others are
``null``. The ``raw_response`` is the judge's verbatim content.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.templates import _safe_substitute, load_template

# Path to the committed corpus. The parser test loads from here.
FIXTURE_PATH = Path(__file__).parent / "judge_response_samples.jsonl"


# ---------------------------------------------------------------------------
# Hand-picked test inputs
#
# Each triple is designed to elicit a specific rating (0/1/2) from a
# well-behaved judge, which gives us a signal to cross-check parser
# correctness vs. model correctness. But the CORPUS we commit is
# anchored on the raw response text — not our expectation — so a
# surprise response still goes into the fixture and the test author
# (me) annotates the expected parse result after seeing the raw text.
# ---------------------------------------------------------------------------


_EXAMPLES_CASES: list[dict[str, str]] = [
    # Strong match — description and examples align on the same theme.
    {
        "description": "legal contract language about breach of warranty",
        "examples": (
            "Example 1:\n"
            "...the Seller hereby disclaims any warranty of merchantability "
            "and assumes no liability for breach thereof, whether express "
            "or implied, arising from the sale...\n"
            "---\n"
            "Example 2:\n"
            "...in the event of any material breach of warranty, Buyer's "
            "sole remedy shall be limited to repair or replacement of the "
            "defective goods, at Seller's election...\n"
            "---\n"
            "Example 3:\n"
            "...notwithstanding the foregoing, no warranty shall apply if "
            "the product has been subjected to misuse, neglect, or "
            "unauthorized modification...\n"
        ),
    },
    # Strong match — cooking / food preparation
    {
        "description": "cooking preparation instructions",
        "examples": (
            "Example 1:\n"
            "...preheat the oven to 375°F and lightly grease a 9x13 baking "
            "dish. In a large mixing bowl, whisk together the flour, sugar, "
            "baking powder, and salt...\n"
            "---\n"
            "Example 2:\n"
            "...sauté the diced onions over medium heat for about 5 minutes "
            "until translucent, then add the garlic and cook for another "
            "minute until fragrant...\n"
            "---\n"
            "Example 3:\n"
            "...bring a large pot of salted water to a rolling boil. Add the "
            "pasta and cook according to package directions, stirring "
            "occasionally to prevent sticking...\n"
        ),
    },
    # Partial match — description captures surface feature but not the theme.
    {
        "description": "words containing the letter Q",
        "examples": (
            "Example 1:\n"
            "The queen quickly questioned the quiet quartermaster about "
            "the quality of the quills.\n"
            "---\n"
            "Example 2:\n"
            "Quaint qualms quibbled in quaint quarters.\n"
            "---\n"
            "Example 3:\n"
            "Question marks and quotation marks appear in every "
            "quixotic query.\n"
        ),
    },
    # Weak / 1-rating — partial thematic overlap
    {
        "description": "historical events in Europe",
        "examples": (
            "Example 1:\n"
            "The signing of the Treaty of Westphalia in 1648 is often "
            "considered a pivotal moment in modern diplomacy...\n"
            "---\n"
            "Example 2:\n"
            "Napoleon's march on Moscow in 1812 ended in catastrophic "
            "retreat for the Grande Armée...\n"
            "---\n"
            "Example 3:\n"
            "The fall of the Berlin Wall in 1989 marked the symbolic end "
            "of the Cold War division of Europe.\n"
        ),
    },
    # Zero match — description is unrelated to examples
    {
        "description": "medical symptoms of appendicitis",
        "examples": (
            "Example 1:\n"
            "The quarterback threw a perfect spiral to the wide receiver "
            "in the corner of the end zone for a touchdown.\n"
            "---\n"
            "Example 2:\n"
            "Lake Superior is the largest of the Great Lakes by surface "
            "area, straddling the US-Canada border.\n"
            "---\n"
            "Example 3:\n"
            "Python's list comprehensions offer a concise syntax for "
            "building lists from iterables.\n"
        ),
    },
    # Zero match — description is a technical term; examples are casual
    {
        "description": "quaternion rotation mathematics",
        "examples": (
            "Example 1:\n"
            "I went to the grocery store yesterday and bought some milk, "
            "bread, and eggs for breakfast this morning.\n"
            "---\n"
            "Example 2:\n"
            "The weather has been lovely this week with sunny skies and "
            "warm temperatures every afternoon.\n"
            "---\n"
            "Example 3:\n"
            "My dog loves chasing tennis balls in the backyard after I "
            "come home from work.\n"
        ),
    },
    # Strong — programming / debugging
    {
        "description": "Python exception handling and traceback output",
        "examples": (
            "Example 1:\n"
            "...Traceback (most recent call last):\n"
            "  File \"app.py\", line 42, in <module>\n"
            "    result = divide(10, 0)\n"
            "ZeroDivisionError: division by zero...\n"
            "---\n"
            "Example 2:\n"
            "...try:\n"
            "    data = json.loads(response)\n"
            "except json.JSONDecodeError as exc:\n"
            "    logger.error(f\"failed to parse: {exc}\")...\n"
            "---\n"
            "Example 3:\n"
            "...except (KeyError, IndexError) as exc:\n"
            "    raise RuntimeError(\"lookup failed\") from exc...\n"
        ),
    },
    # Strong — music / instruments
    {
        "description": "classical orchestral music terminology",
        "examples": (
            "Example 1:\n"
            "The first movement is marked allegro ma non troppo and opens "
            "with a sweeping theme in the violins, echoed by the cellos...\n"
            "---\n"
            "Example 2:\n"
            "...the concerto features a lengthy cadenza at the end of the "
            "development section, giving the soloist a virtuosic platform "
            "before the recapitulation...\n"
            "---\n"
            "Example 3:\n"
            "...the string quartet performed Beethoven's Op. 132 with "
            "particular attention to the slow movement's Heiliger "
            "Dankgesang marking...\n"
        ),
    },
    # Partial — description is too narrow for the examples
    {
        "description": "German word 'und' meaning 'and'",
        "examples": (
            "Example 1:\n"
            "Ich gehe heute Abend ins Kino und dann essen wir zusammen.\n"
            "---\n"
            "Example 2:\n"
            "Der Hund rennt schnell durch den Park und jagt den Ball.\n"
            "---\n"
            "Example 3:\n"
            "Mathematik und Physik sind meine Lieblingsfächer in der "
            "Schule.\n"
        ),
    },
    # Strong — sports play-by-play
    {
        "description": "basketball play-by-play commentary",
        "examples": (
            "Example 1:\n"
            "...Curry brings the ball up the court, crosses over Irving, "
            "pulls up from 28 feet — BANG! Three-pointer good, "
            "Warriors by five!...\n"
            "---\n"
            "Example 2:\n"
            "...LeBron drives the lane, kicks it out to the corner, "
            "Bronny shoots for three — rattles out! Rebound to the "
            "Lakers, they push the pace...\n"
            "---\n"
            "Example 3:\n"
            "...Jokic with the ball in the post, backs down his defender, "
            "spins baseline and finger-rolls it in. What touch from the "
            "big man!...\n"
        ),
    },
]


_CONCEPT_RELEVANCE_CASES: list[dict[str, str]] = [
    {
        "concept": "medieval castle fortifications",
        "sentence": (
            "The massive stone keep rose above the moat, its crenellated "
            "battlements bristling with archers ready to repel any siege."
        ),
    },
    {
        "concept": "medieval castle fortifications",
        "sentence": (
            "I had pancakes for breakfast and then went to the coffee shop "
            "on the corner."
        ),
    },
    {
        "concept": "anxiety and nervous tension",
        "sentence": (
            "Her hands trembled slightly as she approached the podium, "
            "and she had to swallow hard to steady her voice."
        ),
    },
    {
        "concept": "anxiety and nervous tension",
        "sentence": (
            "The chef calmly plated the dish, sprinkled some parsley on "
            "top, and served it with a confident smile."
        ),
    },
    {
        "concept": "quantum physics concepts",
        "sentence": (
            "Schrödinger's equation describes how the wave function "
            "evolves over time in a closed quantum system."
        ),
    },
    {
        "concept": "quantum physics concepts",
        "sentence": (
            "The local farmer's market opens at 7am every Saturday during "
            "the summer months."
        ),
    },
    {
        "concept": "corporate restructuring announcements",
        "sentence": (
            "After the merger, the board announced layoffs affecting "
            "roughly 12% of the workforce and a consolidation of the "
            "three regional offices."
        ),
    },
    {
        "concept": "corporate restructuring announcements",
        "sentence": (
            "She wandered through the market stalls, pausing to admire the "
            "hand-painted ceramics."
        ),
    },
    {
        "concept": "French culinary techniques",
        "sentence": (
            "The beurre blanc needed to be emulsified carefully over low "
            "heat, whisking constantly to prevent the sauce from breaking."
        ),
    },
    {
        "concept": "French culinary techniques",
        "sentence": (
            "His laptop battery died right before the deadline, so he "
            "had to scramble to find a charger."
        ),
    },
]


_FLUENCY_CASES: list[str] = [
    # Fluent
    "The sun set slowly behind the mountains, casting long shadows across the valley.",
    # Fluent
    "She had always dreamed of visiting Paris, ever since she first read about it as a child.",
    # Awkward but comprehensible
    "He walking fast towards building while carrying many bag and big umbrella.",
    # Repetitive
    "The dog ran and the dog ran and the dog ran and the dog was running fast.",
    # Incoherent / broken
    "Purple electric sandwich freedom between below the tomorrow algorithm.",
    # Fluent
    "Although she had never studied chess formally, her intuitive grasp of positional play impressed even experienced competitors.",
    # Grammatically fine but semantically odd
    "Colorless green ideas sleep furiously under the unseen rainbow of tomorrow.",
    # Mid-fluency
    "The team won the game despite several injuries during the second half of play, celebrating at midcourt.",
    # Broken syntax
    "The went running store very quickly yesterday he to buy bread milk.",
    # Fluent, long
    (
        "After a long and challenging week at work, during which she had to "
        "navigate a series of increasingly complex technical problems, she "
        "finally allowed herself to relax with a cup of tea and a good book."
    ),
]


def _load_cfg() -> LLMClientConfig:
    """Build the same default LLM config the pipeline uses."""
    # Explicitly pin the model the production pipeline runs.
    # If the default ever changes, the fixture will need to be
    # regenerated and the corpus file replaced.
    return LLMClientConfig(
        provider="openrouter",
        model="openai/gpt-5-nano",
        requests_per_minute=60,
        max_concurrent=1,  # serial capture; don't race the API
    )


def _build_examples_prompt(case: dict[str, str]) -> str:
    template = load_template("judge_concept_examples.txt")
    return _safe_substitute(
        template,
        DESCRIPTION=case["description"],
        EXAMPLES=case["examples"],
    )


def _build_concept_relevance_prompt(case: dict[str, str]) -> str:
    template = load_template("judge_concept_relevance.txt")
    return _safe_substitute(
        template,
        CONCEPT=case["concept"],
        SENTENCE=case["sentence"],
    )


def _build_fluency_prompt(sentence: str) -> str:
    template = load_template("judge_fluency.txt")
    return _safe_substitute(template, SENTENCE=sentence)


def _ask(client: OpenRouterLLMClient, prompt: str) -> str:
    return client.complete([{"role": "user", "content": prompt}])


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY not set. Run `export OPENROUTER_API_KEY=...` "
            "and re-invoke. The capture script needs real API access."
        )

    cfg = _load_cfg()
    client = OpenRouterLLMClient(cfg, max_retries=5)
    rng = random.Random(42)

    rows: list[dict[str, Any]] = []

    print(f"Capturing {len(_EXAMPLES_CASES)} examples-QC responses...")
    for i, case in enumerate(_EXAMPLES_CASES):
        prompt = _build_examples_prompt(case)
        raw = _ask(client, prompt)
        rows.append({
            "id": f"examples_{i:03d}",
            "template": "judge_concept_examples",
            "description": case["description"],
            "examples": case["examples"],
            "concept": None,
            "sentence": None,
            "raw_response": raw,
            "model": cfg.model,
        })
        print(f"  [{i + 1}/{len(_EXAMPLES_CASES)}] {raw[:100]!r}")

    print(f"Capturing {len(_CONCEPT_RELEVANCE_CASES)} concept-relevance responses...")
    for i, case in enumerate(_CONCEPT_RELEVANCE_CASES):
        prompt = _build_concept_relevance_prompt(case)
        raw = _ask(client, prompt)
        rows.append({
            "id": f"relevance_{i:03d}",
            "template": "judge_concept_relevance",
            "description": None,
            "examples": None,
            "concept": case["concept"],
            "sentence": case["sentence"],
            "raw_response": raw,
            "model": cfg.model,
        })
        print(f"  [{i + 1}/{len(_CONCEPT_RELEVANCE_CASES)}] {raw[:100]!r}")

    print(f"Capturing {len(_FLUENCY_CASES)} fluency responses...")
    for i, sentence in enumerate(_FLUENCY_CASES):
        prompt = _build_fluency_prompt(sentence)
        raw = _ask(client, prompt)
        rows.append({
            "id": f"fluency_{i:03d}",
            "template": "judge_fluency",
            "description": None,
            "examples": None,
            "concept": None,
            "sentence": sentence,
            "raw_response": raw,
            "model": cfg.model,
        })
        print(f"  [{i + 1}/{len(_FLUENCY_CASES)}] {raw[:100]!r}")

    # Shuffle so the corpus isn't artificially ordered by expected score.
    rng.shuffle(rows)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} rows to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
