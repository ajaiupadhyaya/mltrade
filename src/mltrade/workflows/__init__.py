"""MLTrade orchestration workflows.

Public API
----------
- :func:`~mltrade.workflows.demo.run_demo`         — offline demo (no network)
- :func:`~mltrade.workflows.research.run_research` — research on existing snapshot
- :func:`~mltrade.workflows.paper.run_paper`       — paper-trading (live broker)
- :class:`~mltrade.workflows.demo.DemoResult`
- :class:`~mltrade.workflows.research.ResearchResult`
- :class:`~mltrade.workflows.paper.PaperResult`
"""

from mltrade.workflows.demo import DemoResult, run_demo
from mltrade.workflows.paper import PaperResult, run_paper
from mltrade.workflows.research import ResearchResult, run_research

__all__ = [
    "DemoResult",
    "PaperResult",
    "ResearchResult",
    "run_demo",
    "run_paper",
    "run_research",
]
