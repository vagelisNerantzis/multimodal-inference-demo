"""Regenerate the public workflow SVG deterministically using only stdlib.

Run from the repository root: python assets/generate_workflow.py
Check committed bytes: python assets/generate_workflow.py --check
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path

STEPS = (
    (30, 115, "Physical Sample", "", "Explicit identity and supplied observations"),
    (315, 115, "Available Modalities", "", "Selected numeric feature vectors"),
    (600, 115, "Modality Encoders", "", "Preprocess and fixed projection"),
    (885, 115, "Fusion", "", "Mean in common coordinates"),
    (885, 355, "Sample-Level", "Prediction", "Fixed synthetic class prototypes"),
    (600, 355, "Confidence /", "Entropy", "Uncalibrated model uncertainty"),
    (315, 355, "Next Modality", "Recommendation", "Supplied estimated gains minus cost"),
    (30, 355, "Scientific Report /", "Provenance", "Immutable reproducible run record"),
)


def render_svg() -> str:
    """Return stable, accessible SVG bytes represented as a newline-delimited str."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1165" height="590" viewBox="0 0 1165 590" role="img" aria-labelledby="title description">',
        '<title id="title">Synthetic physical-sample multimodal inference workflow</title>',
        '<desc id="description">Follow eight numbered steps: physical sample, available modalities, modality encoders, fusion, sample-level prediction, confidence and entropy, optional next modality recommendation, scientific report and provenance. A recommended observation can be explicitly supplied before rerunning inference.</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
        '<rect width="1165" height="590" rx="20" fill="#f8fafc"/>',
        '<g font-family="sans-serif" fill="#0f172a">',
        '<text x="30" y="43" font-size="25" font-weight="700">Multimodal scientific inference</text>',
        '<text x="30" y="73" font-size="15" fill="#475569">A synthetic integration demo: physical identity remains attached throughout the workflow.</text>',
    ]
    for index, (x, y, line1, line2, caption) in enumerate(STEPS, start=1):
        color = '#0f766e' if index in (1, 7, 8) else '#2563eb'
        parts.extend([
            f'<rect x="{x}" y="{y}" width="250" height="150" rx="12" fill="white" stroke="#cbd5e1"/>',
            f'<rect x="{x}" y="{y + 12}" width="4" height="126" rx="2" fill="{color}"/>',
            f'<text x="{x + 16}" y="{y + 25}" font-size="12" font-weight="700" fill="{color}">{index:02d}</text>',
            f'<text x="{x + 16}" y="{y + 61}" font-size="18" font-weight="600">{escape(line1)}</text>',
            f'<text x="{x + 16}" y="{y + 86}" font-size="18" font-weight="600">{escape(line2)}</text>',
            f'<text x="{x + 16}" y="{y + 126}" font-size="10.5" fill="#475569">{escape(caption)}</text>',
        ])
    arrows = (
        (287, 190, 306, 190), (572, 190, 591, 190), (857, 190, 876, 190),
        (1010, 273, 1010, 345),
        (877, 430, 858, 430), (592, 430, 573, 430), (307, 430, 288, 430),
    )
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    parts.extend([
        '<text x="30" y="548" font-size="13" fill="#475569">An optional recommendation adds a synthetic observation, then inference is rerun explicitly.</text>',
        '<text x="30" y="572" font-size="12" fill="#64748b">Generated data and fixed models illustrate mechanics; confidence is not calibrated scientific certainty.</text>',
        '</g>', '</svg>',
    ])
    return '\n'.join(parts) + '\n'


def main() -> None:
    """Write workflow.svg or verify its exact UTF-8 rendering with --check."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = Path(__file__).with_name('workflow.svg')
    rendered = render_svg().encode('utf-8')
    if args.check:
        if not output.exists() or output.read_bytes() != rendered:
            parser.exit(1, 'Workflow SVG differs; regenerate with python assets/generate_workflow.py\n')
        print('Workflow SVG matches its generator.')
    else:
        output.write_bytes(rendered)
        print('Generated workflow.svg')


if __name__ == '__main__':
    main()
