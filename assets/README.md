# Public workflow asset

`workflow.svg` illustrates the synthetic inference architecture without private
images, measured spectra or institutional branding. It is generated entirely by
the standard-library script below:

```bash
python assets/generate_workflow.py
python assets/generate_workflow.py --check
```

Both source and rendered SVG are version-controlled. The generator emits UTF-8
bytes with fixed ordering and no timestamp, ensuring byte-for-byte regeneration.
