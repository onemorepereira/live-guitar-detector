# Classifier Prompts

These are the zero-shot text prompts used by the MobileCLIP classifier. Edits
take effect on next worker restart. See DESIGN.md §5.4 for the design rationale
and the "Unknown" rejection-class pattern.

The loader at `services/inference-worker/app/prompts.py` extracts the YAML block
below directly.

Schema: each entry is `{text, brand, model}` — all required, non-empty strings.

`brand` and `model` are used as identity keys; whitespace is trimmed but case
matters (`Gibson` and `gibson` are distinct classes).

> **Warning:** The loader extracts the **first** ```yaml fenced block in this
> file — do not put unrelated YAML before this one.

```yaml
prompts:
  - text: "a photograph of a Gibson Les Paul electric guitar"
    brand: Gibson
    model: Les Paul
  - text: "a photograph of a Gibson SG electric guitar"
    brand: Gibson
    model: SG
  - text: "a photograph of a Gibson Explorer electric guitar"
    brand: Gibson
    model: Explorer
  - text: "a photograph of a Gibson Flying V electric guitar"
    brand: Gibson
    model: Flying V
  - text: "a photograph of a Fender Stratocaster electric guitar"
    brand: Fender
    model: Stratocaster
  - text: "a photograph of a Fender Telecaster electric guitar"
    brand: Fender
    model: Telecaster
  - text: "a photograph of an acoustic guitar"
    brand: Unknown
    model: Unknown
  - text: "a photograph of a bass guitar"
    brand: Unknown
    model: Unknown
  # NOTE: removed "a photograph of a different electric guitar" rejection
  # prompt — manual testing showed it dominated real-target crops (~75% of
  # classifier outputs) and prevented vote convergence. Acoustic + bass are
  # sufficient rejection coverage for the typical use case.
```
