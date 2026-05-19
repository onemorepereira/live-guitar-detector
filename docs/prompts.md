# Classifier Prompts

These are the zero-shot text prompts used by the MobileCLIP classifier. Edits
take effect on next worker restart. See DESIGN.md §5.4 for the design rationale
and the "Unknown" rejection-class pattern.

The loader at `services/inference-worker/app/prompts.py` extracts the YAML block
below directly.

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
  - text: "a photograph of a different electric guitar"
    brand: Unknown
    model: Unknown
```
