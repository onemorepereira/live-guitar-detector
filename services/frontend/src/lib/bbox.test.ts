import { describe, it, expect } from "vitest";
import { denormalizeBbox } from "./bbox";

describe("denormalizeBbox", () => {
  it("maps full-frame [0,0,1,1] to the element when aspects match", () => {
    const result = denormalizeBbox([0, 0, 1, 1], {
      videoW: 1920,
      videoH: 1080,
      elW: 960,
      elH: 540,
    });
    expect(result[0]).toBeCloseTo(0);
    expect(result[1]).toBeCloseTo(0);
    expect(result[2]).toBeCloseTo(960);
    expect(result[3]).toBeCloseTo(540);
  });

  it("letterbox: 16:9 video in a 1000x1000 element shifts vertically", () => {
    // scale = 1000 / 1920 ≈ 0.520833
    // renderedW = 1000, renderedH = 562.5
    // offsetX = 0, offsetY = (1000 - 562.5) / 2 = 218.75
    const result = denormalizeBbox([0, 0, 1, 1], {
      videoW: 1920,
      videoH: 1080,
      elW: 1000,
      elH: 1000,
    });
    expect(result[0]).toBeCloseTo(0);
    expect(result[1]).toBeCloseTo(218.75);
    expect(result[2]).toBeCloseTo(1000);
    expect(result[3]).toBeCloseTo(781.25);
  });

  it("pillarbox: 9:16 video in a 1000x1000 element shifts horizontally", () => {
    // scale = 1000 / 1920 ≈ 0.520833
    // renderedW = 562.5, renderedH = 1000
    // offsetX = (1000 - 562.5) / 2 = 218.75, offsetY = 0
    const result = denormalizeBbox([0, 0, 1, 1], {
      videoW: 1080,
      videoH: 1920,
      elW: 1000,
      elH: 1000,
    });
    expect(result[0]).toBeCloseTo(218.75);
    expect(result[1]).toBeCloseTo(0);
    expect(result[2]).toBeCloseTo(781.25);
    expect(result[3]).toBeCloseTo(1000);
  });

  it("midpoint bbox preserves its center under 1:1 mapping", () => {
    const result = denormalizeBbox([0.25, 0.25, 0.75, 0.75], {
      videoW: 1920,
      videoH: 1080,
      elW: 1920,
      elH: 1080,
    });
    expect(result[0]).toBeCloseTo(480);
    expect(result[1]).toBeCloseTo(270);
    expect(result[2]).toBeCloseTo(1440);
    expect(result[3]).toBeCloseTo(810);
  });

  it("returns floats, not rounded ints, for sub-pixel accuracy", () => {
    // 1000/1920 * 1920 = 1000 (exact), but a fractional bbox should yield
    // fractional output.
    const result = denormalizeBbox([0.1, 0.1, 0.3, 0.3], {
      videoW: 1920,
      videoH: 1080,
      elW: 1000,
      elH: 1000,
    });
    expect(result).toHaveLength(4);
    for (const v of result) {
      expect(typeof v).toBe("number");
    }
    // At least one coordinate should be non-integer to prove we are not
    // rounding internally.
    const hasFractional = result.some((v) => v !== Math.trunc(v));
    expect(hasFractional).toBe(true);
  });
});
