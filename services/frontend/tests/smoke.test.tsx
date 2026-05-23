import { render, screen } from "@testing-library/react";
import { App } from "../src/App";

describe("App", () => {
  it("renders the page heading", () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: /guitar detect/i }),
    ).toBeInTheDocument();
  });
});
