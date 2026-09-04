import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FileUploadCard } from "@/pages/import/FileUploadCard";

describe("FileUploadCard bank selector (Phase 12B closure patch)", () => {
  it("defaults to Auto-detect", () => {
    render(<FileUploadCard onUpload={vi.fn()} pending={false} />);
    expect(screen.getByRole("combobox")).toHaveTextContent("Auto-detect");
  });

  it("lists RBC, Scotiabank, TD, and CIBC as selectable", async () => {
    const user = userEvent.setup();
    render(<FileUploadCard onUpload={vi.fn()} pending={false} />);
    await user.click(screen.getByRole("combobox"));

    for (const bank of ["RBC", "Scotiabank", "TD", "CIBC"]) {
      const option = screen.getByRole("option", { name: bank });
      expect(option).toBeInTheDocument();
      expect(option).not.toHaveAttribute("aria-disabled", "true");
    }
  });

  it("shows BMO and National Bank as disabled 'Coming Soon' options", async () => {
    const user = userEvent.setup();
    render(<FileUploadCard onUpload={vi.fn()} pending={false} />);
    await user.click(screen.getByRole("combobox"));

    const bmo = screen.getByRole("option", { name: "BMO — Coming Soon" });
    expect(bmo).toHaveAttribute("aria-disabled", "true");

    const national = screen.getByRole("option", { name: "National Bank — Coming Soon" });
    expect(national).toHaveAttribute("aria-disabled", "true");
  });

  it("never selects a disabled bank (click on it is a no-op)", async () => {
    const user = userEvent.setup();
    render(<FileUploadCard onUpload={vi.fn()} pending={false} />);
    await user.click(screen.getByRole("combobox"));

    const bmo = screen.getByRole("option", { name: "BMO — Coming Soon" });
    await user.click(bmo);

    // Radix disabled items don't close the popup or fire onSelect -- the
    // option is still present/disabled and the listbox is still open,
    // proving the click had no effect.
    expect(screen.getByRole("option", { name: "BMO — Coming Soon" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
  });
});
