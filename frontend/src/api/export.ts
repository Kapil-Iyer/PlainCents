import { ApiError, type ErrorResponse } from "@/types/common";

/**
 * Downloads the Power BI "current-state" export (PATCH D) and saves it via
 * the browser's normal download flow.
 *
 * Not built on `apiClient` (@/api/client.ts): every other endpoint returns
 * JSON, this one returns a binary ZIP, so the response needs to be read as
 * a Blob instead of parsed with `.json()`. Error handling still throws the
 * same ApiError shape as apiClient so callers can branch on `.status`/
 * `.error` identically to any other request.
 */
export async function downloadPowerBIExport(): Promise<void> {
  const res = await fetch("/api/export/powerbi");

  if (!res.ok) {
    let body: ErrorResponse;
    try {
      body = await res.json();
    } catch {
      body = { error: "unknown_error", message: res.statusText, details: {} };
    }
    throw new ApiError(res.status, body);
  }

  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="([^"]+)"/.exec(disposition);
  const filename = match?.[1] ?? "plaincents_export.zip";

  // The standard "invisible link click" download trick -- no library,
  // works the same way GET file downloads have always worked in a browser.
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
