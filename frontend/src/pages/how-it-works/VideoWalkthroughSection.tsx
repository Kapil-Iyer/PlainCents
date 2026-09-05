import { useEffect, useRef, useState } from "react";
import { Clapperboard, FileVideo } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";

/**
 * Where the walkthrough recording is expected to live. Both files are served
 * straight from `frontend/public/`, so dropping them in is the entire
 * install step — no build change, no code change, no config.
 *
 *   frontend/public/media/plaincents-walkthrough.mp4   (required)
 *   frontend/public/media/plaincents-walkthrough.jpg   (optional poster)
 *
 * Neither file is in the repository today, and this component does not
 * pretend otherwise: it probes for the video on mount and shows an explicit
 * "not recorded yet" state when it isn't there. A broken <video> element
 * with a dead source would look like a bug; an honest placeholder that names
 * the expected path looks like what it is — a slot waiting for a recording.
 */
const VIDEO_SRC = "/media/plaincents-walkthrough.mp4";
const POSTER_SRC = "/media/plaincents-walkthrough.jpg";

type Availability = "checking" | "available" | "missing";

const CHAPTERS = [
  "Empty state, and loading demo data",
  "Importing a real bank CSV and reading the preview",
  "Correcting a category, and seeing it reused on the next import",
  "Dashboard, category trends and merchant analysis",
  "Generating a forecast",
] as const;

export function VideoWalkthroughSection() {
  const [status, setStatus] = useState<Availability>("checking");
  const [hasPoster, setHasPoster] = useState(false);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;

    // HEAD rather than letting <video> fail: a dev server that returns
    // index.html for unknown paths (the SPA fallback this app uses in
    // packaged mode) would give the video element a 200 of HTML, which it
    // would report as a decode error rather than a missing file. Checking
    // the content type makes the distinction explicit.
    const probe = async (url: string) => {
      try {
        const res = await fetch(url, { method: "HEAD" });
        const type = res.headers.get("content-type") ?? "";
        return res.ok && !type.includes("text/html");
      } catch {
        return false;
      }
    };

    void (async () => {
      const [video, poster] = await Promise.all([probe(VIDEO_SRC), probe(POSTER_SRC)]);
      if (cancelled.current) return;
      setStatus(video ? "available" : "missing");
      setHasPoster(poster);
    })();

    return () => {
      cancelled.current = true;
    };
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Video walkthrough</h2>
        <p className="text-sm text-muted-foreground">
          A short recorded tour of the real interface, end to end.
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div className="overflow-hidden rounded-lg border border-border bg-elevated">
              {status === "available" ? (
                <video
                  className="aspect-video w-full"
                  controls
                  preload="metadata"
                  playsInline
                  poster={hasPoster ? POSTER_SRC : undefined}
                >
                  <source src={VIDEO_SRC} type="video/mp4" />
                  Your browser can&apos;t play embedded video. The recording is available at{" "}
                  {VIDEO_SRC}.
                </video>
              ) : (
                <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 px-6 text-center">
                  {status === "checking" ? (
                    <div
                      className="h-8 w-8 animate-pulse rounded-full bg-muted"
                      role="status"
                      aria-label="Checking for the walkthrough recording"
                    />
                  ) : (
                    <>
                      <FileVideo className="h-8 w-8 text-muted-foreground" aria-hidden />
                      <div className="flex flex-col gap-1">
                        <p className="text-sm font-medium">The walkthrough hasn&apos;t been recorded yet</p>
                        <p className="mx-auto max-w-sm text-sm text-muted-foreground">
                          This player is wired up and waiting. Drop an MP4 at the path below and it
                          appears here on the next page load — nothing else to change.
                        </p>
                      </div>
                      <code className="rounded bg-muted px-2 py-1 text-xs text-muted-foreground">
                        frontend/public{VIDEO_SRC}
                      </code>
                    </>
                  )}
                </div>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <Clapperboard className="h-4 w-4 text-primary" aria-hidden />
                <h3 className="text-sm font-semibold">What the recording covers</h3>
              </div>
              <ol className="flex flex-col gap-2">
                {CHAPTERS.map((chapter, i) => (
                  <li key={chapter} className="flex gap-2.5 text-sm text-muted-foreground">
                    <span
                      aria-hidden
                      className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-foreground"
                    >
                      {i + 1}
                    </span>
                    <span className="leading-relaxed">{chapter}</span>
                  </li>
                ))}
              </ol>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                The recording is served from this app, not from a video host — nothing on this page
                loads from a third party.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
