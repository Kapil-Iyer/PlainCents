import { useRef, useState } from "react";
import { Loader2, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { COMING_SOON_BANKS, SUPPORTED_BANKS } from "@/types/import";

interface FileUploadCardProps {
  onUpload: (file: File, bank: string) => void;
  pending: boolean;
}

export function FileUploadCard({ onUpload, pending }: FileUploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const [bank, setBank] = useState<string>("Auto");

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    setSelected(file);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Import a transaction CSV</CardTitle>
        <CardDescription>
          Upload a transaction CSV exported from your bank. You'll see a preview before anything is
          saved.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-border px-6 py-10 text-center transition-colors",
            dragActive && "border-primary bg-primary/5",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <UploadCloud className="h-8 w-8 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium">
              {selected ? selected.name : "Drag & drop your CSV here"}
            </p>
            <p className="text-xs text-muted-foreground">or</p>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        <div className="mt-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Bank</span>
            <Select value={bank} onValueChange={setBank}>
              <SelectTrigger className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Auto">Auto-detect</SelectItem>
                {SUPPORTED_BANKS.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
                {/* Visible for roadmap honesty, but disabled -- selecting one
                 * of these and only failing after upload is exactly what
                 * this patch removes (Phase 12B closure). */}
                {COMING_SOON_BANKS.map((name) => (
                  <SelectItem key={name} value={name} disabled>
                    {name} — Coming Soon
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button disabled={!selected || pending} onClick={() => selected && onUpload(selected, bank)}>
            {pending && <Loader2 className="h-4 w-4 animate-spin" />}
            Upload &amp; preview
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
